"""Hand-calculated and interface regression checks for the HI protocol."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.feature_selection import mutual_info_regression

from eval.inference import extract_hi
from eval.metrics import evaluate_all, monotonicity, trendability
from eval.protocol import associate, read_reference, read_rul, scale_dev_test
from models.oc_ivae import OC_iVAE


def example():
    hi = np.array([1, .5, .5, 0, .8, .65, .5, .35, .2])
    gt = hi.copy(); gt[4:] += .2
    return pd.DataFrame(dict(unit=[1]*4+[2]*5, cycle=[1,2,4,9,1,3,5,7,11],
                             hi=hi, hi_gt=gt, rul=[9,6,2,0,11,7,5,3,0]))


def test_six_hand_calculated_metrics():
    f = example(); result = evaluate_all(f)
    assert result['mae'] == pytest.approx(1 / 9)  # cycle weighted, not unit mean 0.1
    expected_mape = sum(.2 / x for x in [1,.85,.7,.55,.4]) / 8 * 100
    assert result['mape'] == pytest.approx(expected_mape)
    assert result['mape_valid_count'] == 8
    assert result['monotonicity'] == 1  # zero difference counts as nonpositive
    assert result['trendability'] == pytest.approx((np.sqrt(.9) + 1) / 2)
    assert result['fc_denominator'] == pytest.approx(.8)
    assert result['fc'] == pytest.approx(np.exp(-.1 / .8))
    expected = []
    for _, g in f.groupby('unit'):
        raw = mutual_info_regression(g.hi.to_numpy()[:,None], g.rul, random_state=42)[0]
        expected.append(1-np.exp(-raw))
    assert result['mi'] == pytest.approx(np.mean(expected))
    assert all('fc' not in v for v in result['per_unit'].values())
    assert result == evaluate_all(f.sample(frac=1, random_state=4))


def test_signed_trends_cancel_and_monotonicity_is_absolute():
    f = example()
    f.loc[f.unit == 1, 'hi'] = [1,.75,.5,0]
    f.loc[f.unit == 2, 'hi'] = [0,.2,.5,.8,1]
    assert evaluate_all(f)['trendability'] == pytest.approx(0, abs=1e-15)
    assert monotonicity(np.array([0,1,2,3])) == 1
    assert monotonicity(np.array([1,1,0,1])) == pytest.approx(1/3)
    assert trendability(np.ones(4), np.array([1,3,8,9])) == 0


def test_diagnostics_without_epsilon_or_formula_clipping():
    f = example(); f['hi'] = 1
    r = evaluate_all(f)
    assert r['fc'] is None and 'fc' in r['issues']
    assert r['monotonicity'] == 1 and r['trendability'] == 0
    assert 'constant_output' in r['issues']
    f = example(); f['hi'] *= -1
    r = evaluate_all(f)
    assert r['fc'] > 1 and 'fc_denominator' in r['issues']
    f['hi_gt'] = np.nan
    r = evaluate_all(f)
    assert r['mae'] is None and r['mape'] is None
    f['hi_gt'] = 0
    assert evaluate_all(f)['mape_valid_count'] == 0
    short = example().iloc[:1]
    r = evaluate_all(short)
    assert r['mi'] is None and r['monotonicity'] is None
    f = example(); f.loc[0,'hi_gt'] = np.nan
    with pytest.raises(ValueError, match='partial'):
        evaluate_all(f)


def test_dev_only_scaling_without_smoothing_or_per_unit_scaling():
    dev = pd.DataFrame(dict(unit=[1,1,2,2],cycle=[1,2,1,2],score=[4,2,3,2.5]))
    test = pd.DataFrame(dict(unit=[3,3],cycle=[1,2],score=[6,0]))
    d,t,meta = scale_dev_test(dev,test)
    np.testing.assert_array_equal(d.hi,[1,0,.5,.25])
    np.testing.assert_array_equal(t.hi,[2,-1])
    assert meta['n_samples_seen_'] == 4 and meta['data_min_'] == [2]
    changed = test.copy(); changed['score'] = [1e8,-1e8]
    d2,_,m2 = scale_dev_test(dev,changed)
    pd.testing.assert_frame_equal(d,d2); assert meta == m2
    dev['score'] = 3
    d,t,meta = scale_dev_test(dev,test)
    assert meta['constant_dev_output'] and meta['scale_'] == [1]
    np.testing.assert_array_equal(d.hi,[0,0,0,0])
    np.testing.assert_array_equal(t.hi,[3,-3])


def test_key_join_not_position_and_window_mean():
    pred = {1:dict(cycle_ids=[2,1],hi_raw=[.2,.8])}
    rul = pd.DataFrame(dict(unit=[1,1],cycle=[1,2],rul=[9,0]))
    ref = pd.DataFrame(dict(unit=[1,1],cycle=[2,1],hi_gt=[0,1]))
    f = associate(pred,rul,ref)
    np.testing.assert_array_equal(f.hi_gt,[1,0])
    np.testing.assert_array_equal(f.rul,[9,0])
    raw = np.array([.23456789, .81234567], dtype=np.float32)
    precise = associate({1:dict(cycle_ids=[2,1],hi_raw=raw)},rul,ref)
    assert precise.score.dtype == np.float64
    np.testing.assert_array_equal(precise.score,raw[::-1].astype(np.float64))
    pred = {1:dict(cycle_ids=[1,1,2],hi_raw=[1,.6,.2])}
    with pytest.raises(ValueError,match='duplicate'):
        associate(pred,rul,ref)
    np.testing.assert_allclose(associate(pred,rul,ref,True).score,[.8,.2])
    with pytest.raises(ValueError,match='missing or extra'):
        associate({1:dict(cycle_ids=[1],hi_raw=[1])},rul,ref)
    with pytest.raises(ValueError,match='duplicate'):
        associate({1:dict(cycle_ids=[1,2],hi_raw=[1,0])},rul,pd.concat([ref,ref]))
    with pytest.raises(ValueError,match='nonfinite'):
        associate({1:dict(cycle_ids=[1,2],hi_raw=[1,np.inf])},rul,ref)


def test_original_rul_constant_within_flight(tmp_path: Path):
    p = tmp_path/'raw.h5'
    with h5py.File(p,'w') as f:
        f['A_dev'] = [[1,2,1,0],[1,1,1,1],[1,1,1,1]]
        f['Y_dev'] = [[0],[9],[9]]
    r = read_rul(str(p),'dev')
    np.testing.assert_array_equal(r.rul,[9,0])
    with h5py.File(p,'r+') as f:
        f['Y_dev'][2,0] = 8
    with pytest.raises(ValueError,match='not constant'):
        read_rul(str(p),'dev')


def test_reference_explicit_and_legacy_keys(tmp_path: Path):
    keys = pd.DataFrame(dict(unit=[1,1],cycle=[1,2]))
    p = tmp_path/'ref.h5'
    pd.DataFrame(dict(unit=[1,1],cycle=[2,1],HI=[0,1])).to_hdf(p,key='df')
    ref,reason = read_reference(str(p),keys)
    assert reason.startswith('explicit') and ref.cycle.tolist() == [2,1]
    pd.DataFrame(dict(unit=[1,1],HI=[1,0])).to_hdf(p,key='df',mode='w')
    ref,reason = read_reference(str(p),keys)
    assert ref.cycle.tolist() == [1,2] and 'ordinal' in reason
    with pytest.raises(ValueError,match='contiguous'):
        read_reference(str(p),keys.assign(cycle=[1,3]))
    pd.DataFrame(dict(unit=[1,1],HI=[1,0]),index=[1,0]).to_hdf(p,key='df',mode='w')
    with pytest.raises(ValueError,match='index'):
        read_reference(str(p),keys)


def test_raw_score_is_posterior_mean_and_no_sigmoid():
    torch.manual_seed(42)
    model = OC_iVAE(dict(in_features=14)).eval()
    x=torch.randn(2,8,14); mask=torch.ones(2,8,dtype=torch.bool)
    u=torch.randn(2,11); w=torch.randn(2,8,4)
    with torch.no_grad():
        first=model(x,mask,u,w); second=model(x,mask,u,w)
    torch.testing.assert_close(first['z_d'], first['mu_d'], rtol=0,atol=0)
    torch.testing.assert_close(first['hi_raw'],model.hi_head(first['mu_d']))
    torch.testing.assert_close(first['hi_raw'], second['hi_raw'],rtol=0,atol=0)
    class Flights:
        def __len__(self): return 2
        def __getitem__(self,i):
            return x[i],mask[i],u[i],w[i],torch.tensor(1),torch.tensor(i+1),torch.tensor(float('nan'))
    extracted=extract_hi(model,Flights())
    np.testing.assert_array_equal(extracted[1]['hi_raw'],first['hi_raw'].numpy())


@pytest.mark.parametrize('wrapped', [False, True])
def test_unified_entrypoint_old_checkpoint_and_artifacts(tmp_path: Path, monkeypatch, wrapped):
    import json
    import yaml
    import scripts.run_eval as entry
    from data.dataset import NCMAPSSDataset
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(dict(data=dict(h5_path='mock.h5',val_split=.2),model={})))
    checkpoint = tmp_path/'old.pt'
    model = torch.nn.Linear(1,1)
    torch.save(dict(model_state_dict=model.state_dict(), epoch=7) if wrapped else model.state_dict(), checkpoint)
    class Dataset:
        _split_units = staticmethod(NCMAPSSDataset._split_units)
        def __init__(self, h5_path, path, flat, split, **kwargs):
            self.split=split; self.samples=[(1,i) for i in range(1,5)]; self.L_max=8
        def __getitem__(self,index): return (torch.ones(8,14),)
        def get_train_stats(self): return {}
    monkeypatch.setattr(entry,'NCMAPSSDataset',Dataset)
    monkeypatch.setattr(entry,'_resolve_main_h5_path',lambda _: 'mock.h5')
    monkeypatch.setattr(entry,'_resolve_cycle_hi_path',lambda *args: None)
    monkeypatch.setattr(entry,'build_model',lambda *args: torch.nn.Linear(1,1))
    monkeypatch.setattr(entry,'extract_hi',lambda m,ds,device: {1:dict(cycle_ids=[1,2,3,4],hi_raw=[4,3,2,1] if ds.split=='train' else [5,3,1,0])})
    monkeypatch.setattr(entry,'read_rul',lambda *args: pd.DataFrame(dict(unit=[1]*4,cycle=[1,2,3,4],rul=[4,3,2,1])))
    monkeypatch.setattr(entry,'_plot_curves',lambda f,p: p.write_text('plot placeholder'))
    out=entry.run_evaluation(str(config),str(checkpoint),str(tmp_path/'evaluation'),'cpu')
    saved=json.loads((out/'metrics.json').read_text())
    assert saved['dev']['mae'] is None
    assert saved['metadata']['scaling']['n_samples_seen_'] == 4
    assert saved['metadata']['mi_random_state'] == 42
    assert saved['metadata']['evaluated_splits'] == ['dev','test']
    assert set(p.name for p in out.iterdir()) == {'metrics.json','scaler.json','evaluation_config.yaml','predictions.csv','hi_curves_dev.png','hi_curves_test.png'}
    pred=pd.read_csv(out/'predictions.csv')
    assert pred.columns.tolist() == ['split','unit','cycle','score','hi','hi_gt','rul']
    assert pred[pred.split=='test'].hi.min() < 0 and pred[pred.split=='test'].hi.max() > 1
    np.testing.assert_allclose(pred[pred.split=='dev'].hi, [1,2/3,1/3,0], rtol=0, atol=1e-15)
    assert torch.load(checkpoint,weights_only=False).keys() == (dict(model_state_dict={},epoch=0).keys() if wrapped else model.state_dict().keys())
    with pytest.raises(FileExistsError):
        entry.run_evaluation(str(config),str(checkpoint),str(out),'cpu')
