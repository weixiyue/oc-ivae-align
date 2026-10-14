"""Permanent validation for the sole decoder W-summary ablation."""
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from models.oc_ivae import OC_iVAE
from models.tcn import TCNDecoder
from experiments.verify_w_summary import verify_switch
from train import save_checkpoint, train_epoch


def test_switch_smoke_with_legacy_checkpoint(tmp_path: Path):
    raw=yaml.safe_load(Path('config/exp_tuning_mono_weight.yaml').read_text())
    model=OC_iVAE(dict(raw['model'],in_features=14))
    # Synthetic learned nonzero zd prior head, preserving the production zero init.
    with torch.no_grad():
        model.prior.zd_head.weight.fill_(.02)
    optimizer=torch.optim.AdamW(model.parameters(),lr=raw['training']['lr'],
                                weight_decay=raw['training']['weight_decay'])
    save_checkpoint(tmp_path/'new.pt',model,optimizer,raw,0,float('inf'))
    saved=torch.load(tmp_path/'new.pt',weights_only=False)
    legacy={key:saved[key] for key in ['model_state_dict','optimizer_state_dict','epoch','best_val_recon']}
    legacy['best_val_recon']=1.0
    torch.save(legacy,tmp_path/'legacy.pt')
    report=verify_switch(tmp_path/'smoke','4c38401d0a5fd0f2b536c26da93fd60d42d3e6c4',tmp_path/'legacy.pt')
    assert report['true_matches_original_forward']
    assert report['false_W_column_gradient_nonzeros']==0
    assert report['condition_posterior_and_both_priors_preserved']


def test_boolean_switch_and_equal_shapes():
    torch.manual_seed(42); default=TCNDecoder({})
    torch.manual_seed(42); explicit=TCNDecoder(dict(decoder_use_W_summary=True))
    torch.manual_seed(42); off=TCNDecoder(dict(decoder_use_W_summary=False))
    assert default.use_W_summary is True
    for key,value in default.state_dict().items():
        torch.testing.assert_close(value,explicit.state_dict()[key],rtol=0,atol=0)
        torch.testing.assert_close(value,off.state_dict()[key],rtol=0,atol=0)
    with pytest.raises(ValueError,match='boolean'):
        TCNDecoder(dict(decoder_use_W_summary='false'))


def test_nonfinite_training_stops_before_optimizer_step(tmp_path: Path):
    torch.set_num_threads(2)
    raw=yaml.safe_load(Path('config/exp_tuning_mono_weight.yaml').read_text())
    class NonfiniteModel(OC_iVAE):
        def forward(self,*args,**kwargs):
            out=super().forward(*args,**kwargs)
            out['x_hat']=out['x_hat']*float('nan')
            return out
    model=NonfiniteModel(dict(raw['model'],in_features=14))
    samples=[(torch.ones(8,14),torch.ones(8,dtype=torch.bool),torch.ones(11),torch.ones(8,4),
              torch.tensor(1),torch.tensor(i),torch.tensor(1.0)) for i in range(4)]
    train=DataLoader(samples,batch_size=4)
    optimizer=torch.optim.AdamW(model.parameters(),lr=raw['training']['lr'],
                                weight_decay=raw['training']['weight_decay'])
    settings={**raw['training'],**raw['loss_weights']}
    with pytest.raises(FloatingPointError,match='nonfinite training loss'):
        train_epoch(model,train,optimizer,0,settings,torch.device("cpu"))
    assert not optimizer.state


def test_existing_experiment_directory_is_untouched(tmp_path: Path, monkeypatch):
    import sys
    from experiments.run_w_summary_ablation import main
    root=tmp_path/'existing'; root.mkdir()
    marker=root/'existing_result.txt'; marker.write_text('keep')
    monkeypatch.setattr(sys,'argv',['run_w_summary_ablation.py','--output-dir',str(root)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code==2
    assert list(root.iterdir())==[marker]
    assert marker.read_text()=='keep'


def test_nonfinite_clipping_norm_stops_before_optimizer_step(tmp_path: Path):
    raw=yaml.safe_load(Path('config/exp_tuning_mono_weight.yaml').read_text())
    model=OC_iVAE(dict(raw['model'],in_features=14))
    for parameter in model.parameters():
        parameter.register_hook(lambda gradient: gradient*1e30)
    samples=[(torch.ones(8,14),torch.ones(8,dtype=torch.bool),torch.ones(11),torch.ones(8,4),
              torch.tensor(1),torch.tensor(i),torch.tensor(1.0)) for i in range(4)]
    loader=DataLoader(samples,batch_size=4)
    optimizer=torch.optim.AdamW(model.parameters(),lr=raw['training']['lr'],
                                weight_decay=raw['training']['weight_decay'])
    settings={**raw['training'],**raw['loss_weights']}
    with pytest.raises((RuntimeError,FloatingPointError),match='non-finite|nonfinite'):
        train_epoch(model,loader,optimizer,0,settings,torch.device("cpu"))
    assert not optimizer.state
