"""Verify the completed paired experiment without training or creating test weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

from eval.metrics import evaluate_all
from model.oc_ivae import OC_iVAE
from scripts.run_w_summary_ablation import seed_training, state_digest


def verify_artifacts(root: Path) -> dict[str, Any]:
    """Check initialization, actual random/batch rules, selection, scaling and metrics.

    Args:
        root: Completed formal paired experiment directory.
    Returns:
        Verification report, also saved in artifact_validation.json.
    """
    manifest=json.loads((root/'manifest.json').read_text())
    paired=json.loads((root/'paired_verification.json').read_text())
    initial=torch.load(root/'shared_initial_state.pt',map_location='cpu',weights_only=False)
    assert state_digest(initial['model_state_dict'])==manifest['initial_state_sha256']==paired['initial_state_sha256']
    assert initial['seed']==manifest['seed']==42
    assert json.loads((root/'started_runs.json').read_text())==['true','false']
    assert (root/'true/batch_sequences.jsonl').read_bytes()==(root/'false/batch_sequences.jsonl').read_bytes()
    histories={role:pd.read_csv(root/role/'training_metrics.csv',float_precision='round_trip') for role in ['true','false']}
    for key in ['rng_before_training','rng_after_training','rng_after_validation','batch_sha256','lr']:
        pd.testing.assert_series_equal(histories['true'][key],histories['false'][key])
    assert len(histories['true'])==len(histories['false'])==100
    configurations={role:yaml.safe_load((root/role/'config.yaml').read_text()) for role in ['true','false']}
    for section in ['data','training','loss_weights','postprocess','logging','experiment']:
        assert configurations['true'][section]==configurations['false'][section]
    off=dict(configurations['false']['model']); off['decoder_use_W_summary']=True
    assert off==configurations['true']['model']
    checkpoint_info={}; scaled_frames={}; metric_keys=['mae','mape','monotonicity','trendability','fc','mi']
    for role,enabled in [('true',True),('false',False)]:
        config=configurations[role]
        seed_training(42)
        model=OC_iVAE(config['model'])
        assert state_digest(model.state_dict())==manifest['initial_state_sha256']
        path=root/role/'checkpoints/best_model.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        assert checkpoint['decoder_use_W_summary'] is enabled
        assert checkpoint['model_config']['decoder_use_W_summary'] is enabled
        assert checkpoint['run_metadata']['seed']==42
        assert checkpoint['run_metadata']['initial_state_sha256']==manifest['initial_state_sha256']
        model.load_state_dict(checkpoint['model_state_dict'])
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,checkpoint['model_state_dict'][key],rtol=0,atol=0)
        history=histories[role]
        finite_fields=['recon','kl_d','kl_c','hsic','mono','spearman','total','val_recon','lr']
        assert np.isfinite(history[finite_fields]).all().all()
        selected=history.loc[history.val_recon.idxmin()]
        assert checkpoint['epoch']==int(selected.epoch)
        assert checkpoint['best_val_recon']==float(selected.val_recon)
        checkpoint_info[role]=dict(selected_epoch=checkpoint['epoch'],best_val_recon=checkpoint['best_val_recon'],
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        evaluation=root/role/'evaluation'
        frame=pd.read_csv(evaluation/'predictions.csv',float_precision='round_trip')
        assert frame.columns.tolist()==['split','unit','cycle','score','hi','hi_gt','rul']
        assert not frame.duplicated(['split','unit','cycle']).any()
        assert np.isfinite(frame[['unit','cycle','score','hi','hi_gt','rul']]).all().all()
        dev=frame[frame.split=='dev']; test=frame[frame.split=='test']
        assert len(dev)==663 and len(test)==438
        assert dev[['unit','cycle']].values.tolist()==sorted(dev[['unit','cycle']].values.tolist())
        assert test[['unit','cycle']].values.tolist()==sorted(test[['unit','cycle']].values.tolist())
        scaler_meta=json.loads((evaluation/'scaler.json').read_text())
        assert scaler_meta['fit_split']=='dev' and scaler_meta['n_samples_seen_']==663
        assert scaler_meta['fit_keys']==dev[['unit','cycle']].astype(int).values.tolist()
        assert scaler_meta['clip'] is False and scaler_meta['labels_used'] is False
        scaler=MinMaxScaler(clip=False).fit(dev[['score']].to_numpy())
        for key in ['data_min_','data_max_','data_range_','scale_','min_']:
            np.testing.assert_array_equal(getattr(scaler,key),scaler_meta[key])
        saved=json.loads((evaluation/'metrics.json').read_text())
        assert saved['metadata']['decoder_use_W_summary'] is enabled
        assert all(saved['metadata']['postprocessing'][key] is False for key in ['per_unit_scaling','ewma','smoothing','clipping'])
        assert saved['metadata']['mi_random_state']==42
        for split,subset in [('dev',dev),('test',test)]:
            np.testing.assert_allclose(subset.hi,scaler.transform(subset[['score']].to_numpy()).ravel(),rtol=0,atol=1e-14)
            recalculated=evaluate_all(subset,random_state=42)
            for key in metric_keys:
                assert recalculated[key] is not None and np.isfinite(recalculated[key])
                np.testing.assert_allclose(recalculated[key],saved[split][key],rtol=0,atol=1e-14)
        scaled_frames[role]=frame
    pd.testing.assert_frame_equal(scaled_frames['true'][['split','unit','cycle','hi_gt','rul']],
                                  scaled_frames['false'][['split','unit','cycle','hi_gt','rul']])
    source_hashes_verified=[]
    for source,expected in manifest['source_hashes'].items():
        # Retain the exact executed sources for post-run safety/type annotation updates.
        archived={'scripts/run_w_summary_ablation.py': root/'executed_run_w_summary_ablation.py',
                  'train/trainer.py': root/'executed_train_trainer.py'}
        source_path=archived.get(source,Path(source))
        if not source_path.exists():
            source_path=Path(source)
        assert hashlib.sha256(source_path.read_bytes()).hexdigest()==expected
        source_hashes_verified.append(source)
    historical_metrics=Path(manifest['historical']['evaluation'])/'metrics.json'
    historical=json.loads(historical_metrics.read_text())
    history_path=Path(historical['metadata']['checkpoint']['path'])
    assert hashlib.sha256(history_path.read_bytes()).hexdigest()==historical['metadata']['checkpoint']['sha256']
    history_rows=[]
    for split in ['dev','test']:
        for key in metric_keys:
            history_rows.append(dict(role='historical_uncontrolled_reference_only',split=split,metric=key,
                                     value=historical[split][key],checkpoint=str(history_path)))
    pd.DataFrame(history_rows).to_csv(root/'historical_reference.csv',index=False)
    curves=pd.read_csv(root/'curve_comparison.csv')
    curve_distance={split:float((g.n_cycles*g.mean_absolute_HI_difference).sum()/g.n_cycles.sum())
                    for split,g in curves.groupby('split')}
    report=dict(actual_training_runs=2,completed_epochs=paired['completed_epochs'],
                same_initialized_state_and_parameter_shapes=True,actual_batch_files_byte_identical=True,
                actual_all_100_epoch_rng_and_lr_sequences_identical=True,
                checkpoint_flags_and_weights_roundtrip_verified=True,
                selection_matches_minimum_validation_reconstruction=True,
                dev_only_scaling_and_transforms_reproduced=True,
                six_metrics_recomputed_from_exported_predictions=True,
                flight_keys_reference_HI_and_original_RUL_identical_across_pair=True,
                source_hashes_verified=source_hashes_verified,
                checkpoint_info=checkpoint_info,historical_checkpoint_unchanged=True,
                mean_absolute_curve_distance=curve_distance,
                nonfinite_training_or_evaluation_values=False,
                source_note='exact executed runner/trainer retained; post-run updates add CLI non-overwrite guard, iterator annotation and explicit nonfinite clipping-norm guard; finite-value training/evaluation math unchanged',
                unverified=['other seeds/datasets/hardware; intentionally outside this single-factor experiment',
                            'semantic legacy reference row order without explicit cycle ids (existing evaluation limitation)'])
    (root/'artifact_validation.json').write_text(json.dumps(report,indent=2))
    return report


def main() -> None:
    """Validate formal artifacts without rerunning training or inference."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment-dir',required=True)
    args=parser.parse_args()
    print(json.dumps(verify_artifacts(Path(args.experiment_dir)),indent=2))


if __name__=='__main__':
    main()
