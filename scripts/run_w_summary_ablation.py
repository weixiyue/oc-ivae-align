"""One seed-42 paired ablation of only the decoder W-summary input path."""
from __future__ import annotations

import argparse
from collections.abc import Iterator
import copy
import csv
import hashlib
import json
import os
import pickle
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

from data.dataset import NCMAPSSDataset
from data.sampler import UnitStratifiedSampler
from model.oc_ivae import OC_iVAE
from scripts.run_eval import run_evaluation, _resolve_main_h5_path
from scripts.verify_w_summary import verify_switch
from train.trainer import Trainer


def seed_training(seed: int) -> None:
    """Reset Python, NumPy and Torch RNGs and use identical deterministic rules.

    Args:
        seed: Sole experiment seed.
    Returns:
        None.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def state_digest(state: dict[str, torch.Tensor]) -> str:
    """Hash named tensor values to audit equal initialization independent of serialization.

    Args:
        state: Model state_dict.
    Returns:
        SHA256 of sorted names, shapes, dtypes and tensor bytes.
    """
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        digest.update(key.encode()); digest.update(str((tuple(value.shape),value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_digest() -> str:
    """Return an audit digest of Python, NumPy, CPU and available CUDA RNG states."""
    states=(random.getstate(), np.random.get_state(), torch.get_rng_state().numpy(),
            [s.cpu().numpy() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])
    return hashlib.sha256(pickle.dumps(states)).hexdigest()


class RecordedSampler(UnitStratifiedSampler):
    """Unchanged sampler with passive recording of actual flight batch sequences."""
    def __init__(self, dataset: NCMAPSSDataset, config: dict, trace_path: Path) -> None:
        super().__init__(dataset,k=config['k_units_per_batch'],m=config['m_cycles_per_unit'],random_seed=42)
        self.dataset=dataset; self.trace_path=trace_path; self.last_batches: list[list[int]]=[]

    def __iter__(self) -> Iterator[list[int]]:
        epoch=self.epoch_counter
        self.last_batches=[]
        for batch in super().__iter__():
            self.last_batches.append(batch)
            with self.trace_path.open('a') as output:
                output.write(json.dumps(dict(epoch_zero_based=epoch,indices=batch,
                                              flights=[self.dataset.samples[i] for i in batch]))+'\n')
            yield batch


class AuditedTrainer(Trainer):
    """Use Trainer's unchanged epoch/selection logic with structured loss and RNG audits."""
    def __init__(self, model: OC_iVAE, train_loader: DataLoader, val_loader: DataLoader,
                 config: dict, run_dir: Path, device: str) -> None:
        super().__init__(model,train_loader,val_loader,config,device)
        self.run_dir=run_dir; self.audit_rows: list[dict[str, Any]]=[]
        self._train_losses: dict[str,float]={}; self._start_rng=''; self._end_rng=''

    def train_epoch(self) -> dict:
        self._start_rng=rng_digest()
        values=super().train_epoch()
        self._end_rng=rng_digest()
        self._train_losses={k:float(v.detach().cpu()) if isinstance(v,torch.Tensor) else float(v) for k,v in values.items()}
        return values

    def val_epoch(self) -> dict:
        values=super().val_epoch()
        sampler=self.train_loader.batch_sampler
        row=dict(epoch=self.epoch,**self._train_losses,**values,
                 lr=self.optimizer.param_groups[0]['lr'],
                 best_val_before_epoch=self.best_val_recon if np.isfinite(self.best_val_recon) else None,
                 selected_by_val_recon=values['val_recon'] < self.best_val_recon,
                 rng_before_training=self._start_rng,rng_after_training=self._end_rng,
                 rng_after_validation=rng_digest(),
                 batch_sha256=hashlib.sha256(json.dumps(sampler.last_batches).encode()).hexdigest())
        self.audit_rows.append(row)
        with (self.run_dir/'training_metrics.csv').open('a',newline='') as output:
            writer=csv.DictWriter(output,fieldnames=list(row))
            if len(self.audit_rows)==1: writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row),flush=True)
        return values

    def save_checkpoint(self, path: str) -> None:
        # Metadata is passive; preserve the exact original best-val selection rule.
        self.run_metadata['epochs_recorded']=self.epoch
        self.run_metadata['rng_at_checkpoint']=rng_digest()
        super().save_checkpoint(path)


class Tee:
    """Mirror stdout to an independent run log without changing training."""
    def __init__(self, original: Any, file: Any) -> None:
        self.original=original; self.file=file
    def write(self, text: str) -> int:
        self.file.write(text); self.file.flush()
        return self.original.write(text)
    def flush(self) -> None:
        self.file.flush(); self.original.flush()


def run_pair(output: Path, original_revision: str, device: str) -> None:
    """Smoke, train exactly true/false once, evaluate and compare the sole ablation.

    Args:
        output: New, independent formal experiment root.
        original_revision: Revision for pre-switch smoke comparison.
        device: CPU or CUDA inference/training device.
    Returns:
        None; formal artifacts persist. Any nonfinite value aborts the pair.
    """
    output.mkdir(parents=True,exist_ok=False)
    smoke_dir=output/'_smoke'
    checks=verify_switch(smoke_dir,original_revision,PROJECT_ROOT/'checkpoints/exp_tuning_mono_weight/best_model.pt')
    (output/'verification.json').write_text(json.dumps(checks,indent=2))
    names={'true':'exp_tuning_mono_weight_w_summary_seed42','false':'exp_tuning_mono_weight_no_w_summary_seed42'}
    configs={role:yaml.safe_load((PROJECT_ROOT/f'configs/{name}.yaml').read_text()) for role,name in names.items()}
    off_copy=copy.deepcopy(configs['false']); off_copy['model']['decoder_use_W_summary']=True
    assert configs['true']==off_copy, 'paired configs differ beyond W-summary switch'
    baseline=yaml.safe_load((PROJECT_ROOT/'configs/exp_tuning_mono_weight.yaml').read_text())
    baseline['experiment']=dict(seed=42,paired_w_summary_ablation=True)
    assert configs['true']==baseline, 'control settings differ from specified baseline'
    data=configs['true']['data']
    h5=_resolve_main_h5_path(data)
    candidates=list(Path(h5).parent.glob('*cycle_dev.h5'))
    path=data.get('cycle_hi_path') or (str(candidates[0]) if len(candidates)==1 else None)
    seed_training(42)
    train=NCMAPSSDataset(h5,path,data,split='train',val_fraction=data['val_split'],random_seed=42)
    val=NCMAPSSDataset(h5,path,data,split='val',val_fraction=data['val_split'],random_seed=42)
    model_cfg=dict(configs['true']['model'],in_features=int(train[0][0].shape[-1]))
    seed_training(42)
    initial=OC_iVAE(model_cfg)
    shared_state={k:v.detach().clone().cpu() for k,v in initial.state_dict().items()}
    initial_digest=state_digest(shared_state)
    training_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    # Store actual initial weights and RNG states as formal (untracked) evidence.
    torch.save(dict(model_state_dict=shared_state,seed=42,initial_state_sha256=initial_digest,
                    torch_rng_state=training_rng,python_rng_state=random.getstate(),numpy_rng_state=np.random.get_state(),
                    cuda_rng_states=cuda_rng),output/'shared_initial_state.pt')
    historical=torch.load(PROJECT_ROOT/'checkpoints/exp_tuning_mono_weight/best_model.pt',map_location='cpu',weights_only=False)
    manifest=dict(seed=42,device=device,torch_version=torch.__version__,numpy_version=np.__version__,
                  initial_state_sha256=initial_digest,initialization_source='shared_initial_state.pt',
                  historical=dict(role='history_only',checkpoint_keys=list(historical),epoch=historical['epoch'],
                                  reason='no initialization seed/RNG/config in checkpoint; repeated runs share checkpoint path',
                                  evaluation='runs/exp_tuning_mono_weight/evaluation_unsupervisedhi_v1_20261014_verified'),
                  training_budget=configs['true']['training'],
                  source_revision=original_revision,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                      Path('model/decoder.py'),Path('train/trainer.py'),Path('scripts/run_w_summary_ablation.py'),Path('loss/total.py'),Path('data/dataset.py'),Path('data/sampler.py')]},
                  random_rules=dict(python_numpy_torch_seed=42,sampler='default_rng(42 + epoch_counter)',
                                    deterministic_algorithms=True,cudnn_benchmark=False,num_workers=0,
                                    cpu_threads=torch.get_num_threads(),batch_sequence='recorded actual unit/cycle keys',
                                    training_start_rng='shared post-initialization Torch state restored before each run'),
                  input_preprocessing='unchanged per-unit standardization; not aligned with reference input preprocessing')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    all_audits={}; summaries={}; started=[]
    for role in ['true','false']:
        run_dir=output/role; run_dir.mkdir()
        config=copy.deepcopy(configs[role]); config['model']['in_features']=model_cfg['in_features']
        config['checkpoint_dir']=str(run_dir/'checkpoints'); config['log_dir']=str(run_dir)
        config['run_metadata']=dict(seed=42,role=role,initial_state_sha256=initial_digest,
                                     shared_initial_state=str((output/'shared_initial_state.pt').resolve()),
                                     original_revision=original_revision,deterministic_algorithms=True,
                                     batch_trace=str((run_dir/'batch_sequences.jsonl').resolve()))
        cfg_path=run_dir/'config.yaml'; cfg_path.write_text(yaml.safe_dump(config,sort_keys=False))
        seed_training(42)
        model=OC_iVAE(config['model']); model.load_state_dict(shared_state)
        assert state_digest(model.state_dict())==initial_digest
        sampler=RecordedSampler(train,config['training'],run_dir/'batch_sequences.jsonl')
        train_loader=DataLoader(train,batch_sampler=sampler,num_workers=0)
        val_loader=DataLoader(val,batch_size=config['training']['k_units_per_batch']*config['training']['m_cycles_per_unit'],shuffle=False,num_workers=0)
        trainer=AuditedTrainer(model,train_loader,val_loader,config,run_dir,device)
        torch.set_rng_state(training_rng)
        if cuda_rng: torch.cuda.set_rng_state_all(cuda_rng)
        started.append(role)
        (output/'started_runs.json').write_text(json.dumps(started))
        with (run_dir/'stdout.log').open('w') as log:
            original_stdout=sys.stdout; sys.stdout=Tee(original_stdout,log)
            try:
                print(f'Paired {role}: seed=42, initial_state={initial_digest}',flush=True)
                trainer.train()
            finally:
                sys.stdout=original_stdout
        all_audits[role]=trainer.audit_rows
        summary=dict(epochs_completed=trainer.epoch,best_val_recon=trainer.best_val_recon,
                     best_checkpoint=str(Path(trainer.best_checkpoint_path).resolve()),
                     initial_state_sha256=initial_digest,decoder_use_W_summary=config['model']['decoder_use_W_summary'])
        (run_dir/'training_summary.json').write_text(json.dumps(summary,indent=2))
        # Evaluation starts only after train/val selection is finished; test never informs selection.
        eval_dir=run_evaluation(str(cfg_path),trainer.best_checkpoint_path,str(run_dir/'evaluation'),device)
        metrics=json.loads((eval_dir/'metrics.json').read_text())
        for split in ['dev','test']:
            for key in ['mae','mape','monotonicity','trendability','fc','mi']:
                if metrics[split][key] is None or not np.isfinite(metrics[split][key]):
                    raise FloatingPointError(f'nonfinite {role}/{split}/{key}; stopping pair')
        summaries[role]=dict(training=summary,evaluation=metrics)
        # Release optimizer and model graph before constructing the next run.
        del trainer,model,train_loader,val_loader
    common=min(len(all_audits['true']),len(all_audits['false']))
    for index in range(common):
        for key in ['rng_before_training','rng_after_training','rng_after_validation','batch_sha256','lr']:
            assert all_audits['true'][index][key]==all_audits['false'][index][key], f'paired audit mismatch epoch {index+1}: {key}'
    paired_verification=dict(common_epochs_compared=common,initial_state_sha256=initial_digest,
                             actual_batch_and_rng_sequences_identical=True,lr_sequences_identical=True,
                             identical_max_epoch_and_early_stop_settings=True,
                             completed_epochs={role:len(rows) for role,rows in all_audits.items()})
    (output/'paired_verification.json').write_text(json.dumps(paired_verification,indent=2))
    rows=[]
    for split in ['dev','test']:
        for metric in ['mae','mape','monotonicity','trendability','fc','mi']:
            on=summaries['true']['evaluation'][split][metric]; off=summaries['false']['evaluation'][split][metric]
            rows.append(dict(split=split,metric=metric,true=on,false=off,false_minus_true=off-on))
    with (output/'comparison.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/'comparison.json').write_text(json.dumps(dict(paired_results=rows,training={k:v['training'] for k,v in summaries.items()},
                                                          historical_reference=manifest['historical']),indent=2))
    _plot_comparison(output)
    print('Pair complete:',output,flush=True)


def _plot_comparison(output: Path) -> None:
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    on=pd.read_csv(output/'true/evaluation/predictions.csv')
    off=pd.read_csv(output/'false/evaluation/predictions.csv')
    joined=on.merge(off,on=['split','unit','cycle'],suffixes=('_true','_false'),validate='one_to_one')
    curve_rows=[]
    for split,group in joined.groupby('split'):
        units=list(group.unit.unique()); nrows=(len(units)+2)//3
        fig,axes=plt.subplots(nrows,3,figsize=(15,3.5*nrows),squeeze=False)
        for ax,unit in zip(axes.ravel(),units):
            g=group[group.unit==unit].sort_values('cycle')
            ax.plot(g.cycle,g.hi_true,label='W summary: true')
            ax.plot(g.cycle,g.hi_false,label='W summary: false')
            ax.plot(g.cycle,g.hi_gt_true,'--',label='Reference HI')
            ax.set(title=f'{split} unit {unit}',xlabel='Actual cycle',ylabel='HI'); ax.legend(fontsize=8)
            curve_rows.append(dict(split=split,unit=int(unit),n_cycles=len(g),
                                   mean_absolute_HI_difference=float(abs(g.hi_false-g.hi_true).mean()),
                                   true_first=float(g.hi_true.iloc[0]),true_last=float(g.hi_true.iloc[-1]),
                                   false_first=float(g.hi_false.iloc[0]),false_last=float(g.hi_false.iloc[-1])))
        for ax in axes.ravel()[len(units):]: ax.set_visible(False)
        fig.tight_layout(); fig.savefig(output/f'comparison_curves_{split}.png',dpi=150); plt.close(fig)
    with (output/'curve_comparison.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(curve_rows[0])); writer.writeheader(); writer.writerows(curve_rows)


def main() -> None:
    """Execute only the authorized paired experiment and stop on any failure."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--original-revision',default='4c38401d0a5fd0f2b536c26da93fd60d42d3e6c4')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    output=Path(args.output_dir)
    if output.exists():
        parser.error(f'experiment directory already exists: {output}')
    try:
        run_pair(output,args.original_revision,args.device)
    except Exception as exc:
        if output.exists():
            (output/'failure.json').write_text(json.dumps(dict(error=str(exc),traceback=traceback.format_exc()),indent=2))
        raise


if __name__=='__main__':
    main()
