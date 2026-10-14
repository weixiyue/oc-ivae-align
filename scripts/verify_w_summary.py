"""Independent smoke verification for the decoder-only W-summary switch."""
from __future__ import annotations

import argparse
import ast
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sampler import UnitStratifiedSampler
from models.tcn import TCNDecoder
from models.oc_ivae import OC_iVAE
from train.trainer import Trainer


def verify_switch(work_dir: Path, original_revision: str, checkpoint: Path) -> dict[str, Any]:
    """Verify isolation, legacy behavior, checkpoint IO and paired random rules.

    Args:
        work_dir: New temporary directory for smoke checkpoint artifacts.
        original_revision: Git revision before decoder switch was connected.
        checkpoint: Existing historical checkpoint, read only.
    Returns:
        Passed assertions and measured differences for the formal experiment record.
    """
    work_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    raw = yaml.safe_load((PROJECT_ROOT/'configs/exp_tuning_mono_weight.yaml').read_text())
    cfg = dict(raw['model'], in_features=14)
    model_true = OC_iVAE(cfg).eval()
    model_false = OC_iVAE(dict(cfg, decoder_use_W_summary=False)).eval()
    assert model_true.state_dict().keys() == model_false.state_dict().keys()
    model_false.load_state_dict(model_true.state_dict())
    assert sum(p.numel() for p in model_true.parameters()) == sum(p.numel() for p in model_false.parameters())
    # Load the historical decoder and its dependencies from the same revision.
    # Resolve those imports locally, without old-path compatibility shims.
    namespace: dict[str, Any] = {'__name__': 'original_decoder_smoke'}
    for path in ['model/constants.py', 'model/backbone.py', 'model/decoder.py']:
        source = subprocess.check_output(
            ['git', 'show', f'{original_revision}:{path}'], cwd=PROJECT_ROOT, text=True)
        tree = ast.parse(source, filename=path)
        tree.body = [node for node in tree.body
                     if not (isinstance(node, ast.ImportFrom)
                             and node.module in {'model.constants', 'model.backbone'})]
        exec(compile(tree, path, 'exec'), namespace)
    old = namespace['TCNDecoder'](cfg).eval()
    old.load_state_dict(model_true.decoder.state_dict())
    x=torch.randn(3,16,14); mask=torch.ones(3,16,dtype=torch.bool); mask[0,:3]=False
    w=torch.randn(3,16,4); u=torch.randn(3,11)
    d=torch.randn(3,2); c=torch.randn(3,8)
    with torch.no_grad():
        before = old(d,c,w,mask)
        after = model_true.decoder(d,c,w,mask)
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        w_changed=w*1.7+2
        off_first=model_false.decoder(d,c,w,mask)
        off_changed=model_false.decoder(d,c,w_changed,mask)
        torch.testing.assert_close(off_first,off_changed,rtol=0,atol=0)
        on_changed=model_true.decoder(d,c,w_changed,mask)
        on_difference=float((on_changed-after).abs().max())
        assert on_difference > 0
    # Projection columns still exist but receive exactly zero loss gradient in false.
    for model in [model_true,model_false]:
        model.decoder.zero_grad()
        model.decoder(d,c,w,mask).square().mean().backward()
    on_grad=model_true.decoder.input_projection.weight.grad[:,10:]
    off_grad=model_false.decoder.input_projection.weight.grad[:,10:]
    assert torch.count_nonzero(off_grad) == 0 and torch.count_nonzero(on_grad) > 0
    # Historical prior's zd head is learned; its initialization is deliberately zero.
    historical=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model_false.load_state_dict(historical['model_state_dict'])
    model_true.load_state_dict(historical['model_state_dict'])
    historical_full=copy.deepcopy(model_true)
    old.load_state_dict(model_true.decoder.state_dict())
    historical_full.decoder=old
    changed_u=u.clone()
    changed_u[:,:4]+=2; changed_u[:,4:8]*=1.7
    with torch.no_grad():
        a=model_false(x,mask,u,w); b=model_false(x,mask,changed_u,w_changed)
        unchanged=model_true(x,mask,u,w)
        original=historical_full(x,mask,u,w)
        for key in original:
            torch.testing.assert_close(original[key],unchanged[key],rtol=0,atol=0)
        for key in ['mu_c','log_var_c','mu_d_prior','log_var_d_prior','mu_c_prior','log_var_c_prior']:
            assert not torch.equal(a[key],b[key]), key
        for key in ['mu_d','log_var_d','hi_raw']:
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    # Checkpoint metadata roundtrip, explicit false reload, and legacy default true.
    training_raw=copy.deepcopy(raw)
    training_raw['model']=dict(cfg,decoder_use_W_summary=False)
    training_raw['checkpoint_dir']=str(work_dir/'checkpoints')
    trainer=Trainer(model_false, [], [], training_raw)
    saved=work_dir/'false.pt'; trainer.save_checkpoint(str(saved))
    stored=torch.load(saved,map_location='cpu',weights_only=False)
    assert stored['decoder_use_W_summary'] is False
    assert stored['model_config']['decoder_use_W_summary'] is False
    reloaded=OC_iVAE(dict(cfg,decoder_use_W_summary=False)).eval()
    loader=Trainer(reloaded,[],[],training_raw); loader.load_checkpoint(str(saved))
    for key,value in model_false.state_dict().items():
        torch.testing.assert_close(value,reloaded.state_dict()[key],rtol=0,atol=0)
    legacy=Trainer(model_true,[],[],dict(raw,checkpoint_dir=str(work_dir/'legacy')))
    legacy.load_checkpoint(str(checkpoint))
    try:
        loader.load_checkpoint(str(checkpoint))
    except ValueError:
        pass
    else:
        raise AssertionError('false model accepted legacy true checkpoint')
    # Same seed gives identical initialized weights and stochastic encoder samples.
    torch.manual_seed(42); init_true=OC_iVAE(cfg)
    rng_after_true=torch.get_rng_state()
    torch.manual_seed(42); init_false=OC_iVAE(dict(cfg,decoder_use_W_summary=False))
    torch.testing.assert_close(rng_after_true,torch.get_rng_state(),rtol=0,atol=0)
    for key,value in init_true.state_dict().items():
        torch.testing.assert_close(value,init_false.state_dict()[key],rtol=0,atol=0)
    state=torch.get_rng_state()
    with torch.no_grad():
        init_true.train(); sampled_true=init_true(x,mask,u,w)
        rng_end_true=torch.get_rng_state()
        torch.set_rng_state(state)
        init_false.train(); sampled_false=init_false(x,mask,u,w)
        for key in ['z_d','z_c','hi_raw']:
            torch.testing.assert_close(sampled_true[key],sampled_false[key],rtol=0,atol=0)
        torch.testing.assert_close(rng_end_true,torch.get_rng_state(),rtol=0,atol=0)
    class Samples:
        samples=[(unit,cycle) for unit in range(1,8) for cycle in range(1,21)]
    sampler_true=UnitStratifiedSampler(Samples(),k=7,m=12,random_seed=42)
    sampler_false=UnitStratifiedSampler(Samples(),k=7,m=12,random_seed=42)
    for _ in range(raw['training']['total_epochs']):
        assert list(sampler_true) == list(sampler_false)
    checks=dict(true_matches_original_forward=True,false_decoder_W_invariant=True,
                true_decoder_W_max_difference=on_difference,false_W_column_gradient_nonzeros=0,
                same_parameter_count=sum(p.numel() for p in model_true.parameters()),
                condition_posterior_and_both_priors_preserved=True,
                note='zd conditional head starts at zero as before; sensitivity checked using learned historical weights',
                checkpoint_roundtrip=True,legacy_checkpoint_default_true=True,
                switch_mismatch_rejected=True,identical_seed42_initialization=True,
                identical_stochastic_encoder_draws_and_rng_consumption=True,
                identical_sampler_sequences_for_100_epochs=True,original_revision=original_revision)
    (work_dir/'smoke.json').write_text(json.dumps(checks,indent=2))
    return checks


def main() -> None:
    """Run smoke and print machine-readable results (no training)."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--work-dir',required=True)
    parser.add_argument('--original-revision',required=True)
    parser.add_argument('--checkpoint',default='checkpoints/exp_tuning_mono_weight/best_model.pt')
    args=parser.parse_args()
    print(json.dumps(verify_switch(Path(args.work_dir),args.original_revision,Path(args.checkpoint)),indent=2))


if __name__ == '__main__':
    main()
