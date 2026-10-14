# Task T1b · Baseline Router in run_train.py and run_eval.py

> Context file: AGENTS.md (auto-loaded)
> Scope: ONLY modify `scripts/run_train.py` and `scripts/run_eval.py`
> Prerequisite: Task T1 must be complete (`baselines/cvae.py` exists)

---

## Background

`run_train.py` and `run_eval.py` currently hardcode `OC_iVAE` as the model class.
The cVAE baseline (and all future baselines T2–T4) use the same training and
evaluation pipeline, but need a different model builder. This task adds a
minimal routing layer so that any config with `experiment.baseline_type` set
will use the correct builder, while configs without it (all existing Full Model
configs) behave exactly as before.

**Design principle**: The router must be the only change. Do not restructure,
refactor, or "improve" any other logic in these scripts.

---

## Current hardcoded lines to replace

### In `scripts/run_train.py`

**Current** (around line 46, inside `main()`):
```python
model = OC_iVAE(model_config)
```

**Replace with** a call to a new helper `_build_model(config, model_config)`.

### In `scripts/run_eval.py`

**Current** (around line 56, inside `main()`):
```python
model = OC_iVAE(flat_config)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict)
```

**Replace with** a call to the same helper `_build_model(raw_config, flat_config)`,
then load the checkpoint as before.

---

## Task

### Step 1 — Add `_build_model()` to `scripts/run_train.py`

Add the following function anywhere before `main()`. Do not change anything else.

```python
def _build_model(raw_config: dict[str, Any], model_config: dict[str, Any]):
    """Instantiate the correct model class based on experiment.baseline_type.

    Returns an nn.Module whose forward() matches the OC_iVAE interface.
    If experiment.baseline_type is absent or None, returns a plain OC_iVAE.

    Args:
        raw_config: The full nested YAML config dict (with sections).
        model_config: The flat model config dict passed to the model constructor.

    Returns:
        An nn.Module ready for training.
    """
    baseline_type = (
        raw_config.get("experiment", {}).get("baseline_type") or None
    )
    if baseline_type is None:
        return OC_iVAE(model_config)
    if baseline_type == "cvae":
        from baselines.cvae import build_cvae_model
        return build_cvae_model(model_config)
    raise ValueError(
        f"Unknown baseline_type {baseline_type!r}. "
        "Supported values: None, 'cvae'."
    )
```

Then replace the hardcoded line in `main()`:
```python
# BEFORE
model = OC_iVAE(model_config)

# AFTER
model = _build_model(config, model_config)
```

`config` here is the full raw nested config dict (the return value of
`_load_config(config_path)` — already in scope at that point).

Also add the following import at the top of the import block (alongside the
existing `from model.oc_ivae import OC_iVAE`):
```python
# No new top-level imports needed — baselines.cvae is imported lazily inside
# _build_model() to avoid ImportError when baselines/ does not yet exist.
```
Do not add any top-level `from baselines...` import. The lazy import inside
`_build_model()` is intentional — it avoids breaking existing runs that don't
have T2/T3/T4 implemented yet.

### Step 2 — Add `_build_model()` to `scripts/run_eval.py`

Add the **identical** `_build_model()` function to `run_eval.py` (same docstring,
same logic). Then replace the hardcoded model instantiation in `main()`:

```python
# BEFORE
model = OC_iVAE(flat_config)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict)

# AFTER
model = _build_model(raw_config, flat_config)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict)
```

Note: `weights_only` is changed from `True` to `False` because baseline
checkpoints may include optimizer state. This matches the existing behavior
in `train/trainer.py:load_checkpoint()`.

`raw_config` is already in scope in `run_eval.py:main()` — it is the return
value of `_load_config(args.config)`.

---

## Verification

After implementing, verify mentally:

```python
# Case 1: existing Full Model config — no experiment section
config_full = {"data": {...}, "model": {...}, "training": {...}}
model = _build_model(config_full, model_config)
assert type(model).__name__ == "OC_iVAE"

# Case 2: cVAE baseline config — experiment.baseline_type = "cvae"
config_cvae = {"experiment": {"baseline_type": "cvae"}, "model": {...}, ...}
model = _build_model(config_cvae, model_config)
from baselines.cvae import StandardNormalPrior
assert isinstance(model.prior, StandardNormalPrior)

# Case 3: experiment section present but baseline_type is null/absent
config_no_type = {"experiment": {"name": "test"}, ...}
model = _build_model(config_no_type, model_config)
assert type(model).__name__ == "OC_iVAE"

# Case 4: unknown baseline_type raises ValueError
config_bad = {"experiment": {"baseline_type": "unknown"}, ...}
try:
    _build_model(config_bad, model_config)
    assert False, "should have raised"
except ValueError:
    pass
```

Then run:
```bash
python -m pytest tests/ -v
```

---

## Task completion checklist

- [ ] `_build_model()` added to `scripts/run_train.py` with correct docstring
- [ ] `_build_model()` added to `scripts/run_eval.py` with correct docstring
- [ ] `model = OC_iVAE(model_config)` replaced in `run_train.py:main()`
- [ ] `model = OC_iVAE(flat_config)` replaced in `run_eval.py:main()`
- [ ] `weights_only=False` set in `run_eval.py` checkpoint load
- [ ] No other changes to either script
- [ ] No top-level `from baselines...` imports added (lazy import only)
- [ ] `python -m pytest tests/ -v` passes
