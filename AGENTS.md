# Project Guidelines

This is a research codebase for implementing and comparing models.
Prioritize algorithm correctness, readability, and reproducibility.
Use the minimum complexity needed for the current experiment.

## Project Structure

- Keep train.py and evaluate.py as separate executable entry points.
- Keep data loading, cycle/window construction, collation, and sampling
  in dataset.py.
- Keep model implementations and necessary model components in models/.
- Keep experiment configurations in config/ as standalone YAML files.
- Keep original data in data/raw/ and prepared data in data/processed/.
- Keep offline data preparation in utils/preprocess.py.
- Keep reusable utilities in clearly named modules under utils/.
  Do not use utils/ as a dumping ground for unrelated code.
- Keep experiment orchestration scripts in experiments/, such as
  experiments/run_w_summary_ablation.py.
- Experiment scripts should call the training and evaluation entry points
  or their ordinary function interfaces. Do not duplicate training loops,
  inference workflows, or metric calculations.
- Save configuration snapshots, checkpoints, and results in
  runs/<experiment_name>/.
- Keep research documentation in docs/ and meaningful tests in tests/.
- Document executable commands in README.md and dependencies in
  requirements.txt.
- Do not introduce main.py, a generic Trainer, model registries,
  configuration inheritance, or additional framework layers unless
  the current task clearly requires them.
- This is the target structure. Migrate existing files only through an
  explicitly approved refactoring plan.

## Coding Standard

1. Simplicity first.
   - Make only changes directly requested or clearly necessary.
   - Prefer explicit control flow, ordinary functions, and direct calls.
   - Do not add configurability for hypothetical future requirements.
   - Prefer a few clear repeated lines over a premature abstraction.
   - Optimize for comprehension, not minimum line count.

2. Fail fast.
   - Let unexpected errors surface with useful tracebacks.
   - Validate necessary assumptions at boundaries: YAML configuration,
     external data, and checkpoint loading.
   - Trust established internal contracts and framework guarantees.
   - Avoid redundant validation, broad exception handling, silent
     fallbacks, automatic retries, and implicit parameter correction.
   - Never silently skip invalid samples or substitute missing features.

3. Keep abstractions justified.
   - Extract functions when they name a meaningful algorithm step,
     reduce actual duplication, or make complex logic easier to read.
   - Do not create helpers that merely wrap a single obvious operation.
   - Avoid base classes, factories, plugins, and indirection without
     a concrete current need.
   - Keep model-specific logic inside the corresponding model file.
   - Keep training and evaluation workflows directly readable.

4. Remove obsolete code.
   - Delete code that is confirmed to be unused.
   - Do not preserve unused variables, obsolete exports, commented-out
     implementations, or comments announcing removed code.
   - Do not add compatibility shims unless compatibility is explicitly
     required by the task.
   - Do not refactor unrelated code.

## Research Correctness and Reproducibility

- Treat structural refactoring as behavior-preserving unless algorithm
  changes are explicitly requested.
- Preserve model computations, latent-variable semantics, conditioning
  paths, loss definitions, loss reductions, schedules, and gradients.
- Preserve dataset splits, feature ordering, normalization, auxiliary
  descriptors, masking, padding, truncation, and sampling semantics.
- Preserve checkpoint selection, inference behavior, HI direction,
  post-processing, metric definitions, and evaluation protocols.
- Preserve existing experiment hyperparameters when moving YAML files;
  update paths only as necessary.
- Do not add label supervision to an unsupervised training objective.
- Keep existing model variants and ablations behaviorally distinct.
- Record the resolved configuration and random seed for each run.
- Keep raw data, prepared datasets, checkpoints, and large generated
  outputs out of Git.
- Never delete or overwrite existing datasets or experiment results
  as part of code cleanup.
- Report suspected algorithmic bugs separately. Do not silently fix them
  during a structural refactor.

## Verification

- Run checks appropriate to the change.
- For structural refactoring, verify imports, executable entry points,
  configuration loading, and the existing meaningful tests.
- Use focused numerical comparisons where model, loss, data, or evaluation
  code is moved or reorganized.
- Where available, verify an existing checkpoint can still be loaded and
  produces equivalent inference results.
- Do not report empty test files or zero collected tests as evidence
  of correctness.
- Do not run full training or expensive experiments unless explicitly
  authorized.
- Report unavailable data, checkpoints, dependencies, and unverified
  behavior clearly.

## Workflow

- Report the plan before making changes and wait for my explicit approval.
- Approval applies to the stated scope; request approval again only if
  the scope materially changes.
- After changes pass verification, git commit only the task-related changes.
- Never force-push. Report any synchronization failures.
- Do not push or publish changes unless explicitly requested.

