# Model runners

The migrated runners are repository-relative and preserve the validated V3
seed-level export contract.

## Tree and tabular models

```bash
python scripts/runners/run_gpu_seed_tasks.py \
  --family family2_full_microbiome_clr_raw_env \
  --family-label "Family 2" \
  --task-kind occurrence \
  --model random_forest \
  --tasks-csv data/processed/tasks.csv \
  --device cuda:0 \
  --seeds 0,1,2 \
  --resume --cleanup-partials \
  --run-tag smoke_rf
```

The same entry point supports `xgboost`, `tabpfn`, and `tabiclv2`. Independent
workers must receive disjoint seeds and explicit physical GPU IDs. The
launcher is designed for one process per GPU after a measured probe; it does
not silently oversubscribe a GPU.

## PySR

```bash
python scripts/runners/run_pysr_seed_tasks.py \
  --family family1_no_clr_microbiome_raw \
  --family-label "Family 1" \
  --task-kind occurrence \
  --tasks-csv data/processed/tasks.csv \
  --seeds 0,1,2 --resume
```

## Dependencies and external backends

The core package is CPU-installable. SHAP, XGBoost, TabPFN, TabICLv2 and PySR
are optional runtime dependencies. Their exact versions and model checkpoints
must be pinned in a release lockfile before publication. No runner may use a
machine-specific absolute project path; external model repositories are
provided through `TABFM_REPO` or an equivalent documented configuration.
