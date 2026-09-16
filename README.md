# TFMs Aspergillus

Reproducible analysis code and consolidated result tables for the
environmental Aspergillus occurrence and abundance study.

## Scope

The workflow evaluates five model families: Random Forest, XGBoost, TabPFN,
TabICLv2, and PySR. It supports three prediction tasks:
occurrence classification, positive-abundance regression, and all-sample
abundance regression including zeros.

The code supports three predictor scenarios (`A_environment`,
`B_microbiome`, and `C_environment_microbiome`) across three preprocessing
families: raw microbiome predictors, CLR-transformed microbiome predictors,
and a predefined subset with CLR-transformed microbiome predictors. The
public final-result bundle included in this repository focuses on
`C_environment_microbiome`; the broader configuration matrix is preserved in
`configs/reproducibility.yaml`.

This repository contains code, provenance records, and consolidated result
tables. Raw sample-level inputs are intentionally excluded from Git.

## Layout

```text
configs/                 YAML experiment definitions
src/tfms_aspergillus/    reusable data, splits, preprocessing, metrics
scripts/                 reproducible command-line entry points
tests/                   unit and contract tests
docs/                    protocol, data dictionary, provenance
data/raw/                user-provided data (ignored by Git)
data/processed/          derived data (ignored by Git)
results/                 generated figures/tables (ignored by Git)
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
# GPU models and SHAP are optional:
python -m pip install -e '.[gpu]'
```

The paper experiments used model-specific environments because the GPU
foundation models and PySR have different runtime requirements. The verified
historical environments are documented in
[`docs/environment-matrix.md`](docs/environment-matrix.md). The flat lock file
is intended for the CPU utilities; it is not a claim that all five model
backends must share one Python environment.

## Reproduce from raw seed outputs

1. Place the input table in `data/raw/` and update `configs/default.yaml`.
2. Run `python scripts/validate_inputs.py`.
3. Run the CPU smoke/metadata entry point with `python scripts/run_seed_experiment.py --config configs/default.yaml --seed 0`.
4. Run a model adapter with `python scripts/runners/run_gpu_seed_tasks.py --help`
   or `python scripts/runners/run_pysr_seed_tasks.py --help`; both use
   repository-relative paths and the validated seed-level export contract.
5. Aggregate only validated seed exports with `python scripts/aggregate_results.py`.
6. Generate figures from the aggregated tables with the plotting scripts.

For a clean reconstruction of the final tabular products from raw
`seed_*_metrics.csv` files, use:

```bash
python scripts/build_final_tables.py \
  --raw-root results/raw \
  --output-dir data/final/rebuilt \
  --shap-table data/final/v3_100_replicas/shap/shap_mean_abs_C_environment_microbiome_100seeds.csv \
  --pysr-table data/final/v3_100_replicas/pysr/pysr_mean_relative_frequency_C_environment_microbiome_100seeds.csv
```

## Reproduce from consolidated tables

The release table bundle is stored in
`data/final/v3_100_replicas/`. It contains the seed-level metrics used for
boxplots, the aggregated SHAP table, and the PySR importance tables. Figures
can be regenerated without the raw result tree:

```bash
python scripts/plot_final_results.py \
  --metrics-table data/final/v3_100_replicas/metrics/seed_metrics_C_environment_microbiome.csv \
  --shap-table data/final/v3_100_replicas/shap/shap_mean_abs_C_environment_microbiome_100seeds.csv \
  --pysr-table data/final/v3_100_replicas/pysr/pysr_mean_relative_frequency_C_environment_microbiome_100seeds.csv \
  --output-dir results/reproduced_figures \
  --models random_forest xgboost tabpfn tabiclv2 pysr \
  --top-n 3 5 10
```

The command writes PNG/PDF boxplots, column-maximum-normalized heatmaps, and
the matrices used by the heatmaps. Use `--no-title` when reproducing the
title-free manuscript variants. The consolidated bundle is checked by
`data/final/v3_100_replicas/manifests/checksums.sha256`.

Every run must write its resolved configuration, software versions, seed,
input checksum, host, and output checksum to a provenance record. Results
from the final study are described in `docs/final-results-manifest.md`; raw
inputs, model checkpoints, and large fold-level result trees are intentionally
excluded from source control.

## Methodological safeguards

- Splits are generated with explicit seeds and persisted before model fitting.
- Classification uses stratified folds where the task permits it.
- Environmental predictors remain on their original scale in all three
  preprocessing configurations.
- The CLR pseudocount and predefined predictor subset were established before
  resampling; the estimates are internally cross-validated rather than the
  product of a fully nested preprocessing pipeline.
- Test metrics are computed only on held-out samples and averaged at the seed level.
- SHAP and PySR frequency are reported as distinct explainability measures.
- A run cannot be marked complete unless expected seed-level artifacts pass schema validation.

## Reproducibility status

The repository is derived from the validated V3 implementation. Exact
versions that were observed in the historical environments are recorded in
`docs/runtime-manifest.yaml` and `docs/environment-matrix.md`. The raw input
data are not redistributed. The original TabICLv2 checkpoint hashes remain
unavailable for independent verification and are documented as a
reproducibility limitation.

Additional conventions:

- `docs/glossary.md` defines abbreviations and response labels.
- `configs/reproducibility.yaml` defines the canonical 100-seed experiment
  matrix.
- SHAP products are written separately to `results/raw_shap/` and
  `results/normalized_shap/`; the latter uses column-maximum normalization
  only for visualization.
- The default figure policy is top-15 predictors, configurable with
  `--top-n`.
- RF-reference population tests use `scripts/statistical_tests.py` and yield
  one p-value per model comparison within family, task, and scenario.
