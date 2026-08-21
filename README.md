# TFMs Aspergillus

Reproducible code for the Aspergillus occurrence and abundance study. The repository follows the separation used by the Inria-Chile `planktonzilla` project: configuration, installable source package, executable scripts, tests, documentation, and provenance are kept distinct.

## Scope

The workflow evaluates five model families: Random Forest, XGBoost, TabPFN, TabICLv2, and PySR. It supports three tasks (occurrence classification, positive-abundance regression, and complete zero-inflated regression) and three predictor strategies (Environment, Microbiome, and Environment + Microbiome), with raw, CLR, and subset-CLR variants.

This repository contains code and small metadata/manifests. Large raw data and generated results are intentionally excluded from Git and are referenced through a versioned manifest.

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

## Reproduce

1. Place the input table in `data/raw/` and update `configs/default.yaml`.
2. Run `python scripts/validate_inputs.py`.
3. Run the CPU smoke/metadata entry point with `python scripts/run_seed_experiment.py --config configs/default.yaml --seed 0`.
4. Run a model adapter with `python scripts/runners/run_gpu_seed_tasks.py --help` or `python scripts/runners/run_pysr_seed_tasks.py --help`; both use repository-relative paths and the V3 seed-level export contract.
5. Aggregate only validated seed exports with `python scripts/aggregate_results.py`.
6. Generate figures from the aggregated tables with the plotting scripts.

Every run must write its resolved configuration, software versions, seed, input checksum, host, and output checksum to a provenance record. Results from the final study are described in `docs/final-results-manifest.md`; they are not silently bundled into source control.

## Methodological safeguards

- Splits are generated with explicit seeds and persisted before model fitting.
- Classification uses stratified folds where the task permits it.
- Environmental transforms and feature selection are fit on training folds only.
- Test metrics are computed only on held-out samples and averaged at the seed level.
- SHAP and PySR frequency are reported as distinct explainability measures.
- A run cannot be marked complete unless expected seed-level artifacts pass schema validation.

## Reproducibility status

The initial repository is derived from the validated V3 implementation. The GPU and PySR runners have now been migrated from their former absolute V2/V3 paths to repository-relative paths. Before publication, pin exact dependency versions, add the final public data DOI/access instructions, add CI coverage for the model adapters, and publish a release tag matching the paper.
