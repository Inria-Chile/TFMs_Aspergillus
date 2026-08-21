# Architecture

The public repository has four layers:

1. `configs/` describes tasks, folds, preprocessing, models, and output locations.
2. `src/tfms_aspergillus/` contains small, testable primitives for data loading, fold-local transformations, splits, metrics, and provenance.
3. `scripts/` orchestrates seed-level execution and aggregation. A seed is complete only after its metric and explainability contracts validate.
4. `results/` is generated and ignored. Large final artifacts are released separately and pinned by checksums.

The legacy V3 package and builders are kept under `src/marta_omar_repro/` and `scripts/v3_builders/` during migration. New code should import the public package and progressively replace direct path-dependent calls.
