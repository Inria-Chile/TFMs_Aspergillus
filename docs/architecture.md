# Architecture

The public repository has four layers:

1. `configs/` describes tasks, folds, preprocessing, models, and output locations.
2. `src/tfms_aspergillus/` contains small, testable primitives for data loading, fold-local transformations, splits, metrics, and provenance.
3. `scripts/` orchestrates seed-level execution and aggregation. A seed is complete only after its metric and explainability contracts validate.
4. `results/` is generated and ignored. Large final artifacts are released separately and pinned by checksums.

The validated V3 compatibility layer and builders are retained under
`src/tfms_aspergillus/v3_workflow/` and `scripts/v3_builders/` for provenance
and backward-compatible reconstruction. New development should import the
public package and avoid direct path-dependent calls.
