# Input data schema

The default configuration expects the following files under `data/raw/`:

- `aspergillus_predictors.csv`: an authorized local table with one row per
  environmental sample;
- `01A_dictionary_taxa.csv`: an authorized local mapping between anonymized
  microbiome predictor names and their biological names.

The predictor table must contain the sample identifier `Sample` and the
response `F_Aspergillus`, as specified in `configs/default.yaml`. This is the
anonymized response column used by the V3 experiments.
Environmental columns are numeric predictors. Microbiome columns are the
taxonomic predictors identified by the project dictionary. The response is
non-negative; positive-abundance regression uses only rows with a strictly
positive response, while occurrence classification uses the indicator
`F_Aspergillus > 0`.

The upstream Google Earth Engine scripts that generated and extracted the
environmental covariates are archived in
`scripts/gee_environmental_covariates/`. The final environmental analytical
stack used by V3 contains 36 variables. The candidate variables `CO_mean`,
`landforms`, and `geomorphons` are documented there but were not retained in
the ML input matrix.

This repository does not redistribute either input file. The workflow must be
run only after the user supplies authorized copies locally and validates them
with `scripts/validate_inputs.py`.
