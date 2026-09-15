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
`Aspergillus_abundance > 0`.

This repository does not redistribute either input file. The workflow must be
run only after the user supplies authorized copies locally and validates them
with `scripts/validate_inputs.py`.
