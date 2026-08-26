# Input data schema

The default configuration expects the following files under `data/raw/`:

- `aspergillus_predictors.csv` (private V3 snapshot): one row per environmental sample;
- `01A_dictionary_taxa.csv` (private V3 snapshot): mapping between anonymized microbiome predictor
  names and their biological names.

The predictor table must contain the sample identifier `Sample` and the
response `Aspergillus_abundance`, as specified in `configs/default.yaml`.
Environmental columns are numeric predictors. Microbiome columns are the
taxonomic predictors identified by the project dictionary. The response is
non-negative; positive-abundance regression uses only rows with a strictly
positive response, while occurrence classification uses the indicator
`Aspergillus_abundance > 0`.

The current private release does not redistribute these files. Before a
public release, add the approved data citation, license, file-level SHA-256
checksums, column definitions, missing-value policy, and the exact dictionary
version to the release manifest. Until then, `scripts/validate_inputs.py`
must be run locally after the user supplies the approved files.
