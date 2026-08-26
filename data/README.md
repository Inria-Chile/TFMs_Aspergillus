Place approved/public input data here; raw and processed data are ignored by Git.

## Private V3 input snapshot

This private repository includes the anonymized input snapshot used by the
experiments:

- `raw/aspergillus_predictors.csv`: 50 environmental samples and their
  predictor/response columns. Source filename:
  `01A_table_anonymized_for_external_analysis.csv`.
- `raw/01A_dictionary_taxa.csv`: mapping for the anonymized microbiome
  predictors. Source filename: `01A_dictionary_taxa_private.csv`.

The files are retained in `data/raw/` because the repository is private at
this stage. Their SHA-256 values are recorded in
`../docs/runtime-verification.md` and must be rechecked before any public
release. Do not publish these files until the data-licensing decision is
complete.
