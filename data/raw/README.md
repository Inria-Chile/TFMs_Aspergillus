# Raw input data

This directory contains the authorized machine-learning input files used by the
V3 Aspergillus analyses.

- `aspergillus_predictors.csv`: sample-level predictor and response table with
  50 rows. It contains environmental covariates, anonymized bacterial
  predictors (`B*`), anonymized fungal predictors (`F*`), the response column
  `F_Aspergillus`, sample identifiers, and environment dummy variables.
- `01A_dictionary_taxa.csv`: mapping from anonymized microbiome predictor
  names to original genus names and taxonomic annotations.
- `checksums.sha256`: SHA-256 checksums for the released raw input files.

The microbiome data are redistributed with permission following acceptance of
the companion microbiota study:

G. Chauvin, B. Defaye, R. Enaud, M. Rodriguez, D. Vieira, D. Malvy,
C. Imbert, and L. Delhaes. Urbanization impacts the environmental microbial
community and its antibiotic phenotypic resistance potential by reshaping the
diversity and network complexity. Environmental Research. To appear.
