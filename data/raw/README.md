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

## Citation

The sample-level microbiome predictors are redistributed with permission. If
you use these raw input data, please cite the companion microbiota study:

```bibtex
@misc{chauvin2026urbanization,
  title = {Urbanization impacts the environmental microbial community and its antibiotic resistome by reshaping the diversity and network complexity},
  author = {Chauvin, Gautier and Defaye, Baptiste and Enaud, Raphael and Rodriguez, Marion and Vieira, Dânia and Malvy, Denis and Imbert, Christine and Delhaes, Laurence},
  year = {2026},
  note = {Available at SSRN},
  url = {https://ssrn.com/abstract=6988265},
  doi = {10.2139/ssrn.6988265}
}
```

For the machine-learning analyses and consolidated results, cite the associated
SIMBig 2026 paper listed in the repository `README.md`.
