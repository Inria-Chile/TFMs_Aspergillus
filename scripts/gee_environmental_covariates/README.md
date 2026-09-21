# Google Earth Engine Environmental Covariate Provenance

This directory contains the Google Earth Engine JavaScript scripts used as
provenance code for the environmental covariate generation workflow associated
with the Aspergillus prediction study in Aquitaine, France.

The scripts are included for transparency and traceability. They are not part
of the repository test suite because they depend on Google Earth Engine, on
project-specific Earth Engine assets under Omar Orellana's INRIA workspace, and
on terrain derivatives that were produced outside Earth Engine with SAGA GIS
before being uploaded as Earth Engine assets.

## Relationship to the machine-learning inputs

The V3 machine-learning analyses used a 36-variable environmental covariate set
plus three sample-environment dummy variables. The final extraction script,
`19_Final_Covariate_Stack_and_Extraction_Aquitaine_France.js`, documents the
larger 39-variable candidate pool but exports the final 36-variable analytical
stack used by the ML workflow.

The three candidate variables documented but excluded from the final analytical
ML matrix are:

- `CO_mean`
- `landforms`
- `geomorphons`

The 36 environmental variables used in the full environmental predictor set are:

```text
AER_AI_mean
HBV
HCHO_mean
HFI
HMI
HPD
LS_PC1
LS_PC2
NDFI
NO2_mean
NTL
P_PC1
P_PC2
SO2_mean
SOC_0_5cm_pct
T_PC1
T_PC2
asp_cos
asp_sin
built
crops
curv
curv_plan
curv_prof
dist_river_m
dist_sea
dist_urban
elevation_m
flow_acc_log
flow_dir_cos
flow_dir_sin
flow_len
pH_0_5cm
slope_pct
trees
wind_speed_10m
```

The sample-environment dummy variables used by the ML tasks were created in the
tabular ML preprocessing code rather than in Earth Engine:

```text
C1_salt
C2_fresh
C3_sed
```

`sand` is represented as the reference category.

## Script order

Scripts `01` through `18` generate or prepare individual environmental
covariate products. Script `19` integrates those products and extracts
covariate values at the 50 sampling locations.

The script sequence is summarized in `manifest.csv`.

## Reproducibility note

These scripts are intended to document how the environmental layers were
generated and assembled. Re-running them requires access to the original Earth
Engine account, assets, and any externally generated terrain rasters. The
published ML results in this repository should therefore be treated as the
reproducible tabular analysis layer, while this directory documents the upstream
remote-sensing and GIS provenance.
