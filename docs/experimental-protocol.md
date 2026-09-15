# Experimental protocol

This document defines the analysis unit and the operations that must be
preserved when reproducing the Aspergillus study.

## Data and prediction tasks

The dataset contains 50 environmental samples, of which 18 have detected
Aspergillus and 32 have a zero response. The response is denoted by
`y >= 0`. Three tasks are evaluated:

1. **Occurrence classification:**
   \(z_i = 1[y_i > 0]\), evaluated with AUC, F1, and balanced accuracy.
2. **Positive-abundance regression:** the response is restricted to
   \(\{y_i:y_i>0\}\), evaluated with RMSE, MAE, and R2.
3. **Complete zero-inflated regression:** all 50 response values are retained,
   including zeros, and the same regression metrics are used.

The 100 seeds (`123` through `222`) are analysis replicates, not independent
biological samples. For each seed, fold-level predictions are aggregated to
one seed-level metric before the boxplots are built.

## Predictor families and scenarios

Each family defines a preprocessing view of the predictors:

- Family 1: microbiome predictors without CLR preprocessing.
- Family 2: full CLR microbiome predictors with the environmental table kept
  on its original scale.
- Family 3: the retained Marta subset with CLR-transformed microbiome
  predictors and the corresponding environmental predictors.

Within each family, scenarios are:

- `A_environment`: environmental predictors;
- `B_microbiome`: microbiome predictors;
- `C_environment_microbiome`: the combined predictor set.

The configuration and output labels are defined in
`configs/reproducibility.yaml`. Raw, full-CLR, and subset-CLR variants must
never be mixed when building a comparison.

## Resampling and leakage control

The canonical configuration specifies stratified 15-fold resampling for
classification, leave-one-out evaluation for positive-abundance regression,
and 18-fold resampling for the complete zero-inflated regression. The
classification folds preserve the occurrence labels as far as the fold size
permits; because there are only 18 positive observations, the number of
positives per fold must be reported from the resolved fold manifest rather
than inferred from the nominal fold count.

Environmental predictors remain untransformed. Microbiome predictors are used
as raw relative abundances in Family 1 and with a CLR transformation in
Families 2 and 3. The CLR pseudocount and the Family 3 predictor subset were
defined before resampling rather than estimated inside each training fold.
Consequently, the results compare fixed predictor configurations under
internal cross-validation, not a fully nested feature-selection pipeline.
Held-out outcomes are not used to tune the fitted models.

The exact fold assignments, resolved configuration, input checksum, package
versions, and host are part of the per-seed provenance contract. A seed is
complete only when all expected fold exports and the metric schema validate.

## Model settings

The repository runners expose the following validated defaults:

- Random Forest: 500 estimators, `class_weight=balanced` for classification,
  `max_features=sqrt`, one model worker per process.
- XGBoost: 500 estimators, learning rate 0.03, maximum depth 3,
  `min_child_weight=1`, subsample 0.632, `reg_alpha=0`, `reg_lambda=1`,
  `max_bin=256`, one worker, and the task-specific XGBoost objectives.
- TabICLv2: six estimators, batch size 2, representation cache, and the
  classifier/regressor checkpoints named in `docs/runtime-manifest.yaml`.
- PySR: eight populations, population size 5,000, 20 iterations,
  120-second timeout, maximum expression size 10, multiprocessing with eight
  processes, sigmoid occurrence loss and L1 regression loss.

TabPFN and TabICLv2 use their model-specific GPU environments. Any change to
these settings creates a different computational experiment and must be
recorded in the resolved configuration.

## Evaluation and explainability

For a seed with fold metrics \(m_{s,f}\), the reported seed value is the
arithmetic mean over its valid test folds. RMSE is
\(\sqrt{n^{-1}\sum_i(y_i-\hat y_i)^2}\), MAE is
\(n^{-1}\sum_i|y_i-\hat y_i|\), and R2 is the coefficient of determination.
AUC is the area under the ROC curve; F1 is the harmonic mean of precision and
recall; balanced accuracy is the mean of sensitivity and specificity.

SHAP values describe the contribution of a predictor to a model output. PySR
frequency describes how often a predictor occurs in the selected symbolic
equations. These are complementary quantities and must not be interpreted as
the same importance measure. The final SHAP heatmaps use absolute SHAP
values and column-wise maximum normalization only for visualization; the
unscaled values and their dispersion remain in the tabular product.

## Reproduction order

1. Obtain the approved input table and microbiome dictionary.
2. Record their SHA-256 checksums and place them under `data/raw/`.
3. Resolve the configuration and generate the fold/task manifest.
4. Run seed-level model jobs with disjoint seeds and explicit GPU IDs.
5. Validate metric, prediction, SHAP, and equation exports.
6. Build the consolidated tables with `scripts/build_final_tables.py`.
7. Generate boxplots and heatmaps with `scripts/plot_final_results.py`.

The repository includes consolidated tabular results but not the sample-level
input data. External end-to-end reproduction therefore requires authorized
access to the inputs; figures remain reproducible from the consolidated tables.
