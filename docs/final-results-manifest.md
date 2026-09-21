# Final results manifest

The final public tabular products are stored under
`data/final/v3_100_replicas/`. They correspond to scenario
`C_environment_microbiome` and were rebuilt from archived consolidation
batches `403`, `457`, `458`, `459`, and `471`.

The bundle contains:

- seed-level performance metrics for 105 family-task-model-metric groups;
- 36 complete 100-seed SHAP groups;
- nine complete 100-seed PySR predictor-frequency groups;
- a group-level coverage manifest;
- SHA-256 checksums for the released tables.

All performance, SHAP, and PySR-importance groups in the release contain 100
seeds. The three PySR classification metrics for the predefined subset with
CLR were replaced by complete 100-seed series from the validated batch `403`;
this resolves the earlier 60-seed public-table gap.

The bundle contains no figures, fold-level monitoring tree, or model
checkpoints. The authorized sample-level input table and microbiome dictionary
are distributed separately under `data/raw/`.
