#!/usr/bin/env bash
set -euo pipefail

V3=/home/lvalenzuela/group_storage_nancy/lvalenzuela_SR_results/0_0000000000_a_Marta_Inria_v3
CURRENT="$V3/outputs/231_seed_metrics_shap_consolidation_current_20260710_0251"
COMBINED="$V3/outputs/232_combined_historical_and_current_seed_shap_20260710_0251"
HEATMAPS="$V3/outputs/233_latest_integrated_importance_heatmaps_20260710_0251"
LOG="$V3/outputs/232_233_refresh.log"

exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Iseconds)] waiting for current SHAP consolidation"
for _ in $(seq 1 720); do
    if [[ -s "$CURRENT/tables/metadata.json" ]]; then
        break
    fi
    sleep 30
done
test -s "$CURRENT/tables/metadata.json"

python "$V3/scripts/v3_builders/refresh_combined_shap_summary.py" \
    --historical-long "$V3/outputs/130_combined_v3_seed_and_historical_fold_shap_20260706_2132/tables/combined_shap_seed_feature_long.csv" \
    --current-long "$CURRENT/tables/shap_seed_feature_long.csv" \
    --output-dir "$COMBINED"

SHAP_SUMMARY="$COMBINED/tables/combined_shap_mean_abs_by_feature.csv"
TAXA="$V3/01A_dictionary_taxa_private.csv"

for model in random_forest xgboost tabpfn tabiclv2; do
    python "$V3/scripts/v3_builders/plot_integrated_importance_heatmap.py" \
        --input-summary "$SHAP_SUMMARY" \
        --taxa-dictionary "$TAXA" \
        --output-dir "$HEATMAPS" \
        --model-name "$model" \
        --top-k 3 5 10 \
        --normalize raw column_max zscore_rows \
        --fill-missing 0 \
        --cluster-rows true \
        --cluster-cols false
done

python "$V3/scripts/v3_builders/plot_integrated_importance_heatmap.py" \
    --input-summary "$V3/outputs/135_combined_pysr_equation_predictor_usage_20260706_2211/tables/combined_pysr_equation_predictor_usage_summary.csv" \
    --reg50-summary "$V3/outputs/147_pysr_reg50_equation_usage_corrected_100seeds_named_taxa_20260707_0726/tables/pysr_reg50_predictor_usage_summary_named_taxa_corrected_100seeds.csv" \
    --taxa-dictionary "$TAXA" \
    --output-dir "$HEATMAPS" \
    --model-name pysr \
    --top-k 3 5 10 \
    --normalize raw column_max zscore_rows \
    --fill-missing 0 \
    --cluster-rows true \
    --cluster-cols false

tar -C "$V3/outputs" -czf "$V3/outputs/228_233_updated_boxplots_and_heatmaps_20260710_0251.tar.gz" \
    228_incremental_metric_consolidation_for_boxplots_20260710_0248 \
    229_incremental_99_boxplots_by_scheme_with_r2clip_20260710_0248 \
    230_incremental_100_boxplots_families_by_scheme_with_r2clip_20260710_0248 \
    232_combined_historical_and_current_seed_shap_20260710_0251 \
    233_latest_integrated_importance_heatmaps_20260710_0251

echo "[$(date -Iseconds)] refresh complete"
