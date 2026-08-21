"""Raw and column-max-normalized SHAP export products.""" 
from pathlib import Path
import pandas as pd

GROUP_COLUMNS = ["family", "task", "scenario", "model"]

def _contract(frame, family, task, scenario, model):
    out = frame.copy()
    out["family"], out["task"], out["scenario"], out["model"] = family, task, scenario, model
    if "fold_id" not in out:
        out["fold_id"] = -1
    if "shap_value" not in out:
        raise ValueError("SHAP input must contain shap_value")
    if "abs_shap_value" not in out:
        out["abs_shap_value"] = pd.to_numeric(out["shap_value"], errors="coerce").abs()
    return out

def summarize(frame):
    required = set(GROUP_COLUMNS + ["variable", "fold_id", "seed", "shap_value", "abs_shap_value"])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("SHAP input missing columns: " + str(sorted(missing)))
    return (
        frame.groupby(GROUP_COLUMNS + ["variable", "fold_id"], dropna=False, as_index=False)
        .agg(
            mean_signed_shap=("shap_value", "mean"),
            mean_abs_shap=("abs_shap_value", "mean"),
            sd_signed_shap=("shap_value", "std"),
            sd_abs_shap=("abs_shap_value", "std"),
            n_values=("shap_value", "size"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(GROUP_COLUMNS + ["mean_abs_shap"], ascending=[True, True, True, True, False])
    )

def export_shap_products(frame, output_root, *, family, task, scenario, model, seed):
    out = _contract(frame, family, task, scenario, model)
    out["seed"] = int(seed)
    seed_dir = f"seed_{int(seed):03d}"
    raw = Path(output_root) / "raw_shap" / family / task / scenario / model / seed_dir
    norm = Path(output_root) / "normalized_shap" / family / task / scenario / model / seed_dir
    raw.mkdir(parents=True, exist_ok=True)
    norm.mkdir(parents=True, exist_ok=True)
    out.to_csv(raw / "shap_values.csv", index=False)
    summarize(out).to_csv(raw / "shap_summary.csv", index=False)
    denom = out.groupby(GROUP_COLUMNS + ["fold_id"], dropna=False)["abs_shap_value"].transform("max")
    denom = denom.where(denom.ne(0), 1.0)
    normal = out.copy()
    normal["abs_shap_value_column_max"] = normal["abs_shap_value"].div(denom)
    normal["shap_value_column_max"] = normal["shap_value"].div(denom)
    normal.to_csv(norm / "shap_values.csv", index=False)
    summarize(
        normal.assign(
            abs_shap_value=normal["abs_shap_value_column_max"],
            shap_value=normal["shap_value_column_max"],
        )
    ).to_csv(norm / "shap_summary.csv", index=False)
    return {
        "raw_shap": str(raw / "shap_values.csv"),
        "normalized_shap": str(norm / "shap_values.csv"),
    }
