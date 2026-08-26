#!/usr/bin/env python3
"""Build the repository's final tables from raw seed-level outputs.

Example:
    python scripts/build_final_tables.py --raw-root results/raw \
        --output-dir data/final/rebuilt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tfms_aspergillus.results import aggregate_raw_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shap-table", type=Path)
    parser.add_argument("--pysr-table", type=Path)
    args = parser.parse_args()
    metrics_out = args.output_dir / "metrics" / "seed_metrics_long_for_boxplots.csv"
    frame = aggregate_raw_metrics(args.raw_root, metrics_out)
    print(f"wrote {len(frame)} metric rows to {metrics_out}")
    for source, destination in ((args.shap_table, args.output_dir / "shap" / "shap_mean_abs_by_feature.csv"),
                                (args.pysr_table, args.output_dir / "pysr" / "aggregated_importances.csv")):
        if source:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            print(f"copied {source} -> {destination}")


if __name__ == "__main__":
    main()
