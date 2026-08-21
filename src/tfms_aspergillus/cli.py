import argparse
from pathlib import Path
import pandas as pd
from .metrics import classification_metrics, regression_metrics


def metrics():
    parser = argparse.ArgumentParser(description="Validate metric table columns.")
    parser.add_argument("table", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.table)
    required = {"seed", "model", "task", "scheme"}
    missing = required - set(frame.columns)
    if missing:
        parser.error(f"missing columns: {sorted(missing)}")
    print(f"validated {len(frame)} rows")
