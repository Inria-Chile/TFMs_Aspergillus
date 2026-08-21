#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default="data/raw/aspergillus_predictors.csv")
    parser.add_argument("--target", default="Aspergillus_abundance")
    args = parser.parse_args()
    frame = pd.read_csv(args.table)
    if args.target not in frame:
        raise SystemExit(f"missing target: {args.target}")
    if frame[args.target].isna().any():
        raise SystemExit("target contains missing values")
    print(f"OK: {Path(args.table)} rows={len(frame)} columns={len(frame.columns)}")


if __name__ == "__main__":
    main()
