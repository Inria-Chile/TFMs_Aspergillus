#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results")
    parser.add_argument("--output", default="results/metrics_seed_long.csv")
    args = parser.parse_args()
    paths = sorted(Path(args.input).glob("**/seed_*_metrics.csv"))
    if not paths:
        raise SystemExit("no validated seed metric files found")
    frames = [pd.read_csv(path).assign(source_file=str(path)) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated().any():
        raise SystemExit("duplicate metric rows detected")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"aggregated {len(paths)} files and {len(result)} rows")


if __name__ == "__main__":
    main()
