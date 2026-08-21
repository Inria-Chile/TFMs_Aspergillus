#!/usr/bin/env python3
"""Run RF-reference population comparisons on seed-level metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tfms_aspergillus.statistics import compare_populations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--value-column", default="value")
    parser.add_argument("--reference-model", default="Random Forest")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    required = {"model", "family", "task", "scenario", "seed", args.value_column}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")
    rows = []
    groups = ["family", "task", "scenario"]
    for keys, group in frame.groupby(groups, dropna=False):
        reference = group[group["model"].eq(args.reference_model)]
        for model, target in group.groupby("model", dropna=False):
            if model == args.reference_model:
                continue
            result = compare_populations(reference[args.value_column], target[args.value_column])
            rows.append(dict(zip(groups, keys), reference_model=args.reference_model, target_model=model, metric=args.value_column, **result))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
