#!/usr/bin/env python3
"""Prepare a reproducible seed run record.

This entry point is deliberately conservative: model-specific execution is
not performed here. Use model-specific runners under scripts/runners for validated fitting and seed-level exports.
"""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run = Path(args.output) / f"seed_{args.seed:03d}"
    run.mkdir(parents=True, exist_ok=False)
    metadata = {"seed": args.seed, "config": config, "python": platform.python_version(),
                "created_utc": datetime.now(timezone.utc).isoformat(), "status": "prepared"}
    (run / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"prepared metadata only: {run}; no model was fitted")


if __name__ == "__main__":
    main()
