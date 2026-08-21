from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureBlocks:
    env_cols: list[str]
    human_cols: list[str]
    dummy_cols: list[str]
    bacteria_cols: list[str]
    fungi_cols: list[str]
    microbiome_predictors: list[str]


def load_table(path: Path, target_col: str, id_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if id_col not in df.columns:
        df.insert(0, id_col, [f"S{i + 1}" for i in range(len(df))])
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    for col in df.columns:
        if col != id_col:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    df["Aspergillus_presence"] = (df[target_col].astype(float) > 0).astype(int)
    df["Aspergillus_abundance"] = df[target_col].astype(float)
    return df


def detect_blocks(df: pd.DataFrame, target_col: str, dummy_cols: list[str]) -> FeatureBlocks:
    env_cols = list(df.columns[:30])
    human_cols = list(df.columns[30:36])
    dummy_present = [c for c in dummy_cols if c in df.columns]
    bacteria_cols = [c for c in df.columns if c.startswith("B") and c[1:].isdigit()]
    fungi_cols = [
        c
        for c in df.columns
        if (c.startswith("F") and c[1:].isdigit()) or c == target_col
    ]
    microbiome = [c for c in bacteria_cols + fungi_cols if c != target_col]
    return FeatureBlocks(
        env_cols=env_cols,
        human_cols=human_cols,
        dummy_cols=dummy_present,
        bacteria_cols=bacteria_cols,
        fungi_cols=fungi_cols,
        microbiome_predictors=microbiome,
    )


def medium_labels(df: pd.DataFrame, dummy_cols: list[str]) -> np.ndarray:
    labels = np.full(len(df), "sand", dtype=object)
    names = ["salt", "fresh", "sediment"]
    for col, name in zip(dummy_cols, names):
        if col in df.columns:
            labels[df[col].astype(int).to_numpy() == 1] = name
    return labels


def write_json(path: Path, payload: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
