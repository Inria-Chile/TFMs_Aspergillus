from pathlib import Path
import pandas as pd


def load_table(path: str | Path, target_column: str) -> pd.DataFrame:
    """Load the study table and derive occurrence from abundance."""
    frame = pd.read_csv(path)
    if target_column not in frame:
        raise ValueError(f"Missing target column: {target_column}")
    frame["Aspergillus_abundance"] = pd.to_numeric(frame[target_column], errors="raise")
    frame["Aspergillus_presence"] = (frame["Aspergillus_abundance"] > 0).astype(int)
    return frame
