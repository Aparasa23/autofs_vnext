from __future__ import annotations
from pathlib import Path
import pandas as pd

def read_table(path: str, fmt: str):
    p = Path(path)
    if fmt == "csv":
        return pd.read_csv(p)
    if fmt == "parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported format: {fmt}")

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
