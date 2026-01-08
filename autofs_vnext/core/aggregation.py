from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

def rrf_aggregate(rank_frames: List[pd.DataFrame], k: int = 60) -> pd.DataFrame:
    # Each frame must have: feature, rank (1=best)
    scores: Dict[str, float] = {}
    for df in rank_frames:
        for _, row in df[["feature", "rank"]].iterrows():
            f = row["feature"]
            r = float(row["rank"])
            scores[f] = scores.get(f, 0.0) + 1.0 / (k + r)
    out = pd.DataFrame({"feature": list(scores.keys()), "rrf_score": list(scores.values())})
    out["rank"] = out["rrf_score"].rank(ascending=False, method="min").astype(int)
    out = out.sort_values(["rank","feature"]).reset_index(drop=True)
    return out

def borda_aggregate(rank_frames: List[pd.DataFrame]) -> pd.DataFrame:
    # Lower rank better; convert to points
    scores: Dict[str, float] = {}
    for df in rank_frames:
        n = len(df)
        for _, row in df[["feature", "rank"]].iterrows():
            f = row["feature"]
            r = float(row["rank"])
            scores[f] = scores.get(f, 0.0) + (n - r + 1.0)
    out = pd.DataFrame({"feature": list(scores.keys()), "borda_score": list(scores.values())})
    out["rank"] = out["borda_score"].rank(ascending=False, method="min").astype(int)
    out = out.sort_values(["rank","feature"]).reset_index(drop=True)
    return out
