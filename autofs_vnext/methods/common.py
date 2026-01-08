from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import numpy as np
import pandas as pd

def make_score_frame(
    feature_names: List[str],
    scores: np.ndarray,
    *,
    method_name: str,
    method_family: str,
    selected_mask: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    scores = np.asarray(scores, dtype=float)
    if selected_mask is None:
        selected_mask = np.ones_like(scores, dtype=bool)
    order = (-scores).argsort(kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores)+1)

    df = pd.DataFrame({
        "feature": feature_names,
        "score": scores,
        "rank": ranks.astype(int),
        "selected_flag": selected_mask.astype(bool),
        "method_name": method_name,
        "method_family": method_family,
        "extra_json": json.dumps(extra or {}),
    })
    df = df.sort_values(["rank","feature"]).reset_index(drop=True)
    return df
