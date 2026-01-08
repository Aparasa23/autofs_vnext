from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from autofs_vnext.core.schemas import AutoFSConfig
from autofs_vnext.core.registry import REGISTRY, autodiscover
from autofs_vnext.core.preprocess import Preprocessor
from autofs_vnext.core.aggregation import rrf_aggregate, borda_aggregate
from autofs_vnext.utils.io import read_table, ensure_dir

def _make_cv(cfg: AutoFSConfig, y: np.ndarray, groups: Optional[np.ndarray]):
    if cfg.cv.scheme == "stratified_kfold":
        return StratifiedKFold(n_splits=cfg.cv.n_splits, shuffle=cfg.cv.shuffle, random_state=cfg.cv.random_state)
    if cfg.cv.scheme == "group_kfold":
        if groups is None:
            raise ValueError("group_kfold requires groups")
        return GroupKFold(n_splits=cfg.cv.n_splits)
    return KFold(n_splits=cfg.cv.n_splits, shuffle=cfg.cv.shuffle, random_state=cfg.cv.random_state)

def _write_df(df: pd.DataFrame, path_base: Path) -> str:
    """Write parquet if an engine exists; otherwise CSV."""
    try:
        df.to_parquet(str(path_base) + ".parquet", index=False)  # type: ignore[call-arg]
        return str(path_base) + ".parquet"
    except Exception:
        df.to_csv(str(path_base) + ".csv", index=False)
        return str(path_base) + ".csv"

def run_autofs(cfg: AutoFSConfig) -> Path:
    autodiscover()

    out_dir = ensure_dir(cfg.output.dir) / cfg.output.run_name
    ensure_dir(str(out_dir))
    (out_dir / "method_results").mkdir(exist_ok=True)

    start = time.time()
    df = read_table(cfg.input.path, cfg.input.format)

    if cfg.input.target not in df.columns:
        raise ValueError(f"Target column '{cfg.input.target}' not found in input.")

    df = df.copy()
    for c in cfg.input.drop_columns:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    y = df[cfg.input.target].values
    X_df = df.drop(columns=[cfg.input.target])

    # Optional row cap
    if cfg.budget.max_rows is not None and len(X_df) > cfg.budget.max_rows:
        X_df = X_df.iloc[: cfg.budget.max_rows].reset_index(drop=True)
        y = y[: cfg.budget.max_rows]

    groups = None
    if cfg.cv.group_column and cfg.cv.group_column in X_df.columns:
        groups = X_df[cfg.cv.group_column].values

    pre = Preprocessor(cfg.preprocess)
    Xt = pre.fit_transform(X_df)
    feature_names = pre.feature_names_out_ or []
    cv = _make_cv(cfg, y, groups)

    (out_dir / "preprocessing_manifest.json").write_text(json.dumps(pre.manifest(), indent=2), encoding="utf-8")

    phase_outputs: List[pd.DataFrame] = []
    all_method_frames: List[pd.DataFrame] = []
    current_features_mask = np.ones(len(feature_names), dtype=bool)

    for phase in cfg.phases:
        rank_frames: List[pd.DataFrame] = []

        for m in phase.methods:
            if not m.get("enabled", True):
                continue

            name = m["name"]
            params = m.get("params", {})
            method_cls = REGISTRY.get(name)
            method = method_cls(**params)

            X_phase = Xt[:, current_features_mask]
            fn_phase = [f for f, keep in zip(feature_names, current_features_mask) if keep]

            t0 = time.time()
            method.fit(
                X_phase, y,
                feature_names=fn_phase,
                task=cfg.task,
                cv=cv,
                groups=groups,
                random_state=cfg.cv.random_state,
            )
            dfm = method.score_features()
            dfm["runtime_sec"] = round(time.time() - t0, 6)
            dfm["phase"] = phase.name

            safe_name = REGISTRY.resolve(name).replace("/", "_")
            _write_df(dfm, out_dir / "method_results" / f"{phase.name}__{safe_name}")

            all_method_frames.append(dfm)
            rank_frames.append(dfm[["feature", "rank"]].copy())

        if not rank_frames:
            continue

        if cfg.aggregation.strategy == "borda":
            agg = borda_aggregate(rank_frames)
            score_col = "borda_score"
        else:
            agg = rrf_aggregate(rank_frames, k=cfg.aggregation.rrf_k)
            score_col = "rrf_score"

        agg["method_votes"] = agg["feature"].map(lambda f: sum((rf["feature"] == f).any() for rf in rank_frames))
        agg["phase"] = phase.name
        phase_outputs.append(agg)

        if phase.top_k is not None:
            top = set(agg.nsmallest(phase.top_k, "rank")["feature"].tolist())
            current_features_mask = np.array([f in top for f in feature_names], dtype=bool)

    # Stability selection
    stability_frame = None
    if cfg.stability.enabled:
        stab_cls = REGISTRY.get("stability_selection")
        stab = stab_cls(
            n_resamples=cfg.stability.n_resamples,
            sample_frac=cfg.stability.sample_frac,
            base_method=cfg.stability.base_method,
            base_params=cfg.stability.base_params,
        )
        stab.fit(Xt, y, feature_names=feature_names, task=cfg.task, cv=cv, groups=groups, random_state=cfg.cv.random_state)
        stability_frame = stab.score_features()
        _write_df(stability_frame, out_dir / "stability_scores")

    redundancy_frame = None
    if cfg.redundancy.enabled:
        red_cls = REGISTRY.get("redundancy_correlation")
        red = red_cls(method=cfg.redundancy.corr_method, threshold=cfg.redundancy.corr_threshold)
        red.fit(Xt, y, feature_names=feature_names, task=cfg.task, cv=cv, groups=groups, random_state=cfg.cv.random_state)
        redundancy_frame = red.score_features()
        _write_df(redundancy_frame, out_dir / "correlation_clusters")

    if not phase_outputs:
        raise RuntimeError("No phases produced any results. Check your config methods.")

    final_rank_frames = [ph[["feature", "rank"]].copy() for ph in phase_outputs]
    if cfg.aggregation.strategy == "borda":
        final = borda_aggregate(final_rank_frames)
        base_score_col = "borda_score"
    else:
        final = rrf_aggregate(final_rank_frames, k=cfg.aggregation.rrf_k)
        base_score_col = "rrf_score"

    final = final.rename(columns={"rank": "agg_rank"})
    final["agg_score"] = final[base_score_col]
    final.drop(columns=[base_score_col], inplace=True)

    if stability_frame is not None and cfg.aggregation.use_stability:
        stab = stability_frame[["feature", "stability"]].copy()
        final = final.merge(stab, on="feature", how="left")
        final["stability"] = final["stability"].fillna(0.0)
        final["final_score"] = final["agg_score"] * (1.0 - cfg.aggregation.stability_weight) + final["stability"] * cfg.aggregation.stability_weight
    else:
        final["stability"] = np.nan
        final["final_score"] = final["agg_score"]

    if redundancy_frame is not None and cfg.aggregation.redundancy_penalty:
        sizes = redundancy_frame.groupby("cluster_id")["feature"].count().rename("cluster_size").reset_index()
        red = redundancy_frame.merge(sizes, on="cluster_id", how="left")[["feature","cluster_size"]]
        final = final.merge(red, on="feature", how="left")
        final["cluster_size"] = final["cluster_size"].fillna(1)
        penalty = (final["cluster_size"] - 1) / final["cluster_size"]
        final["final_score"] = final["final_score"] * (1.0 - cfg.aggregation.redundancy_weight * penalty)
    else:
        final["cluster_size"] = np.nan

    final["final_rank"] = final["final_score"].rank(ascending=False, method="min").astype(int)
    final = final.sort_values(["final_rank","feature"]).reset_index(drop=True)

    sel = cfg.selection
    if sel.policy == "score_threshold":
        if sel.score_threshold is None:
            raise ValueError("score_threshold policy requires selection.score_threshold")
        selected = final[final["final_score"] >= sel.score_threshold].copy()
    elif sel.policy == "stability_threshold":
        if sel.stability_threshold is None:
            raise ValueError("stability_threshold policy requires selection.stability_threshold")
        if stability_frame is None:
            raise ValueError("stability_threshold policy requires stability.enabled=true")
        selected = final[final["stability"] >= sel.stability_threshold].copy()
    else:
        selected = final.nsmallest(sel.top_k, "final_rank").copy()

    _write_df(final, out_dir / "feature_ranking")
    selected[["feature","final_rank","final_score","stability"]].to_csv(out_dir / "selected_features.csv", index=False)

    if all_method_frames:
        all_df = pd.concat(all_method_frames, ignore_index=True)
        _write_df(all_df, out_dir / "all_method_scores")

    meta = {
        "task": cfg.task,
        "input": asdict(cfg.input),
        "duration_sec": round(time.time() - start, 3),
        "n_rows": int(len(df)),
        "n_features_raw": int(df.shape[1] - 1),
        "n_features_after_preprocess": int(len(feature_names)),
        "selected_count": int(len(selected)),
        "phases": [p.name for p in cfg.phases],
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return out_dir
