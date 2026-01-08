from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from autofs_vnext.core.registry import FeatureSelectorMethod, MethodMeta, REGISTRY

class RedundancyCorrelation(FeatureSelectorMethod):
    meta = MethodMeta(
        name="redundancy_correlation",
        family="redundancy",
        tasks={"classification","regression"},
        compute="expensive",
        description="Correlation clustering for redundancy detection (sampled; dense)."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        method = self.params.get("method", "pearson")
        threshold = float(self.params.get("threshold", 0.95))
        sample_rows = int(self.params.get("sample_rows", 5000))

        if hasattr(X, "toarray"):
            Xd = X[:sample_rows, :].toarray()
        else:
            Xd = np.asarray(X)[:sample_rows, :]

        # Standardize to avoid scale issues for Pearson
        Xd = Xd - Xd.mean(axis=0, keepdims=True)
        denom = Xd.std(axis=0, keepdims=True) + 1e-12
        Xd = Xd / denom

        corr = np.corrcoef(Xd, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)

        # Build graph edges for |corr|>=threshold
        n = corr.shape[0]
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a,b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i+1, n):
                if abs(corr[i,j]) >= threshold:
                    union(i,j)

        # assign cluster ids
        roots = [find(i) for i in range(n)]
        uniq = {}
        cid = 0
        cluster_id = np.zeros(n, dtype=int)
        for i, r in enumerate(roots):
            if r not in uniq:
                uniq[r] = cid
                cid += 1
            cluster_id[i] = uniq[r]

        self._df = pd.DataFrame({
            "feature": feature_names,
            "cluster_id": cluster_id,
        })
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RedundancyCorrelation)



import json
from sklearn.feature_selection import f_classif, f_regression

class CorrelationSelector(FeatureSelectorMethod):
    meta = MethodMeta(
        name="correlation_redundancy_filter",
        family="redundancy",
        tasks={"classification","regression"},
        compute="expensive",
        description="Correlation-based redundancy filter: clusters features and keeps one representative per cluster."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        threshold = float(self.params.get("threshold", 0.9))
        max_features = int(self.params.get("max_features", min(2000, X.shape[1])))
        sample_rows = int(self.params.get("sample_rows", min(20000, X.shape[0])))
        representative = self.params.get("representative", "univariate")  # univariate or variance

        # Sample for correlation computation
        rng = np.random.default_rng(random_state)
        row_idx = rng.choice(X.shape[0], size=min(sample_rows, X.shape[0]), replace=False)
        col_idx = np.arange(X.shape[1])
        if X.shape[1] > max_features:
            col_idx = rng.choice(X.shape[1], size=max_features, replace=False)
        Xs = X[row_idx][:, col_idx]
        fn = [feature_names[i] for i in col_idx]

        # Correlation matrix
        C = np.corrcoef(Xs, rowvar=False)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        A = (np.abs(C) >= threshold).astype(int)
        np.fill_diagonal(A, 0)

        # Union-find clustering
        n = A.shape[0]
        parent = np.arange(n)
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a,b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        edges = np.argwhere(A==1)
        for a,b in edges:
            union(int(a), int(b))

        roots = np.array([find(i) for i in range(n)], dtype=int)
        uniq = {}
        cid = 0
        cluster_id = np.zeros(n, dtype=int)
        for i,r in enumerate(roots):
            if int(r) not in uniq:
                uniq[int(r)] = cid
                cid += 1
            cluster_id[i] = uniq[int(r)]

        # Compute representative scores
        if representative == "variance":
            rep_scores = np.var(Xs, axis=0)
        else:
            # univariate score with y
            if task == "classification":
                rep_scores = f_classif(Xs, y[row_idx])[0]
            else:
                rep_scores = f_regression(Xs, y[row_idx])[0]
            rep_scores = np.nan_to_num(rep_scores, nan=0.0, posinf=0.0, neginf=0.0)

        kept = []
        dropped = []
        for c in np.unique(cluster_id):
            members = np.where(cluster_id == c)[0]
            if len(members) == 1:
                kept.append(members[0])
                continue
            best = members[np.argmax(rep_scores[members])]
            kept.append(best)
            for m in members:
                if m != best:
                    dropped.append(m)

        kept_mask = np.zeros(n, dtype=bool)
        kept_mask[kept] = True

        # produce score frame: higher score for kept; but keep rank based on rep_scores
        df = pd.DataFrame({
            "feature": fn,
            "score": rep_scores,
            "cluster_id": cluster_id,
            "selected_flag": kept_mask.astype(int),
        })
        df["rank"] = df["score"].rank(ascending=False, method="min").astype(int)
        df["method_name"] = self.meta.name
        df["method_family"] = self.meta.family
        df["extra_json"] = [json.dumps({"cluster_id": int(c), "kept": bool(k)}) for c,k in zip(cluster_id, kept_mask)]
        self._df = df.sort_values(["rank","feature"]).reset_index(drop=True)
        return self

    def score_features(self):
        # Conform to canonical output columns; extra columns allowed
        return self._df[["feature","score","rank","selected_flag","method_name","method_family","extra_json","cluster_id"]]

REGISTRY.register(CorrelationSelector)
