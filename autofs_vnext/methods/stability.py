from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd

from autofs_vnext.core.registry import FeatureSelectorMethod, MethodMeta, REGISTRY
from autofs_vnext.methods.common import make_score_frame

class StabilitySelection(FeatureSelectorMethod):
    meta = MethodMeta(
        name="stability_selection",
        family="stability",
        tasks={"classification","regression"},
        compute="expensive",
        description="Resampling-based stability selection using an underlying base selector; reports selection frequency."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        n_resamples = int(self.params.get("n_resamples", 20))
        sample_frac = float(self.params.get("sample_frac", 0.75))
        base_method = self.params.get("base_method", "l1_logistic" if task=="classification" else "l1_linear")
        base_params = self.params.get("base_params", {})

        rs = np.random.RandomState(random_state)
        n = len(y)
        k = max(1, int(sample_frac * n))

        counts = np.zeros(len(feature_names), dtype=float)

        base_cls = REGISTRY.get(base_method)

        for _ in range(n_resamples):
            idx = rs.choice(n, size=k, replace=False)
            Xs = X[idx]
            ys = y[idx]
            base = base_cls(**base_params)
            base.fit(Xs, ys, feature_names=feature_names, task=task, cv=cv, groups=None, random_state=int(rs.randint(0, 2**31-1)))
            dfb = base.score_features()

            # decide selected: top fraction OR nonzero coefficients? here: top_k
            top_k = int(self.params.get("top_k", max(10, int(0.1 * len(feature_names)))))
            selected = set(dfb.nsmallest(top_k, "rank")["feature"].tolist())
            for j, f in enumerate(feature_names):
                if f in selected:
                    counts[j] += 1.0

        stability = counts / float(n_resamples)
        self._out = pd.DataFrame({"feature": feature_names, "stability": stability})
        self._out["rank"] = self._out["stability"].rank(ascending=False, method="min").astype(int)
        self._out = self._out.sort_values(["rank","feature"]).reset_index(drop=True)
        return self

    def score_features(self):
        # Map to canonical columns
        scores = self._out["stability"].values
        df = make_score_frame(self._out["feature"].tolist(), scores, method_name=self.meta.name, method_family=self.meta.family)
        df = df.drop(columns=["score"]).rename(columns={"score":"stability_score"}) if "score" in df.columns else df
        df2 = self._out.merge(df[["feature","rank","selected_flag","method_name","method_family","extra_json"]], on="feature", how="left")
        df2 = df2.rename(columns={"stability":"stability"})
        # Keep a consistent view for runner expectations
        return df2

REGISTRY.register(StabilitySelection)
