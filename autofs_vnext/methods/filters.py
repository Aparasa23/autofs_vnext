from __future__ import annotations

from typing import Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression, f_classif, f_regression, chi2, SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe
from sklearn.preprocessing import MinMaxScaler

from autofs_vnext.core.registry import FeatureSelectorMethod, MethodMeta, REGISTRY
from autofs_vnext.methods.common import make_score_frame

class VarianceFilter(FeatureSelectorMethod):
    meta = MethodMeta(
        name="variance",
        family="filter",
        tasks={"classification","regression"},
        compute="cheap",
        description="Unsupervised variance thresholding on transformed feature matrix."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        thr = float(self.params.get("threshold", 0.0))
        sel = VarianceThreshold(threshold=thr)
        sel.fit(X)
        scores = sel.variances_
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family)
        return self

    def score_features(self):
        return self._df

REGISTRY.register(VarianceFilter)

class UnivariateMI(FeatureSelectorMethod):
    meta = MethodMeta(
        name="univariate_mutual_info",
        family="filter",
        tasks={"classification","regression"},
        compute="medium",
        description="Mutual information between each feature and target (univariate)."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        if task == "classification":
            scores = mutual_info_classif(X, y, random_state=random_state)
        else:
            scores = mutual_info_regression(X, y, random_state=random_state)
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family)
        return self

    def score_features(self):
        return self._df

REGISTRY.register(UnivariateMI)

class UnivariateFClassif(FeatureSelectorMethod):
    meta = MethodMeta(
        name="univariate_f_classif",
        family="filter",
        tasks={"classification"},
        compute="cheap",
        description="ANOVA F-test for classification."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        scores, _ = f_classif(X, y)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family)
        return self

    def score_features(self):
        return self._df

REGISTRY.register(UnivariateFClassif)

class UnivariateFRegression(FeatureSelectorMethod):
    meta = MethodMeta(
        name="univariate_f_regression",
        family="filter",
        tasks={"regression"},
        compute="cheap",
        description="F-test for linear dependency for regression."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        scores, _ = f_regression(X, y)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family)
        return self

    def score_features(self):
        return self._df

REGISTRY.register(UnivariateFRegression)

class UnivariateChi2(FeatureSelectorMethod):
    meta = MethodMeta(
        name="univariate_chi2",
        family="filter",
        tasks={"classification"},
        compute="cheap",
        description="Chi-square test (requires non-negative features). Will min-max scale input."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        Xn = MinMaxScaler().fit_transform(X)
        scores, _ = chi2(Xn, y)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family)
        return self

    def score_features(self):
        return self._df

REGISTRY.register(UnivariateChi2)

class VIFFilter(FeatureSelectorMethod):
    meta = MethodMeta(
        name="vif_filter",
        family="filter",
        tasks={"classification","regression"},
        compute="expensive",
        description="Variance Inflation Factor filter (approx; operates on dense sample)."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        # VIF requires dense and can be expensive; we sample columns if huge
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression

        max_cols = int(self.params.get("max_cols", 200))
        sample_rows = int(self.params.get("sample_rows", 5000))

        Xd = X
        if hasattr(X, "toarray"):
            # sparse -> dense sample
            Xd = X[:sample_rows, :].toarray()
        else:
            Xd = np.asarray(X)[:sample_rows, :]

        if Xd.shape[1] > max_cols:
            # compute VIF on a subset, rank those; others get 0
            idx = np.random.RandomState(random_state).choice(Xd.shape[1], size=max_cols, replace=False)
        else:
            idx = np.arange(Xd.shape[1])

        vifs = np.zeros(Xd.shape[1], dtype=float)
        lr = LinearRegression()
        for j in idx:
            X_other = np.delete(Xd, j, axis=1)
            yj = Xd[:, j]
            if X_other.shape[1] == 0:
                vifs[j] = 1.0
                continue
            lr.fit(X_other, yj)
            r2 = lr.score(X_other, yj)
            if r2 >= 0.999999:
                vifs[j] = 1e6
            else:
                vifs[j] = 1.0 / (1.0 - r2)

        # Invert: lower VIF better -> score = 1/vif
        scores = 1.0 / np.clip(vifs, 1e-12, None)
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family, extra={"note":"score=1/VIF"})
        return self

    def score_features(self):
        return self._df

REGISTRY.register(VIFFilter)

# -----------------------------
# SelectKBest policy variants (Milestone 5 coverage)
# -----------------------------

class UnivariateSelectKBestGeneric(FeatureSelectorMethod):
    meta = MethodMeta(
        name="univariate_selectkbest_generic",
        family="filter",
        tasks={"classification","regression"},
        compute="cheap",
        description="Generic SelectKBest (task-aware score_func). Use params: k."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        k = self.params.get("k", "auto")
        if k == "auto":
            k = max(10, int(0.2 * len(feature_names)))
        k = int(k)
        score_func = f_classif if task == "classification" else f_regression
        sel = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        sel.fit(X, y)
        scores = np.nan_to_num(sel.scores_, nan=0.0, posinf=0.0, neginf=0.0)
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    selected_mask=sel.get_support())
        return self

    def score_features(self):
        return self._df

REGISTRY.register(UnivariateSelectKBestGeneric)


class UnivariateSelectPercentile(FeatureSelectorMethod):
    meta = MethodMeta(
        name="univariate_selectpercentile",
        family="filter",
        tasks={"classification","regression"},
        compute="cheap",
        description="SelectPercentile with task-aware score_func. Param: percentile."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        percentile = int(self.params.get("percentile", 20))
        score_func = f_classif if task == "classification" else f_regression
        sel = SelectPercentile(score_func=score_func, percentile=percentile)
        sel.fit(X, y)
        scores = np.nan_to_num(sel.scores_, nan=0.0, posinf=0.0, neginf=0.0)
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    selected_mask=sel.get_support())
        return self

    def score_features(self):
        return self._df

REGISTRY.register(UnivariateSelectPercentile)


class UnivariateSelectFpr(FeatureSelectorMethod):
    meta = MethodMeta(
        name="univariate_selectfpr",
        family="filter",
        tasks={"classification","regression"},
        compute="cheap",
        description="SelectFpr with task-aware score_func. Param: alpha."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        alpha = float(self.params.get("alpha", 0.05))
        score_func = f_classif if task == "classification" else f_regression
        sel = SelectFpr(score_func=score_func, alpha=alpha)
        sel.fit(X, y)
        scores = np.nan_to_num(sel.scores_, nan=0.0, posinf=0.0, neginf=0.0)
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    selected_mask=sel.get_support())
        return self

    def score_features(self):
        return self._df

REGISTRY.register(UnivariateSelectFpr)


class UnivariateSelectFdr(FeatureSelectorMethod):
    meta = MethodMeta(
        name="univariate_selectfdr",
        family="filter",
        tasks={"classification","regression"},
        compute="cheap",
        description="SelectFdr with task-aware score_func. Param: alpha."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        alpha = float(self.params.get("alpha", 0.05))
        score_func = f_classif if task == "classification" else f_regression
        sel = SelectFdr(score_func=score_func, alpha=alpha)
        sel.fit(X, y)
        scores = np.nan_to_num(sel.scores_, nan=0.0, posinf=0.0, neginf=0.0)
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    selected_mask=sel.get_support())
        return self

    def score_features(self):
        return self._df

REGISTRY.register(UnivariateSelectFdr)


class UnivariateSelectFwe(FeatureSelectorMethod):
    meta = MethodMeta(
        name="univariate_selectfwe",
        family="filter",
        tasks={"classification","regression"},
        compute="cheap",
        description="SelectFwe with task-aware score_func. Param: alpha."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        alpha = float(self.params.get("alpha", 0.05))
        score_func = f_classif if task == "classification" else f_regression
        sel = SelectFwe(score_func=score_func, alpha=alpha)
        sel.fit(X, y)
        scores = np.nan_to_num(sel.scores_, nan=0.0, posinf=0.0, neginf=0.0)
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    selected_mask=sel.get_support())
        return self

    def score_features(self):
        return self._df

REGISTRY.register(UnivariateSelectFwe)
