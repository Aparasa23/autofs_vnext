from __future__ import annotations

from typing import List
import numpy as np
import json
from sklearn.linear_model import LogisticRegression, ElasticNet, RidgeClassifier, BayesianRidge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from autofs_vnext.core.registry import FeatureSelectorMethod, MethodMeta, REGISTRY
from autofs_vnext.methods.common import make_score_frame

class L1Logistic(FeatureSelectorMethod):
    meta = MethodMeta(
        name="l1_logistic",
        family="embedded",
        tasks={"classification"},
        compute="medium",
        description="L1-penalized logistic regression; absolute coefficient magnitude as importance."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        C = float(self.params.get("C", 1.0))
        max_iter = int(self.params.get("max_iter", 2000))
        solver = self.params.get("solver", "liblinear")
        model = LogisticRegression(penalty="l1", C=C, solver=solver, max_iter=max_iter, random_state=random_state)
        model.fit(X, y)
        coefs = np.abs(model.coef_).mean(axis=0)
        self._df = make_score_frame(feature_names, coefs, method_name=self.meta.name, method_family=self.meta.family)
        return self

    def score_features(self):
        return self._df

REGISTRY.register(L1Logistic)

class L1Linear(FeatureSelectorMethod):
    meta = MethodMeta(
        name="l1_linear",
        family="embedded",
        tasks={"regression"},
        compute="medium",
        description="ElasticNet with l1_ratio=1 (Lasso); absolute coefficient magnitude as importance."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        alpha = float(self.params.get("alpha", 0.01))
        max_iter = int(self.params.get("max_iter", 5000))
        model = ElasticNet(alpha=alpha, l1_ratio=1.0, max_iter=max_iter, random_state=random_state)
        model.fit(X, y)
        coefs = np.abs(model.coef_)
        self._df = make_score_frame(feature_names, coefs, method_name=self.meta.name, method_family=self.meta.family)
        return self

    def score_features(self):
        return self._df

REGISTRY.register(L1Linear)

class RidgeImportance(FeatureSelectorMethod):
    meta = MethodMeta(
        name="ridge_importance",
        family="embedded",
        tasks={"classification"},
        compute="cheap",
        description="RidgeClassifier; absolute coefficient magnitude."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        alpha = float(self.params.get("alpha", 1.0))
        model = RidgeClassifier(alpha=alpha, random_state=random_state)
        model.fit(X, y)
        coefs = np.abs(model.coef_).mean(axis=0)
        self._df = make_score_frame(feature_names, coefs, method_name=self.meta.name, method_family=self.meta.family)
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RidgeImportance)

class RFImportanceClf(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rf_importance_clf",
        family="embedded",
        tasks={"classification"},
        compute="expensive",
        description="RandomForestClassifier impurity-based feature importances."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        n_estimators = int(self.params.get("n_estimators", 400))
        max_depth = self.params.get("max_depth", None)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=self.params.get("n_jobs", -1),
            random_state=random_state,
        )
        model.fit(X, y)
        scores = model.feature_importances_
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family)
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RFImportanceClf)

class RFImportanceReg(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rf_importance_reg",
        family="embedded",
        tasks={"regression"},
        compute="expensive",
        description="RandomForestRegressor impurity-based feature importances."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        n_estimators = int(self.params.get("n_estimators", 400))
        max_depth = self.params.get("max_depth", None)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=self.params.get("n_jobs", -1),
            random_state=random_state,
        )
        model.fit(X, y)
        scores = model.feature_importances_
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family)
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RFImportanceReg)
# -----------------------------
# Additional embedded methods (Milestone 5/7 coverage)
# -----------------------------

class BayesianRidgeImportance(FeatureSelectorMethod):
    meta = MethodMeta(
        name="bayesian_ridge_importance",
        family="embedded",
        tasks={"regression"},
        compute="medium",
        description="BayesianRidge coefficients; absolute coefficient magnitude as importance."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        model = BayesianRidge(**{k:v for k,v in self.params.items() if k in {
            "n_iter","tol","alpha_1","alpha_2","lambda_1","lambda_2","compute_score","fit_intercept","normalize"
        }})
        model.fit(X, y)
        scores = np.abs(model.coef_)
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family)
        return self

    def score_features(self):
        return self._df

REGISTRY.register(BayesianRidgeImportance)


class RFFoldAggregatedImportance(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rf_fold_aggregated_importance",
        family="embedded",
        tasks={"classification","regression"},
        compute="expensive",
        description="Fold-aggregated RandomForest importances: mean and std across CV folds."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        from sklearn.model_selection import StratifiedKFold, KFold
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        n_estimators = int(self.params.get("n_estimators", 500))
        max_depth = self.params.get("max_depth", None)
        n_jobs = int(self.params.get("n_jobs", -1))
        n_splits = int(self.params.get("n_splits", 5))

        if cv is None:
            if task == "classification":
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            else:
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        importances = []
        for train_idx, _ in cv.split(X, y):
            Xtr = X[train_idx]
            ytr = y[train_idx]
            if task == "classification":
                model = RandomForestClassifier(
                    n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs, max_depth=max_depth
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs, max_depth=max_depth
                )
            model.fit(Xtr, ytr)
            importances.append(np.asarray(model.feature_importances_, dtype=float))

        imp = np.vstack(importances)
        mean_imp = imp.mean(axis=0)
        std_imp = imp.std(axis=0)

        df = make_score_frame(feature_names, mean_imp, method_name=self.meta.name, method_family=self.meta.family)
        df["extra_json"] = [
            json.dumps({"std_importance": float(s)}) for s in std_imp
        ]
        self._df = df
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RFFoldAggregatedImportance)



from autofs_vnext.core.aggregation import rrf_aggregate

class RRFVarImp(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rrf_varimp",
        family="hybrid",
        tasks={"classification","regression"},
        compute="expensive",
        description="Aggregates ranks from multiple internal importance methods using Reciprocal Rank Fusion (RRF)."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        # Internal methods: univariate MI, RF importance, and L1 model (when supported).
        from autofs_vnext.methods.filters import UnivariateMI
        from autofs_vnext.methods.embedded import RFImportanceReg, RFImportanceClf, L1Logistic, L1Linear

        frames = []
        # MI
        mi = UnivariateMI(params={"dummy": 0})
        mi.fit(X, y, feature_names=feature_names, task=task, cv=cv, groups=groups, random_state=random_state)
        frames.append(mi.score_features()[["feature","rank"]])

        # RF importance
        if task == "classification":
            rf = RFImportanceClf(params={"n_estimators": int(self.params.get("n_estimators", 400))})
        else:
            rf = RFImportanceReg(params={"n_estimators": int(self.params.get("n_estimators", 400))})
        rf.fit(X, y, feature_names=feature_names, task=task, cv=cv, groups=groups, random_state=random_state)
        frames.append(rf.score_features()[["feature","rank"]])

        # L1 model
        if task == "classification":
            l1 = L1Logistic(params={"C": float(self.params.get("C", 1.0))})
        else:
            l1 = L1Linear(params={"alpha": float(self.params.get("alpha", 0.01))})
        l1.fit(X, y, feature_names=feature_names, task=task, cv=cv, groups=groups, random_state=random_state)
        frames.append(l1.score_features()[["feature","rank"]])

        k = int(self.params.get("k", 60))
        agg = rrf_aggregate(frames, k=k)
        # Convert to canonical format
        df = agg.rename(columns={"rrf_score":"score"})
        df["selected_flag"] = 0
        df.loc[df["rank"] <= int(self.params.get("top_k", max(10, int(0.2 * len(feature_names))))), "selected_flag"] = 1
        df["method_name"] = self.meta.name
        df["method_family"] = self.meta.family
        df["extra_json"] = "{}"
        self._df = df[["feature","score","rank","selected_flag","method_name","method_family","extra_json"]]
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RRFVarImp)
