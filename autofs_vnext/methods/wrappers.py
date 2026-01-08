from __future__ import annotations

from typing import List
import numpy as np
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.model_selection import StratifiedKFold, KFold

from autofs_vnext.core.registry import FeatureSelectorMethod, MethodMeta, REGISTRY
from autofs_vnext.methods.common import make_score_frame

class RFELogistic(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rfe_logistic",
        family="wrapper",
        tasks={"classification"},
        compute="expensive",
        description="RFE with LogisticRegression estimator."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        n_features_to_select = int(self.params.get("n_features_to_select", max(10, int(0.2 * len(feature_names)))))
        step = float(self.params.get("step", 0.1))
        est = LogisticRegression(max_iter=int(self.params.get("max_iter", 2000)), solver="liblinear", random_state=random_state)
        rfe = RFE(estimator=est, n_features_to_select=n_features_to_select, step=step)
        rfe.fit(X, y)
        # scores: invert ranking
        scores = 1.0 / rfe.ranking_.astype(float)
        selected = rfe.support_
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family, selected_mask=selected,
                                    extra={"n_features_to_select": n_features_to_select, "step": step})
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RFELogistic)

class RFETreeClf(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rfe_tree_clf",
        family="wrapper",
        tasks={"classification"},
        compute="expensive",
        description="RFE with DecisionTreeClassifier estimator."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        n_features_to_select = int(self.params.get("n_features_to_select", max(10, int(0.2 * len(feature_names)))))
        step = float(self.params.get("step", 0.1))
        est = DecisionTreeClassifier(random_state=random_state)
        rfe = RFE(estimator=est, n_features_to_select=n_features_to_select, step=step)
        rfe.fit(X, y)
        scores = 1.0 / rfe.ranking_.astype(float)
        selected = rfe.support_
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family, selected_mask=selected)
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RFETreeClf)

class RFEAdaBoostReg(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rfe_adaboost_reg",
        family="wrapper",
        tasks={"regression"},
        compute="expensive",
        description="RFE with AdaBoostRegressor estimator."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        n_features_to_select = int(self.params.get("n_features_to_select", max(10, int(0.2 * len(feature_names)))))
        step = float(self.params.get("step", 0.1))
        est = AdaBoostRegressor(random_state=random_state)
        rfe = RFE(estimator=est, n_features_to_select=n_features_to_select, step=step)
        rfe.fit(X, y)
        scores = 1.0 / rfe.ranking_.astype(float)
        selected = rfe.support_
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family, selected_mask=selected)
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RFEAdaBoostReg)

class RFECVLinearReg(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rfecv_linear_reg",
        family="wrapper",
        tasks={"regression"},
        compute="expensive",
        description="RFECV with LinearRegression estimator."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        step = float(self.params.get("step", 0.1))
        est = LinearRegression()
        rfecv = RFECV(estimator=est, step=step, cv=cv, scoring=self.params.get("scoring", "r2"), n_jobs=self.params.get("n_jobs", -1))
        rfecv.fit(X, y)
        scores = 1.0 / rfecv.ranking_.astype(float)
        selected = rfecv.support_
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family, selected_mask=selected,
                                    extra={"n_features_selected": int(selected.sum())})
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RFECVLinearReg)

class PermutationImportanceLogistic(FeatureSelectorMethod):
    meta = MethodMeta(
        name="perm_importance_logistic",
        family="wrapper",
        tasks={"classification"},
        compute="expensive",
        description="Permutation importance with LogisticRegression; importance = mean decrease in score."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        est = LogisticRegression(max_iter=int(self.params.get("max_iter", 2000)), solver="liblinear", random_state=random_state)
        est.fit(X, y)
        scoring = self.params.get("scoring", "accuracy")
        n_repeats = int(self.params.get("n_repeats", 10))
        res = permutation_importance(est, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=self.params.get("n_jobs", -1))
        scores = res.importances_mean
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    extra={"scoring": scoring, "n_repeats": n_repeats})
        return self

    def score_features(self):
        return self._df

REGISTRY.register(PermutationImportanceLogistic)

class GeneticPlaceholder(FeatureSelectorMethod):
    meta = MethodMeta(
        name="genetic_placeholder",
        family="wrapper",
        tasks={"classification","regression"},
        compute="expensive",
        description="Placeholder for genetic selection. Implement via sklearn-genetic-opt or custom GA in an extension package."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        scores = np.zeros(len(feature_names), dtype=float)
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    extra={"status":"placeholder"})
        return self

    def score_features(self):
        return self._df

REGISTRY.register(GeneticPlaceholder)

class BorutaRFPlaceholder(FeatureSelectorMethod):
    meta = MethodMeta(
        name="boruta_rf",
        family="wrapper",
        tasks={"classification","regression"},
        compute="expensive",
        description="Placeholder for Boruta (requires boruta_py or boruta). Add via plugin to avoid hard dependency."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        scores = np.zeros(len(feature_names), dtype=float)
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    extra={"status":"placeholder"})
        return self

    def score_features(self):
        return self._df

REGISTRY.register(BorutaRFPlaceholder)

class MICPlaceholder(FeatureSelectorMethod):
    meta = MethodMeta(
        name="mic_placeholder",
        family="filter",
        tasks={"classification","regression"},
        compute="expensive",
        description="Placeholder for MIC (e.g., minepy). Add via plugin."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        scores = np.zeros(len(feature_names), dtype=float)
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    extra={"status":"placeholder"})
        return self

    def score_features(self):
        return self._df

REGISTRY.register(MICPlaceholder)

class RRFPlaceholder(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rrf_placeholder",
        family="filter",
        tasks={"classification","regression"},
        compute="cheap",
        description="Placeholder for RRF varimp from R packages. Not applicable directly in Python core."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        scores = np.zeros(len(feature_names), dtype=float)
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    extra={"status":"placeholder"})
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RRFPlaceholder)
# -----------------------------
# Additional wrapper methods (Milestone 5 coverage)
# -----------------------------

class RFELassoRegressor(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rfe_lasso",
        family="wrapper",
        tasks={"regression"},
        compute="expensive",
        description="RFE with Lasso (ElasticNet l1_ratio=1)."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        alpha = float(self.params.get("alpha", 0.01))
        max_iter = int(self.params.get("max_iter", 5000))
        n_features_to_select = int(self.params.get("n_features_to_select", max(10, int(0.2 * len(feature_names)))))
        step = float(self.params.get("step", 0.1))
        estimator = ElasticNet(alpha=alpha, l1_ratio=1.0, max_iter=max_iter, random_state=random_state)
        selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
        selector.fit(X, y)
        # Rank 1 is selected. Convert to importance by inverse rank.
        ranks = selector.ranking_.astype(float)
        scores = 1.0 / ranks
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    selected_mask=(selector.support_))
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RFELassoRegressor)


class RFEDecisionTreeRegressor(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rfe_decision_tree_reg",
        family="wrapper",
        tasks={"regression"},
        compute="expensive",
        description="RFE with DecisionTreeRegressor estimator."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        n_features_to_select = int(self.params.get("n_features_to_select", max(10, int(0.2 * len(feature_names)))))
        step = float(self.params.get("step", 0.1))
        max_depth = self.params.get("max_depth", None)
        estimator = DecisionTreeRegressor(random_state=random_state, max_depth=max_depth)
        selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
        selector.fit(X, y)
        ranks = selector.ranking_.astype(float)
        scores = 1.0 / ranks
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    selected_mask=(selector.support_))
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RFEDecisionTreeRegressor)


class RFEExtraTreesRegressor(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rfe_extra_trees_reg",
        family="wrapper",
        tasks={"regression"},
        compute="expensive",
        description="RFE with ExtraTreesRegressor estimator."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        n_estimators = int(self.params.get("n_estimators", 300))
        max_depth = self.params.get("max_depth", None)
        n_jobs = int(self.params.get("n_jobs", -1))
        n_features_to_select = int(self.params.get("n_features_to_select", max(10, int(0.2 * len(feature_names)))))
        step = float(self.params.get("step", 0.1))
        estimator = ExtraTreesRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs, max_depth=max_depth
        )
        selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
        selector.fit(X, y)
        ranks = selector.ranking_.astype(float)
        scores = 1.0 / ranks
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    selected_mask=(selector.support_))
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RFEExtraTreesRegressor)


class RFERandomForestRegressor(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rfe_random_forest_reg",
        family="wrapper",
        tasks={"regression"},
        compute="expensive",
        description="RFE with RandomForestRegressor estimator."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        n_estimators = int(self.params.get("n_estimators", 300))
        max_depth = self.params.get("max_depth", None)
        n_jobs = int(self.params.get("n_jobs", -1))
        n_features_to_select = int(self.params.get("n_features_to_select", max(10, int(0.2 * len(feature_names)))))
        step = float(self.params.get("step", 0.1))
        estimator = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs, max_depth=max_depth
        )
        selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
        selector.fit(X, y)
        ranks = selector.ranking_.astype(float)
        scores = 1.0 / ranks
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    selected_mask=(selector.support_))
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RFERandomForestRegressor)


class RFECVRandomForestRegressor(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rfecv_random_forest_reg",
        family="wrapper",
        tasks={"regression"},
        compute="expensive",
        description="RFECV with RandomForestRegressor estimator."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        n_estimators = int(self.params.get("n_estimators", 300))
        max_depth = self.params.get("max_depth", None)
        n_jobs = int(self.params.get("n_jobs", -1))
        step = float(self.params.get("step", 0.1))
        min_features_to_select = int(self.params.get("min_features_to_select", max(5, int(0.05 * len(feature_names)))))
        scoring = self.params.get("scoring", "r2")
        estimator = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs, max_depth=max_depth
        )
        if cv is None:
            cv = KFold(n_splits=int(self.params.get("n_splits", 5)), shuffle=True, random_state=random_state)
        selector = RFECV(
            estimator=estimator,
            step=step,
            cv=cv,
            scoring=scoring,
            min_features_to_select=min_features_to_select,
            n_jobs=n_jobs,
        )
        selector.fit(X, y)
        ranks = selector.ranking_.astype(float)
        scores = 1.0 / ranks
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    selected_mask=(selector.support_))
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RFECVRandomForestRegressor)


class RFECVLassoRegressor(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rfecv_lasso",
        family="wrapper",
        tasks={"regression"},
        compute="expensive",
        description="RFECV with Lasso (ElasticNet l1_ratio=1)."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        alpha = float(self.params.get("alpha", 0.01))
        max_iter = int(self.params.get("max_iter", 5000))
        step = float(self.params.get("step", 0.1))
        min_features_to_select = int(self.params.get("min_features_to_select", max(5, int(0.05 * len(feature_names)))))
        scoring = self.params.get("scoring", "r2")
        estimator = ElasticNet(alpha=alpha, l1_ratio=1.0, max_iter=max_iter, random_state=random_state)
        if cv is None:
            cv = KFold(n_splits=int(self.params.get("n_splits", 5)), shuffle=True, random_state=random_state)
        selector = RFECV(
            estimator=estimator,
            step=step,
            cv=cv,
            scoring=scoring,
            min_features_to_select=min_features_to_select,
            n_jobs=int(self.params.get("n_jobs", -1)),
        )
        selector.fit(X, y)
        ranks = selector.ranking_.astype(float)
        scores = 1.0 / ranks
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    selected_mask=(selector.support_))
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RFECVLassoRegressor)


class RFECVLogistic(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rfecv_logistic",
        family="wrapper",
        tasks={"classification"},
        compute="expensive",
        description="RFECV with LogisticRegression estimator."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        C = float(self.params.get("C", 1.0))
        penalty = self.params.get("penalty", "l2")
        solver = self.params.get("solver", "liblinear")
        step = float(self.params.get("step", 0.1))
        min_features_to_select = int(self.params.get("min_features_to_select", max(5, int(0.05 * len(feature_names)))))
        scoring = self.params.get("scoring", "roc_auc")
        estimator = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=int(self.params.get("max_iter", 2000)))
        if cv is None:
            cv = StratifiedKFold(n_splits=int(self.params.get("n_splits", 5)), shuffle=True, random_state=random_state)
        selector = RFECV(
            estimator=estimator,
            step=step,
            cv=cv,
            scoring=scoring,
            min_features_to_select=min_features_to_select,
            n_jobs=int(self.params.get("n_jobs", -1)),
        )
        selector.fit(X, y)
        ranks = selector.ranking_.astype(float)
        scores = 1.0 / ranks
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    selected_mask=(selector.support_))
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RFECVLogistic)


class RFECVAdaBoostRegressor(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rfecv_adaboost_reg",
        family="wrapper",
        tasks={"regression"},
        compute="expensive",
        description="RFECV with AdaBoostRegressor estimator."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        n_estimators = int(self.params.get("n_estimators", 300))
        learning_rate = float(self.params.get("learning_rate", 0.05))
        step = float(self.params.get("step", 0.1))
        min_features_to_select = int(self.params.get("min_features_to_select", max(5, int(0.05 * len(feature_names)))))
        scoring = self.params.get("scoring", "r2")
        estimator = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
        if cv is None:
            cv = KFold(n_splits=int(self.params.get("n_splits", 5)), shuffle=True, random_state=random_state)
        selector = RFECV(
            estimator=estimator,
            step=step,
            cv=cv,
            scoring=scoring,
            min_features_to_select=min_features_to_select,
            n_jobs=int(self.params.get("n_jobs", -1)),
        )
        selector.fit(X, y)
        ranks = selector.ranking_.astype(float)
        scores = 1.0 / ranks
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    selected_mask=(selector.support_))
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RFECVAdaBoostRegressor)


class RFELinearSVM(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rfe_linear_svm",
        family="wrapper",
        tasks={"classification","regression"},
        compute="expensive",
        description="RFE with LinearSVC (classification) or LinearSVR (regression)."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        n_features_to_select = int(self.params.get("n_features_to_select", max(10, int(0.2 * len(feature_names)))))
        step = float(self.params.get("step", 0.1))
        if task == "classification":
            C = float(self.params.get("C", 1.0))
            estimator = LinearSVC(C=C, dual=False, max_iter=int(self.params.get("max_iter", 5000)))
        else:
            C = float(self.params.get("C", 1.0))
            estimator = LinearSVR(C=C, max_iter=int(self.params.get("max_iter", 5000)), random_state=random_state)
        selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
        selector.fit(X, y)
        ranks = selector.ranking_.astype(float)
        scores = 1.0 / ranks
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    selected_mask=(selector.support_))
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RFELinearSVM)


class RFECVLinearSVM(FeatureSelectorMethod):
    meta = MethodMeta(
        name="rfecv_linear_svm",
        family="wrapper",
        tasks={"classification","regression"},
        compute="expensive",
        description="RFECV with LinearSVC (classification) or LinearSVR (regression)."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        step = float(self.params.get("step", 0.1))
        min_features_to_select = int(self.params.get("min_features_to_select", max(5, int(0.05 * len(feature_names)))))
        n_jobs = int(self.params.get("n_jobs", -1))
        if task == "classification":
            scoring = self.params.get("scoring", "roc_auc")
            estimator = LinearSVC(C=float(self.params.get("C", 1.0)), dual=False, max_iter=int(self.params.get("max_iter", 5000)))
            if cv is None:
                cv = StratifiedKFold(n_splits=int(self.params.get("n_splits", 5)), shuffle=True, random_state=random_state)
        else:
            scoring = self.params.get("scoring", "r2")
            estimator = LinearSVR(C=float(self.params.get("C", 1.0)), max_iter=int(self.params.get("max_iter", 5000)), random_state=random_state)
            if cv is None:
                cv = KFold(n_splits=int(self.params.get("n_splits", 5)), shuffle=True, random_state=random_state)
        selector = RFECV(
            estimator=estimator,
            step=step,
            cv=cv,
            scoring=scoring,
            min_features_to_select=min_features_to_select,
            n_jobs=n_jobs,
        )
        selector.fit(X, y)
        ranks = selector.ranking_.astype(float)
        scores = 1.0 / ranks
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family,
                                    selected_mask=(selector.support_))
        return self

    def score_features(self):
        return self._df

REGISTRY.register(RFECVLinearSVM)


class PermutationImportanceRidgeRegressor(FeatureSelectorMethod):
    meta = MethodMeta(
        name="perm_importance_ridge_reg",
        family="wrapper",
        tasks={"regression"},
        compute="expensive",
        description="Permutation importance using a Ridge regressor (single fit; optionally CV externally)."
    )

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        alpha = float(self.params.get("alpha", 1.0))
        n_repeats = int(self.params.get("n_repeats", 5))
        n_jobs = int(self.params.get("n_jobs", -1))
        scoring = self.params.get("scoring", "r2")
        model = Ridge(alpha=alpha, random_state=random_state)
        model.fit(X, y)
        res = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs, scoring=scoring)
        scores = np.maximum(res.importances_mean, 0.0)
        self._df = make_score_frame(feature_names, scores, method_name=self.meta.name, method_family=self.meta.family)
        return self

    def score_features(self):
        return self._df

REGISTRY.register(PermutationImportanceRidgeRegressor)
