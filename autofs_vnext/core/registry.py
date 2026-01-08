from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type

import importlib

@dataclass
class MethodMeta:
    name: str
    family: str  # filter / wrapper / embedded / stability / redundancy
    tasks: Set[str]  # {"classification","regression"}
    compute: str  # cheap / medium / expensive
    requires: List[str] = field(default_factory=list)
    description: str = ""
    param_schema: Dict[str, Any] = field(default_factory=dict)

class FeatureSelectorMethod:
    """Contract for all methods."""
    meta: MethodMeta

    def __init__(self, **params: Any):
        self.params = params

    def fit(self, X, y, *, feature_names: List[str], task: str, cv, groups=None, random_state: int = 42):
        raise NotImplementedError

    def score_features(self):
        """Return pandas.DataFrame with columns:
        feature, score, rank, selected_flag, method_name, method_family, extra_json
        """
        raise NotImplementedError

class MethodRegistry:
    def __init__(self):
        self._methods: Dict[str, Type[FeatureSelectorMethod]] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, cls: Type[FeatureSelectorMethod]):
        name = cls.meta.name
        self._methods[name] = cls

    def alias(self, legacy: str, modern: str):
        self._aliases[legacy] = modern

    def resolve(self, name: str) -> str:
        return self._aliases.get(name, name)

    def get(self, name: str) -> Type[FeatureSelectorMethod]:
        key = self.resolve(name)
        if key not in self._methods:
            raise KeyError(f"Unknown method: {name} (resolved to {key}). Available: {sorted(self._methods.keys())}")
        return self._methods[key]

    def list(self) -> List[MethodMeta]:
        return [cls.meta for cls in self._methods.values()]

REGISTRY = MethodRegistry()

def autodiscover():
    # Import built-in methods so they register themselves
    for mod in [
        "autofs_vnext.methods.filters",
        "autofs_vnext.methods.embedded",
        "autofs_vnext.methods.wrappers",
        "autofs_vnext.methods.stability",
        "autofs_vnext.methods.redundancy",
    ]:
        importlib.import_module(mod)

    # Legacy aliases from your earlier config style (examples in doc)
    # Phase_1 methods listed like "11_UDF_SelectKBest_f_classif" etc. 
    REGISTRY.alias("11_UDF_SelectKBest_f_classif", "univariate_f_classif")
    REGISTRY.alias("11_UDF_SelectKBest_f_regression", "univariate_f_regression")
    REGISTRY.alias("11_UDF_SelectKBest_mutual_info", "univariate_mutual_info")
    REGISTRY.alias("2_Modeled_RFE_LogisticRegression", "rfe_logistic")
    REGISTRY.alias("40_Modeled_RFE_DecisionTreeClassifier", "rfe_tree_clf")
    REGISTRY.alias("4B_Modeled_RFE_DecisionTreeClassifier", "rfe_tree_clf")
    REGISTRY.alias("29_Modeled_RFE_AdaBoostRegressor", "rfe_adaboost_reg")
    REGISTRY.alias("8_PermutationImportance_LogisticRegression", "perm_importance_logistic")
    REGISTRY.alias("9_hybrid_RidgeClassifier", "ridge_importance")
    REGISTRY.alias("21 Modeled_genetic_selection", "genetic_placeholder")
    REGISTRY.alias("18_UDF_VIF", "vif_filter")
    REGISTRY.alias("15_UDF_Boruta_RF", "boruta_rf")
    REGISTRY.alias("16_UDF_MIC", "mic_placeholder")
    REGISTRY.alias("13_UDF_RRF_varimp", "rrf_placeholder")
    REGISTRY.alias("1_UDF_Chi_square", "univariate_chi2")
    REGISTRY.alias("0_UDF_cor_selector", "correlation_redundancy_filter")
    REGISTRY.alias("11_UDF_SelectKBest", "univariate_selectkbest_generic")
    REGISTRY.alias("11_UDF_SelectKBest_mutual_info_regression", "univariate_mutual_info")
    REGISTRY.alias("11_UDF_SelectKBest_SelectPercentile", "univariate_selectpercentile")
    REGISTRY.alias("11_UDF_SelectKBest_SelectFpr", "univariate_selectfpr")
    REGISTRY.alias("11_UDF_SelectKBest_SelectFdr", "univariate_selectfdr")
    REGISTRY.alias("11_UDF_SelectKBest_SelectFwe", "univariate_selectfwe")
    REGISTRY.alias("23_UDF_Bayesian_Ridge_Regression", "bayesian_ridge_importance")
    REGISTRY.alias("109_hybrid_BayesianRidgeRegressor", "bayesian_ridge_importance")
    REGISTRY.alias("14_UDF_RF_FAgg", "rf_fold_aggregated_importance")
    REGISTRY.alias("13_UDF_RRF_varimp", "rrf_varimp")
    REGISTRY.alias("2_Modeled_RFE_Lasso", "rfe_lasso")
    REGISTRY.alias("29_Modeled_RFE_Lasso", "rfe_lasso")
    REGISTRY.alias("40_Modeled_RFE_DecisionTreeRegressor", "rfe_decision_tree_reg")
    REGISTRY.alias("41_Modeled_RFE_ExtraTreesRegressor", "rfe_extra_trees_reg")
    REGISTRY.alias("39_Modeled_RFE_RandomForestRegressor", "rfe_random_forest_reg")
    REGISTRY.alias("40_Modeled_RFECV_RandomForestRegressor", "rfecv_random_forest_reg")
    REGISTRY.alias("32_Modeled_RFECV_Lasso", "rfecv_lasso")
    REGISTRY.alias("33_Modeled_RFECV_Logistic", "rfecv_logistic")
    REGISTRY.alias("36_Modeled_RFECV_AdaBoostRegressor", "rfecv_adaboost_reg")
    REGISTRY.alias("30_Modeled_RFE_linear_SVM", "rfe_linear_svm")
    REGISTRY.alias("37_Modeled_RFECV_linear_SVM", "rfecv_linear_svm")
    REGISTRY.alias("8_PermutationImportance_Ridge_Regressor", "perm_importance_ridge_reg")