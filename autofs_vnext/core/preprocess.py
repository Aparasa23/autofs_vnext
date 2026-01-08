from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from autofs_vnext.core.schemas import PreprocessSpec

class Preprocessor:
    def __init__(self, spec: PreprocessSpec):
        self.spec = spec
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_names_out_: Optional[List[str]] = None
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []

    def fit(self, X: pd.DataFrame):
        self.numeric_cols_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]

        num_steps = [
            ("imputer", SimpleImputer(strategy=self.spec.numeric_impute)),
        ]
        if self.spec.scale_numeric:
            num_steps.append(("scaler", StandardScaler(with_mean=False)))
        num_pipe = Pipeline(steps=num_steps)

        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=self.spec.categorical_impute)),
            ("ohe", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=self.spec.sparse_ohe,
                max_categories=self.spec.max_ohe_levels,
            )),
        ])

        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.numeric_cols_),
                ("cat", cat_pipe, self.categorical_cols_),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )
        self.pipeline.fit(X)

        self.feature_names_out_ = self._get_feature_names()
        return self

    def transform(self, X: pd.DataFrame):
        if self.pipeline is None:
            raise RuntimeError("Preprocessor not fit.")
        Xt = self.pipeline.transform(X)
        return Xt

    def fit_transform(self, X: pd.DataFrame):
        self.fit(X)
        return self.transform(X)

    def _get_feature_names(self) -> List[str]:
        assert self.pipeline is not None
        names: List[str] = []
        # numeric
        names.extend(self.numeric_cols_)
        # categorical - expand
        if self.categorical_cols_:
            ohe: OneHotEncoder = self.pipeline.named_transformers_["cat"].named_steps["ohe"]
            cat_names = list(ohe.get_feature_names_out(self.categorical_cols_))
            names.extend(cat_names)
        return names

    def manifest(self) -> dict:
        return {
            "spec": asdict(self.spec),
            "numeric_cols": self.numeric_cols_,
            "categorical_cols": self.categorical_cols_,
            "feature_names_out": self.feature_names_out_ or [],
        }
