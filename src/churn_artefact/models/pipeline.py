from __future__ import annotations
from typing import Any, Dict, Tuple, List
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

def _infer_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols

def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols, cat_cols = _infer_column_types(X)

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols)
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    return pre

def get_model_and_search_space(model_key: str, seed: int) -> Tuple[Any, Dict[str, Any]]:
    if model_key == "logreg":
        model = LogisticRegression(
            max_iter=5000,
            solver="saga",
            n_jobs=-1,
            class_weight="balanced"
        )
        space = {
            "clf__C": np.logspace(-3, 2, 60),
            "clf__penalty": ["l1", "l2"]
        }
        return model, space

    if model_key == "rf":
        model = RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample"
        )
        space = {
            "clf__n_estimators": [200, 400, 800],
            "clf__max_depth": [None, 4, 8, 12, 20],
            "clf__min_samples_split": [2, 5, 10, 20],
            "clf__min_samples_leaf": [1, 2, 4, 8],
            "clf__max_features": ["sqrt", "log2", None],
        }
        return model, space

    if model_key in {"xgb", "xgboost"}:
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                random_state=seed,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist"
            )
            space = {
                "clf__n_estimators": [300, 600, 900, 1200],
                "clf__max_depth": [2, 3, 4, 6, 8],
                "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "clf__subsample": [0.7, 0.85, 1.0],
                "clf__colsample_bytree": [0.7, 0.85, 1.0],
                "clf__min_child_weight": [1, 3, 5, 8],
                "clf__reg_lambda": [0.5, 1.0, 2.0, 5.0],
            }
            return model, space
        except Exception:
            from sklearn.ensemble import HistGradientBoostingClassifier
            model = HistGradientBoostingClassifier(
                random_state=seed
            )
            space = {
                "clf__max_depth": [3, 5, 7, 9],
                "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "clf__max_leaf_nodes": [15, 31, 63, 127],
                "clf__min_samples_leaf": [20, 50, 100]
            }
            return model, space

    raise ValueError(f"Unknown model_key: {model_key}")

def tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_key: str,
    seed: int,
    n_iter: int = 40,
    cv_folds: int = 5
):
    pre = build_preprocess(X)
    clf, space = get_model_and_search_space(model_key, seed)

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", clf)
    ])

    scorer = make_scorer(roc_auc_score, response_method="predict_proba")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=space,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X, y)
    return search