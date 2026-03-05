from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score

@dataclass
class ThresholdPolicyResult:
    threshold: float
    policy: str
    details: Dict[str, Any]

def calibrate_prefit(estimator, X_val: pd.DataFrame, y_val: pd.Series, method: str = "sigmoid"):
    """Calibrate a fitted estimator using a separate validation set."""
    calib = CalibratedClassifierCV(estimator, method=method, cv="prefit")
    calib.fit(X_val, y_val)
    return calib

def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    policy: str,
    top_k_fraction: float = 0.10,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0
) -> ThresholdPolicyResult:
    policy = policy.lower()

    if policy == "max_f1_on_val":
        thresholds = np.linspace(0.01, 0.99, 99)
        best_t, best_f1 = 0.5, -1.0
        for t in thresholds:
            f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        return ThresholdPolicyResult(best_t, policy, {"best_f1": float(best_f1)})

    if policy == "top_k":
        k = max(1, int(len(y_prob) * float(top_k_fraction)))
        t = float(np.sort(y_prob)[-k])
        return ThresholdPolicyResult(t, policy, {"k": int(k), "top_k_fraction": float(top_k_fraction)})

    if policy == "cost_based":
        thresholds = np.linspace(0.01, 0.99, 99)
        best_t, best_cost = 0.5, float("inf")
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            cost = fn * float(cost_fn) + fp * float(cost_fp)
            if cost < best_cost:
                best_cost = cost
                best_t = float(t)
        return ThresholdPolicyResult(best_t, policy, {"expected_cost": float(best_cost), "cost_fn": float(cost_fn), "cost_fp": float(cost_fp)})

    raise ValueError(f"Unknown threshold policy: {policy}")
