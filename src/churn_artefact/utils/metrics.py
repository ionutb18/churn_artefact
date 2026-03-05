from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, precision_recall_fscore_support, matthews_corrcoef,
    confusion_matrix
)

@dataclass
class MetricPack:
    roc_auc: float
    pr_auc: float
    brier: float
    logloss: float
    precision: float
    recall: float
    f1: float
    mcc: float
    tn: int
    fp: int
    fn: int
    tp: int

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> MetricPack:
    y_pred = (y_prob >= threshold).astype(int)
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, y_prob, labels=[0,1])
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return MetricPack(
        roc_auc=float(roc), pr_auc=float(pr), brier=float(brier), logloss=float(ll),
        precision=float(p), recall=float(r), f1=float(f1), mcc=float(mcc),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)
    )

def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    """Return (mean, low, high) bootstrap percentile CI (2.5%, 97.5%)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(metric_fn(y_true[idx], y_prob[idx]))
    stats = np.array(stats, dtype=float)
    return float(stats.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
