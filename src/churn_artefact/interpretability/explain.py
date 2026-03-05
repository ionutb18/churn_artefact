from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

def run_permutation_importance(model, X: pd.DataFrame, y: pd.Series, out_dir: str | Path, n_repeats: int = 10, seed: int = 42) -> pd.DataFrame:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=seed, n_jobs=-1, scoring="average_precision")
    imp = pd.DataFrame({"feature": X.columns, "importance_mean": r.importances_mean, "importance_std": r.importances_std})
    imp = imp.sort_values("importance_mean", ascending=False)
    imp.to_csv(out_dir / "permutation_importance.csv", index=False)

    top = imp.head(20).iloc[::-1]
    plt.figure()
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Permutation importance (mean decrease in PR-AUC)")
    plt.title("Top-20 permutation importances")
    plt.tight_layout()
    plt.savefig(out_dir / "permutation_importance_top20.png", dpi=200)
    plt.close()
    return imp

def run_shap_if_available(model, X: pd.DataFrame, out_dir: str | Path, sample_size: int = 500, seed: int = 42) -> Optional[str]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import shap
    except Exception:
        return None

    rng = np.random.default_rng(seed)
    Xs = X.sample(n=min(sample_size, len(X)), random_state=seed)

    try:
        explainer = shap.Explainer(model)
        sv = explainer(Xs)
        shap.plots.beeswarm(sv, show=False, max_display=20)
        plt.tight_layout()
        out_path = out_dir / "shap_beeswarm.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        return str(out_path)
    except Exception:
        return None
