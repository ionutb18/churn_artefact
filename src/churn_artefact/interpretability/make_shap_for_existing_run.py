from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import shap

from churn_artefact.data.telco import load_telco_csv, split_features_target
from churn_artefact.models.pipeline import build_preprocess
from xgboost import XGBClassifier


def project_root_from_run_dir(run_dir: Path) -> Path:
    """
    run_dir = <root>/outputs/<run_name>
    => root = run_dir.parent.parent
    """
    run_dir = run_dir.resolve()
    if run_dir.parent.name != "outputs":
        # if user passes <root>/outputs/<run>, this still works,
        # but we add a clear error for unexpected layouts
        raise ValueError(
            f"run_dir must look like <project_root>/outputs/<run_name>. Got: {run_dir}"
        )
    return run_dir.parent.parent


def load_config_json(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_best_row(results_summary_csv: Path, preferred_model: str = "xgb") -> pd.Series:
    df = pd.read_csv(results_summary_csv)
    if "model" in df.columns and (df["model"] == preferred_model).any():
        # choose best row within preferred model by PR-AUC if present, else first
        sub = df[df["model"] == preferred_model].copy()
        if "pr_auc" in sub.columns:
            sub = sub.sort_values("pr_auc", ascending=False)
        return sub.iloc[0]
    # fallback: global best by pr_auc if present
    if "pr_auc" in df.columns:
        df = df.sort_values("pr_auc", ascending=False)
    return df.iloc[0]


def parse_best_params(val) -> dict:
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        # stored like "{'clf__max_depth': 2, ...}"
        return ast.literal_eval(val)
    raise ValueError(f"Unsupported best_params type: {type(val)}")


def leakage_aware_split(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    train_size: float,
    val_size: float,
    test_size: float,
):
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=(1 - train_size),
        stratify=y,
        random_state=seed,
    )
    rel_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=rel_test,
        stratify=y_tmp,
        random_state=seed,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_shap_beeswarm(
    run_dir: Path,
    seed: int,
    sample_size: int,
    preferred_model: str = "xgb",
    max_display: int = 20,
    copy_to_report_assets: bool = True,
) -> Path:
    run_dir = run_dir.resolve()

    cfg_path = run_dir / "config_used.json"
    results_path = run_dir / "results_summary.csv"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config_used.json: {cfg_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results_summary.csv: {results_path}")

    cfg = load_config_json(cfg_path)
    root = project_root_from_run_dir(run_dir)

    # Load dataset used in THIS run
    data_rel = Path(cfg["dataset"]["path"])
    data_path = (root / data_rel).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = load_telco_csv(str(data_path))
    X, y = split_features_target(df)

    # CRITICAL FIX:
    # Ensure any non-numeric columns are treated as categorical/object,
    # so they DON'T go through the numeric median imputer.
    for c in X.columns:
        if not is_numeric_dtype(X[c]):
            X[c] = X[c].astype("object")

    s = cfg["split"]
    X_train, X_val, X_test, y_train, y_val, y_test = leakage_aware_split(
        X=X,
        y=y,
        seed=seed,
        train_size=float(s["train_size"]),
        val_size=float(s["val_size"]),
        test_size=float(s["test_size"]),
    )

    best_row = pick_best_row(results_path, preferred_model=preferred_model)
    best_params = parse_best_params(best_row["best_params"])

    # Build pipeline (same structure as training)
    pre = build_preprocess(X_train)

    clf = XGBClassifier(
        random_state=seed,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
    )

    model = Pipeline([("preprocess", pre), ("clf", clf)])
    model.set_params(**best_params)
    model.fit(X_train, y_train)

    # Sample from TEST set
    rng = np.random.default_rng(seed)
    n = min(sample_size, len(X_test))
    idx = rng.choice(len(X_test), size=n, replace=False)
    X_sample = X_test.iloc[idx]

    X_sample_t = model.named_steps["preprocess"].transform(X_sample)

    try:
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_sample_t.shape[1])]

    # TreeExplainer expects dense in many setups
    if hasattr(X_sample_t, "toarray"):
        X_dense = X_sample_t.toarray()
    else:
        X_dense = np.asarray(X_sample_t)

    explainer = shap.TreeExplainer(model.named_steps["clf"])
    shap_values = explainer(X_dense)

    # Attach feature names if possible (helps beeswarm labels)
    try:
        shap_values.feature_names = feature_names
    except Exception:
        pass

    out_dir = run_dir / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / "shap_beeswarm.png"

    plt.figure(figsize=(11, 6.5), dpi=250)
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title("SHAP Beeswarm (Top-20 Features) – XGBoost (Test Sample)")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    if copy_to_report_assets:
        ra_dir = root / "report_assets" / "figures"
        ra_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_png, ra_dir / "Fig_5_2_SHAP_Beeswarm_Top20.png")

    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g. outputs/telco_20260303_143755")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_size", type=int, default=500)
    ap.add_argument("--preferred_model", default="xgb")
    args = ap.parse_args()

    out = make_shap_beeswarm(
        run_dir=Path(args.run_dir),
        seed=args.seed,
        sample_size=args.sample_size,
        preferred_model=args.preferred_model,
        max_display=20,
        copy_to_report_assets=True,
    )

    print("[DONE] SHAP beeswarm saved to:", out)
    print("[DONE] Copied to: report_assets/figures/Fig_5_2_SHAP_Beeswarm_Top20.png")


if __name__ == "__main__":
    main()