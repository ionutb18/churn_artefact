from __future__ import annotations
import argparse
from pathlib import Path
import datetime
import numpy as np
import pandas as pd

import matplotlib
# Spunem Matplotlib sa ruleze "in background" fara sa incerce sa deschida ferestre grafice
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve

from churn_artefact.utils.io import read_json, write_json
from churn_artefact.utils.repro import set_global_seed
from churn_artefact.utils.metrics import compute_metrics, bootstrap_ci

from churn_artefact.data.telco import load_telco_csv, split_features_target
from churn_artefact.data.synthetic import maybe_generate_and_save
from churn_artefact.models.pipeline import tune_model
from churn_artefact.models.calibration import calibrate_prefit, select_threshold
from churn_artefact.interpretability.explain import run_permutation_importance, run_shap_if_available
from churn_artefact.reports.actionability import build_actionability_report


def _make_run_dir(base_out: Path, run_name: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_name}_{ts}"
    out_dir = base_out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _split_train_val_test(X: pd.DataFrame, y: pd.Series, train_size: float, val_size: float, test_size: float, seed: int):
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-9, "Split sizes must sum to 1.0"
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=(1-train_size), stratify=y, random_state=seed)
    rel_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=rel_test, stratify=y_tmp, random_state=seed)
    return X_train, y_train, X_val, y_val, X_test, y_test


def _plot_calibration(y_true, y_prob, out_path: Path, n_bins: int = 10):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve (quantile bins)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_single(cfg: dict, base_out: Path, global_seed: int) -> Path:
    name = cfg["name"]
    out_dir = _make_run_dir(base_out, name)
    write_json(out_dir / "config_used.json", cfg)

    ds = cfg["dataset"]
    kind = ds["kind"]

    if kind == "telco_ibm":
        df = load_telco_csv(ds["path"])
        X, y = split_features_target(df)
    elif kind == "synthetic_saas":
        gen_cfg = ds.get("generator", {})
        maybe_generate_and_save(ds["path"], gen_cfg, seed=global_seed)
        df = pd.read_csv(ds["path"])
        y = df["churn_next_30d"].astype(int)
        X = df.drop(columns=["churn_next_30d"])
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")

    # Splits
    s = cfg["split"]
    X_train, y_train, X_val, y_val, X_test, y_test = _split_train_val_test(
        X, y, s["train_size"], s["val_size"], s["test_size"], seed=global_seed
    )

    tuning = cfg["tuning"]
    results = []
    fitted_models = {}

    for model_key in cfg["models"]:
        search = tune_model(X_train, y_train, model_key, seed=global_seed, n_iter=tuning["n_iter"], cv_folds=tuning["cv_folds"])
        best = search.best_estimator_
        fitted = best.fit(X_train, y_train)
        fitted_models[model_key] = fitted

        y_val_prob = fitted.predict_proba(X_val)[:, 1]

        calib_cfg = cfg.get("calibration", {"enabled": False})
        model_for_eval = fitted
        if calib_cfg.get("enabled", False):
            model_for_eval = calibrate_prefit(fitted, X_val, y_val, method=calib_cfg.get("method", "sigmoid"))
            y_val_prob = model_for_eval.predict_proba(X_val)[:, 1]

        th_cfg = cfg["thresholding"]
        th_res = select_threshold(
            y_val.to_numpy(), y_val_prob,
            policy=th_cfg["policy"],
            top_k_fraction=th_cfg.get("top_k_fraction", 0.1),
            cost_fn=th_cfg.get("cost_fn", 10.0),
            cost_fp=th_cfg.get("cost_fp", 1.0)
        )
        threshold = th_res.threshold

        y_test_prob = model_for_eval.predict_proba(X_test)[:, 1]
        pack = compute_metrics(y_test.to_numpy(), y_test_prob, threshold=threshold)

        roc_mean, roc_lo, roc_hi = bootstrap_ci(y_test.to_numpy(), y_test_prob, roc_auc_score, n_boot=1000, seed=global_seed)
        pr_mean, pr_lo, pr_hi = bootstrap_ci(y_test.to_numpy(), y_test_prob, average_precision_score, n_boot=1000, seed=global_seed)

        results.append({
            "dataset": name,
            "model": model_key,
            "best_params": search.best_params_,
            "threshold_policy": th_res.policy,
            "threshold": threshold,
            "roc_auc": pack.roc_auc,
            "pr_auc": pack.pr_auc,
            "brier": pack.brier,
            "logloss": pack.logloss,
            "precision": pack.precision,
            "recall": pack.recall,
            "f1": pack.f1,
            "mcc": pack.mcc,
            "tn": pack.tn, "fp": pack.fp, "fn": pack.fn, "tp": pack.tp,
            "roc_auc_ci_mean": roc_mean, "roc_auc_ci_low": roc_lo, "roc_auc_ci_high": roc_hi,
            "pr_auc_ci_mean": pr_mean, "pr_auc_ci_low": pr_lo, "pr_auc_ci_high": pr_hi,
        })

        _plot_calibration(y_test.to_numpy(), y_test_prob, out_dir / f"calibration_{model_key}.png")
        pd.DataFrame({"y_true": y_test.to_numpy(), "y_prob": y_test_prob}).to_csv(out_dir / f"test_scores_{model_key}.csv", index=False)

    df_res = pd.DataFrame(results).sort_values(["pr_auc"], ascending=False)
    df_res.to_csv(out_dir / "results_summary.csv", index=False)

    best_model_key = df_res.iloc[0]["model"]
    best_model = fitted_models[best_model_key]

    interp_cfg = cfg.get("interpretability", {})
    if interp_cfg.get("permutation_importance", True):
        run_permutation_importance(best_model, X_test, y_test, out_dir / "interpretability", seed=global_seed)
        build_actionability_report(out_dir / "interpretability" / "permutation_importance.csv", out_dir / "reports", top_n=12)

    if interp_cfg.get("shap", False):
        run_shap_if_available(best_model, X_test, out_dir / "interpretability", sample_size=int(interp_cfg.get("shap_sample_size", 500)), seed=global_seed)

    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/default.json")
    ap.add_argument("--out", default="outputs", help="Output directory")
    args = ap.parse_args()

    cfg = read_json(args.config)
    seed = int(cfg.get("random_seed", 42))
    set_global_seed(seed)

    base_out = Path(args.out)
    base_out.mkdir(parents=True, exist_ok=True)

    out_dirs = []
    for run_cfg in cfg["runs"]:
        out_dirs.append(run_single(run_cfg, base_out, global_seed=seed))

    print("Done. Outputs:")
    for d in out_dirs:
        print(" -", d)


if __name__ == "__main__":
    main()