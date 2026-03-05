from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def resolve_file(run_dir: Path, filename: str) -> Path:
    """
    Finds a file by trying common locations, then searching recursively under run_dir.
    Tries:
      1) run_dir / "reports" / filename
      2) run_dir / filename
      3) first match of filename anywhere under run_dir (recursive)
    """
    candidates = [
        run_dir / "reports" / filename,
        run_dir / filename,
    ]
    for c in candidates:
        if c.exists():
            return c

    hits = list(run_dir.rglob(filename))
    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"Could not locate '{filename}' under '{run_dir}'.\n"
        f"Tried: {candidates[0]} and {candidates[1]} and recursive search."
    )


def read_scores_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if not {"y_true", "y_prob"}.issubset(df.columns):
        raise ValueError(
            f"{csv_path} must contain columns: y_true, y_prob\n"
            f"Found columns: {list(df.columns)}"
        )
    return df["y_true"].to_numpy().astype(int), df["y_prob"].to_numpy().astype(float)


def get_threshold_from_results(results_summary_csv: Path, preferred_model: str = "xgb") -> float:
    df = pd.read_csv(results_summary_csv)
    if "threshold" not in df.columns:
        raise ValueError(
            f"{results_summary_csv} must contain a 'threshold' column.\n"
            f"Found columns: {list(df.columns)}"
        )

    if "model" in df.columns and (df["model"] == preferred_model).any():
        return float(df.loc[df["model"] == preferred_model].iloc[0]["threshold"])
    return float(df.iloc[0]["threshold"])


# -----------------------------
# FIG 3.1 - Method Pipeline
# -----------------------------
def _add_box(ax, x, y, w, h, text, fontsize=10):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        facecolor="white",
        edgecolor="black"
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)
    return (x, y, w, h)


def _add_arrow(ax, x1, y1, x2, y2):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.1,
        color="black"
    )
    ax.add_patch(arr)


def make_fig_3_1_pipeline(out_png: Path) -> None:
    fig = plt.figure(figsize=(14, 6), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.94,
        "Churn Prediction – Leakage-aware End-to-End Methodological Pipeline",
        ha="center", va="center",
        fontsize=13, fontweight="bold"
    )

    w, h = 0.21, 0.11
    y1, y2, y3 = 0.72, 0.50, 0.28
    xs = [0.05, 0.29, 0.53, 0.77]

    b1 = _add_box(ax, xs[0], y1, w, h, "Data sources\nTelco + Synthetic SaaS", fontsize=10)
    b2 = _add_box(ax, xs[1], y1, w, h, "Leakage-aware split\nTrain/Val/Test", fontsize=10)
    b3 = _add_box(ax, xs[2], y1, w, h, "Preprocessing\nImpute • Encode • Scale", fontsize=10)
    b4 = _add_box(ax, xs[3], y1, w, h, "Training & tuning\nLR • RF • XGB\n(CV search)", fontsize=10)

    for a, b in [(b1, b2), (b2, b3), (b3, b4)]:
        _add_arrow(ax, a[0] + a[2], a[1] + a[3]/2, b[0], b[1] + b[3]/2)

    b5 = _add_box(ax, 0.16, y2, 0.24, h, "Probability calibration\nPlatt scaling", fontsize=10)
    b6 = _add_box(ax, 0.42, y2, 0.24, h, "Threshold policy\nSelect threshold on validation\n(e.g., maximise F1)", fontsize=10)
    b7 = _add_box(ax, 0.68, y2, 0.24, h, "Final test evaluation\nPR-AUC • ROC-AUC • F1\nBrier • LogLoss", fontsize=10)

    _add_arrow(ax, b4[0] + b4[2]/2, b4[1], b5[0] + b5[2]/2, b5[1] + b5[3])
    _add_arrow(ax, b5[0] + b5[2], b5[1] + b5[3]/2, b6[0], b6[1] + b6[3]/2)
    _add_arrow(ax, b6[0] + b6[2], b6[1] + b6[3]/2, b7[0], b7[1] + b7[3]/2)

    b8 = _add_box(ax, 0.18, y3, 0.28, h, "Interpretability\nPermutation importance\n(+ SHAP optional)", fontsize=10)
    b9 = _add_box(ax, 0.54, y3, 0.28, h, "Business translation\nRetention actions\n& prioritisation", fontsize=10)

    _add_arrow(ax, b7[0] + b7[2]/2, b7[1], b8[0] + b8[2]/2, b8[1] + b8[3])
    _add_arrow(ax, b8[0] + b8[2], b8[1] + b8[3]/2, b9[0], b9[1] + b9[3]/2)

    ax.text(
        0.5, 0.12,
        "Artefact outputs: results_summary.csv • test_scores_*.csv • calibration_*.png • permutation_importance_top20.png",
        ha="center", va="center", fontsize=9
    )

    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


# -----------------------------
# FIG 4.1 - Calibration panel (3 images -> 1)
# -----------------------------
def make_calibration_panel(run_dir: Path, out_png: Path) -> None:
    p_logreg = resolve_file(run_dir, "calibration_logreg.png")
    p_rf = resolve_file(run_dir, "calibration_rf.png")
    p_xgb = resolve_file(run_dir, "calibration_xgb.png")

    print("[FOUND] calibration_logreg:", p_logreg)
    print("[FOUND] calibration_rf    :", p_rf)
    print("[FOUND] calibration_xgb   :", p_xgb)

    imgs = [plt.imread(p_logreg), plt.imread(p_rf), plt.imread(p_xgb)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=300)
    titles = ["Logistic Regression", "Random Forest", "XGBoost"]

    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=10)
        ax.axis("off")

    fig.suptitle("Calibration Curves (Platt Scaling)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# -----------------------------
# FIG 4.2 & 4.3 - ROC and PR curves
# -----------------------------
def make_roc_pr_curves(scores_csv: Path, out_roc_png: Path, out_pr_png: Path) -> None:
    y_true, y_prob = read_scores_csv(scores_csv)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6.5, 5), dpi=300)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.tight_layout()
    plt.savefig(out_roc_png, bbox_inches="tight", pad_inches=0.15)
    plt.close()

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6.5, 5), dpi=300)
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Test Set)")
    plt.tight_layout()
    plt.savefig(out_pr_png, bbox_inches="tight", pad_inches=0.15)
    plt.close()


# -----------------------------
# FIG 4.4 - Confusion Matrix heatmap
# -----------------------------
def make_confusion_matrix(scores_csv: Path, threshold: float, out_png: Path) -> None:
    y_true, y_prob = read_scores_csv(scores_csv)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5.5, 5), dpi=300)
    plt.imshow(cm)
    plt.title(f"Confusion Matrix (threshold={threshold:.2f})")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.15)
    plt.close()


# -----------------------------
# FIG 5.1 - Copy permutation importance plot (rename)
# -----------------------------
def export_permutation_plot(run_dir: Path, out_png: Path) -> None:
    perm_png = resolve_file(run_dir, "permutation_importance_top20.png")
    print("[FOUND] permutation_importance_top20:", perm_png)
    shutil.copyfile(perm_png, out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g. outputs/telco_20260303_143755")
    ap.add_argument("--out_dir", default="report_assets/figures", help="Word-ready figures output folder")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Figure 3.1
    make_fig_3_1_pipeline(out_dir / "Fig_3_1_Method_Pipeline.png")

    # Figure 4.1 - calibration panel
    make_calibration_panel(run_dir, out_dir / "Fig_4_1_Calibration_Panel.png")

    # ROC/PR based on XGB scores
    scores_xgb = resolve_file(run_dir, "test_scores_xgb.csv")
    print("[FOUND] test_scores_xgb:", scores_xgb)

    make_roc_pr_curves(
        scores_csv=scores_xgb,
        out_roc_png=out_dir / "Fig_4_2_ROC_XGB.png",
        out_pr_png=out_dir / "Fig_4_3_PR_XGB.png",
    )

    # Confusion matrix threshold from results_summary
    results_summary = resolve_file(run_dir, "results_summary.csv")
    print("[FOUND] results_summary:", results_summary)

    thr = get_threshold_from_results(results_summary, preferred_model="xgb")
    make_confusion_matrix(
        scores_csv=scores_xgb,
        threshold=thr,
        out_png=out_dir / "Fig_4_4_Confusion_Matrix_XGB.png",
    )

    # Permutation importance
    export_permutation_plot(run_dir, out_dir / "Fig_5_1_Permutation_Importance_Top20.png")

    print("\nDone. Saved Word-ready figures to:", out_dir.resolve())
    for p in sorted(out_dir.glob("Fig_*")):
        print(" -", p.name)


if __name__ == "__main__":
    main()