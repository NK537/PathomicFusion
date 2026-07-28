# KM_Analysis/km_cell_model.py
#
# Kaplan-Meier curves for the cell-graph (Spatiopath) pipeline.
# Reads cell_oof_predictions.csv produced by cell_pipeline/main_cell.py.
#
# Usage:
#   python3 KM_Analysis/km_cell_model.py

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OOF_CSV      = "cell_oof_predictions.csv"
CV_CSV       = "cell_cv_results.csv"
OUT_DIR      = "KM_Analysis/outputs/"
OUT_PREFIX   = "km_cell_model"


def load_oof(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["surv_time"] > 0].dropna(subset=["surv_time", "event", "risk_score"])
    df["event"] = df["event"].astype(int)
    print(f"OOF rows: {len(df)}  events: {df['event'].sum()}")
    return df


def km_median_split(df, ax):
    median_risk = df["risk_score"].median()
    high = df[df["risk_score"] >= median_risk]
    low  = df[df["risk_score"] <  median_risk]

    kmf = KaplanMeierFitter()
    kmf.fit(high["surv_time"], high["event"], label=f"High risk (n={len(high)})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="red")

    kmf.fit(low["surv_time"], low["event"], label=f"Low risk (n={len(low)})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="blue")

    result = logrank_test(
        high["surv_time"], low["surv_time"],
        high["event"],     low["event"],
    )
    ax.set_title(f"Cell-Graph Model (Median Split)\np={result.p_value:.4f}")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.legend()
    return result.p_value


def km_tertile_split(df, ax):
    t33 = df["risk_score"].quantile(0.33)
    t67 = df["risk_score"].quantile(0.67)
    high = df[df["risk_score"] >= t67]
    low  = df[df["risk_score"] <= t33]

    kmf = KaplanMeierFitter()
    kmf.fit(high["surv_time"], high["event"], label=f"Top 33% risk (n={len(high)})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="red")

    kmf.fit(low["surv_time"], low["event"], label=f"Bottom 33% risk (n={len(low)})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="blue")

    result = logrank_test(
        high["surv_time"], low["surv_time"],
        high["event"],     low["event"],
    )
    ax.set_title(f"Cell-Graph Model (Tertile Split)\np={result.p_value:.4f}")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.legend()
    return result.p_value


def plot_cindex_bars(cv_csv, ax):
    if not os.path.exists(cv_csv):
        ax.text(0.5, 0.5, "cell_cv_results.csv not found",
                ha="center", va="center", transform=ax.transAxes)
        return
    df = pd.read_csv(cv_csv)
    df = df[df["config"] == "cell_gnn"]
    folds   = df["fold"].tolist()
    cindex  = df["c_index"].tolist()
    mean_ci = np.mean(cindex)
    ax.bar(folds, cindex, color="steelblue", alpha=0.8)
    ax.axhline(mean_ci, color="red", linestyle="--",
               label=f"Mean = {mean_ci:.3f}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fold")
    ax.set_ylabel("C-index")
    ax.set_title("Per-fold C-index (Cell GNN)")
    ax.legend()


if __name__ == "__main__":
    if not os.path.exists(OOF_CSV):
        print(f"ERROR: {OOF_CSV} not found.")
        print("Run cell_pipeline/main_cell.py first.")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_oof(OOF_CSV)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Cell-Graph Model (Spatiopath) — Kaplan-Meier Analysis", fontsize=14)

    p_median  = km_median_split(df, axes[0])
    p_tertile = km_tertile_split(df, axes[1])
    plot_cindex_bars(CV_CSV, axes[2])

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        out = os.path.join(OUT_DIR, f"{OUT_PREFIX}.{ext}")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close()
    print(f"\nMedian split p={p_median:.4f}   Tertile split p={p_tertile:.4f}")
