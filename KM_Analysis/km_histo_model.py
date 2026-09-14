"""
KM_Analysis/km_histo_model.py

Kaplan-Meier survival curves from the histology model out-of-fold predictions.
Run main_histo.py first — it saves histo_oof_predictions.csv.

Run from project root:
    python KM_Analysis/km_histo_model.py
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _venv in (".pathomic", "pathomic", "venv", ".venv"):
    _sp = os.path.join(_ROOT, _venv, "lib")
    if os.path.isdir(_sp):
        import glob as _g
        for _d in _g.glob(os.path.join(_sp, "python*", "site-packages")):
            if _d not in sys.path:
                sys.path.insert(0, _d)
        break

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

ROOT    = _ROOT
OOF_CSV = os.path.join(ROOT, "histo_oof_predictions.csv")
CV_CSV  = os.path.join(ROOT, "histo_cv_results.csv")
OUT_DIR = os.path.join(ROOT, "KM_Analysis/outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def km_subplot(ax, t_high, e_high, t_low, e_low,
               label_high, label_low, title,
               col_high="#C0392B", col_low="#2980B9"):
    kmf_h = KaplanMeierFitter()
    kmf_l = KaplanMeierFitter()
    kmf_h.fit(t_high, e_high, label=f"{label_high}  (n={len(t_high)})")
    kmf_l.fit(t_low,  e_low,  label=f"{label_low}  (n={len(t_low)})")
    kmf_h.plot_survival_function(ax=ax, ci_show=True, color=col_high, linewidth=2.5)
    kmf_l.plot_survival_function(ax=ax, ci_show=True, color=col_low,  linewidth=2.5,
                                 linestyle="--")
    lr  = logrank_test(t_high, t_low, event_observed_A=e_high, event_observed_B=e_low)
    p   = lr.p_value
    p_s = f"p = {p:.4f}" if p >= 0.0001 else "p < 0.0001"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Time (days)", fontsize=10)
    ax.set_ylabel("Survival probability", fontsize=10)
    ax.set_ylim(-0.02, 1.05); ax.set_xlim(left=0)
    ax.text(0.97, 0.97, p_s, transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="lightyellow", edgecolor="#aaa"))
    ax.legend(fontsize=9, loc="lower left", framealpha=0.8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return p


def cindex_bar_subplot(ax, cv_df):
    folds  = cv_df["fold"].values
    cindex = cv_df["c_index"].values
    mean_c = cindex.mean()
    std_c  = cindex.std()
    colors = ["#C0392B" if c >= mean_c else "#2980B9" for c in cindex]
    bars   = ax.bar([f"Fold {i+1}" for i in folds], cindex,
                    color=colors, edgecolor="white", linewidth=0.8, width=0.55)
    ax.axhline(mean_c, color="#E67E22", linewidth=2, linestyle="--",
               label=f"Mean = {mean_c:.3f} ± {std_c:.3f}")
    ax.axhline(0.5, color="grey", linewidth=1, linestyle=":", alpha=0.7,
               label="Random baseline (0.5)")
    for bar, val in zip(bars, cindex):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title("C-index per Fold — Histology Model", fontsize=12,
                 fontweight="bold", pad=8)
    ax.set_ylabel("C-index", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    print("=" * 60)
    print("KM Analysis — Histology Model (Out-of-Fold Predictions)")
    print("=" * 60)

    assert os.path.exists(OOF_CSV), (
        f"\nERROR: {OOF_CSV} not found.\n"
        "Run  python histo_pipeline/main_histo.py  first."
    )
    oof = pd.read_csv(OOF_CSV)
    print(f"Loaded {len(oof)} OOF predictions across {oof['fold'].nunique()} folds")
    print(f"  Events  : {int(oof['event'].sum())}")
    print(f"  Censored: {int((oof['event']==0).sum())}")

    median_risk        = oof["risk_score"].median()
    oof["risk_group"]  = (oof["risk_score"] >= median_risk).astype(int)
    overall_cindex     = concordance_index(
        oof["surv_time"].values, -oof["risk_score"].values, oof["event"].values
    )
    print(f"Overall OOF C-index : {overall_cindex:.4f}")
    print(f"Risk threshold (median): {median_risk:.4f}")

    hi  = oof[oof["risk_group"] == 1]
    lo  = oof[oof["risk_group"] == 0]
    q33 = oof["risk_score"].quantile(0.33)
    q67 = oof["risk_score"].quantile(0.67)
    top    = oof[oof["risk_score"] >= q67]
    bottom = oof[oof["risk_score"] <= q33]

    cv_df = None
    if os.path.exists(CV_CSV):
        cv_df = pd.read_csv(CV_CSV)
        cv_df = cv_df[cv_df["config"] == "histo_only"].copy()

    n_cols = 3 if cv_df is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    fig.suptitle(
        f"Histology MIL Model (UNI) — Survival Analysis\n"
        f"OOF C-index = {overall_cindex:.3f}  |  "
        f"{len(oof)} patients  |  {int(oof['event'].sum())} events",
        fontsize=13, fontweight="bold", y=1.02
    )

    p1 = km_subplot(
        axes[0],
        hi["surv_time"].values, hi["event"].values,
        lo["surv_time"].values, lo["event"].values,
        "High Risk (≥ median)", "Low Risk (< median)",
        "KM by Histo Risk Score\n(median split)",
        col_high="#C0392B", col_low="#27AE60",
    )

    p2 = float("nan")
    if len(top) >= 3 and len(bottom) >= 3:
        p2 = km_subplot(
            axes[1],
            top["surv_time"].values,    top["event"].values,
            bottom["surv_time"].values, bottom["event"].values,
            f"Top tertile (n={len(top)})",
            f"Bottom tertile (n={len(bottom)})",
            "KM by Risk Tertile\n(top vs bottom 33%)",
            col_high="#8E44AD", col_low="#2980B9",
        )
    else:
        axes[1].set_visible(False)

    if cv_df is not None and n_cols == 3:
        cindex_bar_subplot(axes[2], cv_df)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "km_histo_model.png")
    out_pdf = os.path.join(OUT_DIR, "km_histo_model.pdf")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf,          bbox_inches="tight", facecolor="white")
    print(f"\nSaved PNG : {out_png}")
    print(f"Saved PDF : {out_pdf}")

    oof.to_csv(os.path.join(OUT_DIR, "histo_oof_with_groups.csv"), index=False)
    print(f"Log-rank p (median split) : {p1:.4f}")
    if not np.isnan(p2):
        print(f"Log-rank p (tertile split): {p2:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
