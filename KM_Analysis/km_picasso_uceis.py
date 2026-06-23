"""
KM_Analysis/km_picasso_uceis.py

Kaplan-Meier survival curves stratified by:
  - PICaSSO score >= 3  (High vs Low)
  - UCEIS  score >  1   (High vs Low)

Run from the project root:
    python KM_Analysis/km_picasso_uceis.py

Required files (relative to project root):
    data/Picasso/PicassoOnly_Outcome_train.xlsx     <- TTE + event labels
    data/Picasso/PicassoOnly_AI_corrected_rev.xlsx  <- section-level scores
"""

import sys, os, re
# Auto-add project venv site-packages so script works with plain python3
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
from lifelines.statistics import logrank_test

# ── File paths (relative to project root) ───────────────────────────────────
ROOT          = _ROOT
OUTCOME_FILE  = os.path.join(ROOT, "data/Picasso/PicassoOnly_Outcome_train.xlsx")
SCORES_FILE   = os.path.join(ROOT, "data/Picasso/PicassoOnly_AI_corrected_rev.xlsx")
# fallback name (corretti vs corrected)
SCORES_ALT    = os.path.join(ROOT, "data/Picasso/histo/PicassoOnly_AI_corretti_rev.xlsx")
OUT_DIR       = os.path.join(ROOT, "KM_Analysis/outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Patient ID helpers ───────────────────────────────────────────────────────
def video_to_id(video_str):
    """'01-01 rectum' -> '01-01'"""
    m = re.match(r"(\d{2}-\d{2,3})", str(video_str).strip())
    return m.group(1) if m else None

def id_to_code(pat_id):
    """'01-01' -> '0101'  (matches `code` column in Outcome file)"""
    return pat_id.replace("-", "")


# ── Load outcome labels ──────────────────────────────────────────────────────
def load_outcome():
    assert os.path.exists(OUTCOME_FILE), f"Not found: {OUTCOME_FILE}"
    raw  = pd.read_excel(OUTCOME_FILE)
    cols = list(raw.columns)

    # Rename unnamed days column (sits right after date_of_outcome)
    for i, c in enumerate(cols):
        if str(c).startswith("Unnamed") or str(c).lower() == "days_to_outcome":
            cols[i] = "days_to_outcome"
    # Also look for it by position after date_of_outcome
    try:
        doi = [c.lower() for c in cols].index("date_of_outcome")
        cols[doi + 1] = "days_to_outcome"
    except (ValueError, IndexError):
        pass
    raw.columns = cols

    raw["code"]            = raw["code"].astype(str).str.zfill(4)
    raw["event"]           = pd.to_numeric(raw["ANY OUTCOME"],    errors="coerce")
    raw["days_to_outcome"] = pd.to_numeric(raw["days_to_outcome"], errors="coerce")

    # Use days_to_outcome if positive, else compute from dates
    def get_tte(row):
        if pd.notna(row["days_to_outcome"]) and row["days_to_outcome"] > 0:
            return row["days_to_outcome"]
        try:
            from datetime import datetime
            fmt = ["%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d"]
            for f in fmt:
                try:
                    d1 = datetime.strptime(str(row.get("date_of_procedure","")).strip(), f)
                    d2 = datetime.strptime(str(row.get("date_of_visit","")).strip(), f)
                    return float(abs((d2 - d1).days))
                except: pass
        except: pass
        return np.nan

    raw["surv_time"] = raw.apply(get_tte, axis=1)
    df = raw[["code","event","surv_time"]].dropna(subset=["event","surv_time"])
    df = df[df["surv_time"] > 0].copy()
    print(f"Outcome file: {len(df)} patients  events={int(df['event'].sum())}")
    return df.set_index("code")


# ── Load section-level scores ─────────────────────────────────────────────────
def load_scores():
    path = SCORES_FILE if os.path.exists(SCORES_FILE) else SCORES_ALT
    assert os.path.exists(path), (
        f"Scores file not found.\nLooked for:\n  {SCORES_FILE}\n  {SCORES_ALT}"
    )
    print(f"Scores file : {os.path.basename(path)}")
    df = pd.read_excel(path)

    df["pat_id"] = df["Video"].apply(video_to_id)
    df = df.dropna(subset=["pat_id"])

    # PICaSSO score column is called "Score" in the actual file
    score_col = "Score" if "Score" in df.columns else "PICaSSO>=3"
    uceis_col = "UCEIS>1"

    df["picasso"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
    df["uceis"]   = pd.to_numeric(df[uceis_col], errors="coerce").fillna(0)

    # Aggregate to patient level: max across sections
    pat = df.groupby("pat_id").agg(
        picasso_max=("picasso", "max"),
        uceis_max  =("uceis",   "max"),
    ).reset_index()

    pat["code"]         = pat["pat_id"].apply(id_to_code)
    pat["picasso_high"] = (pat["picasso_max"] >= 3).astype(int)
    pat["uceis_high"]   = (pat["uceis_max"]   >  1).astype(int)

    print(f"Score file  : {len(pat)} patients")
    print(f"  PICaSSO >= 3: {pat['picasso_high'].sum()} High | {(~pat['picasso_high'].astype(bool)).sum()} Low")
    print(f"  UCEIS >  1 : {pat['uceis_high'].sum()} High | {(~pat['uceis_high'].astype(bool)).sum()} Low")
    return pat.set_index("code")


# ── KM subplot ───────────────────────────────────────────────────────────────
def km_subplot(ax, t_high, e_high, t_low, e_low,
               label_high, label_low, title,
               col_high="#C0392B", col_low="#2980B9"):

    kmf_h = KaplanMeierFitter()
    kmf_l = KaplanMeierFitter()
    kmf_h.fit(t_high, e_high, label=f"{label_high}  (n={len(t_high)})")
    kmf_l.fit(t_low,  e_low,  label=f"{label_low}  (n={len(t_low)})")

    kmf_h.plot_survival_function(ax=ax, ci_show=True,  color=col_high, linewidth=2.5)
    kmf_l.plot_survival_function(ax=ax, ci_show=True,  color=col_low,  linewidth=2.5,
                                 linestyle="--")

    lr  = logrank_test(t_high, t_low, event_observed_A=e_high, event_observed_B=e_low)
    p   = lr.p_value
    p_s = f"p = {p:.4f}" if p >= 0.0001 else "p < 0.0001"

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel("Survival probability", fontsize=11)
    ax.set_ylim(-0.02, 1.05); ax.set_xlim(left=0)
    ax.text(0.97, 0.97, p_s, transform=ax.transAxes, ha="right", va="top",
            fontsize=11, bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="lightyellow", edgecolor="#aaa"))
    ax.legend(fontsize=10, loc="lower left", framealpha=0.8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return p


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 56)
    print("Kaplan-Meier Analysis — Picasso Cohort")
    print("=" * 56)

    outcome = load_outcome()   # indexed by code (e.g. "0101")
    scores  = load_scores()    # indexed by code (e.g. "0101")

    merged = scores.join(outcome, how="inner")
    print(f"\nMatched patients: {len(merged)}")
    print(f"  Events  : {int(merged['event'].sum())}")
    print(f"  Censored: {int((merged['event']==0).sum())}")

    if len(merged) == 0:
        print("\nERROR: No patients matched. Check patient ID format.")
        return

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Kaplan-Meier Survival Curves — Picasso Endoscopy Cohort",
                 fontsize=14, fontweight="bold", y=1.01)

    # PICaSSO >= 3
    hi = merged[merged["picasso_high"] == 1]
    lo = merged[merged["picasso_high"] == 0]
    p1 = km_subplot(axes[0],
                    hi["surv_time"].values, hi["event"].values,
                    lo["surv_time"].values, lo["event"].values,
                    "PICaSSO ≥ 3", "PICaSSO < 3",
                    "PICaSSO Score  ( ≥ 3  vs  < 3 )",
                    col_high="#C0392B", col_low="#27AE60")

    # UCEIS > 1
    hi2 = merged[merged["uceis_high"] == 1]
    lo2 = merged[merged["uceis_high"] == 0]
    p2  = km_subplot(axes[1],
                     hi2["surv_time"].values, hi2["event"].values,
                     lo2["surv_time"].values, lo2["event"].values,
                     "UCEIS > 1", "UCEIS ≤ 1",
                     "UCEIS Score  ( > 1  vs  ≤ 1 )",
                     col_high="#8E44AD", col_low="#2980B9")

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "km_picasso_uceis.png")
    out_pdf = os.path.join(OUT_DIR, "km_picasso_uceis.pdf")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf,          bbox_inches="tight", facecolor="white")
    print(f"\nSaved PNG : {out_png}")
    print(f"Saved PDF : {out_pdf}")

    # Patient-level summary CSV
    csv_path = os.path.join(OUT_DIR, "km_patient_groups.csv")
    merged[["pat_id","picasso_max","uceis_max",
            "picasso_high","uceis_high","event","surv_time"]].to_csv(csv_path)
    print(f"Saved CSV : {csv_path}")

    print(f"\nLog-rank p-value (PICaSSO >= 3): {p1:.4f}")
    print(f"Log-rank p-value (UCEIS >  1)  : {p2:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
