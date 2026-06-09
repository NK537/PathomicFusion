# histo_pipeline/audit_histo_data.py
#
# Run this BEFORE training to verify the histology dataset is correctly wired.
# Mirrors audit_picasso_data.py (endoscopy).
#
# Usage:
#   python histo_pipeline/audit_histo_data.py

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from histo_pipeline.config_histo import HISTO_CONFIG


def try_load(path_no_ext):
    for ext in [".pt", ".npy", ""]:
        p = path_no_ext + ext
        if os.path.exists(p):
            try:
                if ext == ".npy":
                    return torch.from_numpy(np.load(p)).float(), ext or "(no ext)"
                obj = torch.load(p, map_location="cpu", weights_only=False)
                t   = torch.from_numpy(obj).float() if isinstance(obj, np.ndarray) \
                      else obj.float()
                return t, ext or "(no ext)"
            except Exception as e:
                print(f"  [WARN] {p}: {e}")
    return None, None


def sep(title=""):
    print("\n" + "=" * 60)
    if title: print(f"  {title}"); print("=" * 60)


def audit(cfg):

    sep("1. LABEL FILE")
    assert os.path.exists(cfg["label_xlsx"]), f"Not found: {cfg['label_xlsx']}"
    raw  = pd.read_excel(cfg["label_xlsx"])
    cols = list(raw.columns)
    try:
        doi = cols.index("date_of_outcome"); cols[doi + 1] = "days_to_outcome"
    except (ValueError, IndexError):
        for i, c in enumerate(cols):
            if str(c).startswith("Unnamed:"): cols[i] = "days_to_outcome"; break
    raw.columns = cols
    raw["code"]        = raw["code"].astype(str).str.zfill(4)
    raw["ANY OUTCOME"] = pd.to_numeric(raw["ANY OUTCOME"], errors="coerce")
    raw["days_to_outcome"] = pd.to_numeric(raw["days_to_outcome"], errors="coerce")
    print(f"  Rows       : {len(raw)}")
    print(f"  Patients   : {raw['code'].nunique()}")

    sep("2. OUTCOME STATISTICS")
    n_event    = int((raw["ANY OUTCOME"] == 1).sum())
    n_censored = int((raw["ANY OUTCOME"] == 0).sum())
    print(f"  Event (=1) : {n_event}")
    print(f"  Censored   : {n_censored}")
    print(f"  Event rate : {n_event / max(n_event+n_censored,1):.1%}")
    vt = raw.loc[raw["days_to_outcome"] > 0, "days_to_outcome"]
    if len(vt):
        print(f"  Time (days): min={vt.min():.0f}  median={vt.median():.0f}  max={vt.max():.0f}")

    sep("3. PATCH CSV")
    patch_csv = cfg["histo_patches_csv"]
    if not os.path.exists(patch_csv):
        print(f"  NOT FOUND: {patch_csv}")
        print(f"  Create this CSV with columns: patient_id, patch_filename")
    else:
        patch_df = pd.read_csv(patch_csv)
        patch_df["patient_id"] = patch_df["patient_id"].astype(str).str.zfill(4)
        print(f"  Rows (patches)   : {len(patch_df)}")
        print(f"  Unique patients  : {patch_df['patient_id'].nunique()}")
        per_pt = patch_df.groupby("patient_id").size()
        print(f"  Patches/patient  : min={per_pt.min()}  median={per_pt.median():.0f}  max={per_pt.max()}")

    sep("4. EMBEDDING FILE SCAN")
    emb_dir = cfg["histo_emb_dir"]
    if not os.path.exists(emb_dir):
        print(f"  Embedding dir NOT FOUND: {emb_dir}")
        print(f"  Run precompute_histo_embeddings.py first.")
    else:
        patch_csv = cfg["histo_patches_csv"]
        if os.path.exists(patch_csv):
            patch_df = pd.read_csv(patch_csv)
            patch_df["patient_id"] = patch_df["patient_id"].astype(str).str.zfill(4)
            found = missing = 0
            shape_counts = {}
            sample_shapes = []
            for _, row in patch_df.iterrows():
                base = os.path.splitext(str(row["patch_filename"]))[0]
                stem = os.path.join(emb_dir, base)
                t, ext = try_load(stem)
                if t is not None:
                    found += 1
                    k = str(tuple(t.shape))
                    shape_counts[k] = shape_counts.get(k, 0) + 1
                    if len(sample_shapes) < 3:
                        sample_shapes.append((base, t.shape))
                else:
                    missing += 1
            print(f"  Embeddings found  : {found}")
            print(f"  Embeddings missing: {missing}")
            print(f"  Shape counts      : {shape_counts}")
            if sample_shapes:
                print(f"  Sample shapes:")
                for name, sh in sample_shapes:
                    print(f"    {name}: {sh}")

            sep("5. CONFIG CHECK")
            if shape_counts:
                most_common = max(shape_counts, key=shape_counts.get)
                actual_dim  = eval(most_common)[-1]
                cfg_dim     = cfg["histo_dim"]
                ok = actual_dim == cfg_dim
                print(f"  config histo_dim : {cfg_dim}")
                print(f"  actual dim       : {actual_dim}")
                print(f"  Status           : {'OK' if ok else f'MISMATCH — update histo_dim to {actual_dim}'}")

    sep("6. READY-TO-TRAIN CHECK")
    checks = {
        "Label file exists"         : os.path.exists(cfg["label_xlsx"]),
        "Patch CSV exists"          : os.path.exists(cfg["histo_patches_csv"]),
        "Embedding dir exists"      : os.path.exists(cfg["histo_emb_dir"]),
        "Events > 0"                : n_event > 0,
    }
    all_pass = True
    for k, v in checks.items():
        if not v: all_pass = False
        print(f"  [{'PASS' if v else 'FAIL'}]  {k}")
    print()
    if all_pass:
        print("  All checks passed.  Run:  python histo_pipeline/main_histo.py")
    else:
        print("  Fix the FAIL items above before training.")
    sep()


if __name__ == "__main__":
    audit(HISTO_CONFIG)
