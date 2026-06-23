"""
audit_picasso_data.py
─────────────────────
Run this BEFORE training to verify the Picasso dataset is correctly wired.

What it checks:
  1. Label file is readable and key columns are present
  2. Event / censoring statistics
  3. How many patients have embedding files on disk
  4. Actual shape of the embedding tensors (crucial for endo_dim in config)
  5. Survival time distribution
  6. Patients that will be skipped and why

Usage:
    python audit_picasso_data.py
"""

import os
import numpy as np
import pandas as pd
import torch
from config_picasso import PICASSO_CONFIG


# ── helpers ───────────────────────────────────────────────────────────────────
def try_load(path_no_ext):
    for ext in [".pt", ".npy",".mat", ""]:
        p = path_no_ext + ext
        if os.path.exists(p):
            try:
                if ext == ".npy":
                    arr = np.load(p)
                    return torch.from_numpy(arr).float(), ext or "(no ext)"
                if ext == ".mat":
                    import scipy.io as sio
                    mat = sio.loadmat(p)
                    keys = [k for k in mat.keys() if not k.startswith("_")]
                    arr = mat[keys[0]]
                    return torch.from_numpy(arr).float(), ext
                obj = torch.load(p, map_location="cpu", weights_only=False)
                if isinstance(obj, torch.Tensor):
                    return obj.float(), ext or "(no ext)"
                if isinstance(obj, np.ndarray):
                    return torch.from_numpy(obj).float(), ext or "(no ext)"
            except Exception as e:
                print(f"  [WARN] Could not load {p}: {e}")
    return None, None


def sep(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


# ── main audit ────────────────────────────────────────────────────────────────
def audit(cfg):

    # ── 1. Label file ────────────────────────────────────────────────────────
    sep("1. LABEL FILE")
    label_path = cfg["label_xlsx"]
    assert os.path.exists(label_path), f"Label file not found: {label_path}"

    raw  = pd.read_excel(label_path)
    cols = list(raw.columns)

    # Rename unnamed days column
    try:
        doi = cols.index("date_of_outcome")
        cols[doi + 1] = "days_to_outcome"
    except (ValueError, IndexError):
        for i, c in enumerate(cols):
            if str(c).startswith("Unnamed:"):
                cols[i] = "days_to_outcome"
                break
    raw.columns = cols

    raw["code"]            = raw["code"].astype(str).str.zfill(4)
    raw["ANY OUTCOME"]     = pd.to_numeric(raw["ANY OUTCOME"],  errors="coerce")
    raw["days_to_outcome"] = pd.to_numeric(raw["days_to_outcome"], errors="coerce")

    print(f"  Rows         : {len(raw)}")
    print(f"  Columns      : {list(raw.columns[:8])} ...")
    print(f"  Patients     : {raw['code'].nunique()}")

    # ── 2. Outcome statistics ─────────────────────────────────────────────────
    sep("2. OUTCOME STATISTICS")
    n_event    = int((raw["ANY OUTCOME"] == 1).sum())
    n_censored = int((raw["ANY OUTCOME"] == 0).sum())
    n_missing  = int(raw["ANY OUTCOME"].isna().sum())
    print(f"  Event (=1)   : {n_event}")
    print(f"  Censored (=0): {n_censored}")
    print(f"  Missing      : {n_missing}")
    print(f"  Event rate   : {n_event / max(n_event+n_censored, 1):.1%}")

    valid_times = raw.loc[raw["days_to_outcome"] > 0, "days_to_outcome"]
    if len(valid_times):
        print(f"  Survival time (days)  "
              f"min={valid_times.min():.0f}  "
              f"median={valid_times.median():.0f}  "
              f"max={valid_times.max():.0f}")

    # ── 3. Embedding file scan ────────────────────────────────────────────────
    sep("3. EMBEDDING FILE SCAN")
    emb_dir    = cfg["endo_emb_dir"]
    prefix     = cfg["endo_emb_prefix"]

    both_found = one_found = none_found = 0
    shape_counts = {}
    ext_counts   = {}
    sample_shapes = []

    for _, row in raw.iterrows():
        code = str(row["code"]).zfill(4)
        found = []
        for sec in ("section1", "section2"):
            stem = f"{prefix}_pat{code}_{sec}"
            full = os.path.join(emb_dir, stem)
            t, ext = try_load(full)
            if t is not None:
                found.append(t.shape)
                shape_counts[str(tuple(t.shape))] = \
                    shape_counts.get(str(tuple(t.shape)), 0) + 1
                ext_counts[ext] = ext_counts.get(ext, 0) + 1
                if len(sample_shapes) < 3:
                    sample_shapes.append((f"pat{code}_{sec}", t.shape))

        if len(found) == 2:
            both_found += 1
        elif len(found) == 1:
            one_found  += 1
        else:
            none_found += 1

    print(f"  Patients with BOTH sections  : {both_found}")
    print(f"  Patients with ONE  section   : {one_found}")
    print(f"  Patients with NO   sections  : {none_found}")
    print(f"  Total usable patients        : {both_found + one_found}")
    print(f"\n  File extension counts : {ext_counts}")
    print(f"  Embedding shapes      : {shape_counts}")
    if sample_shapes:
        print(f"\n  Sample shapes:")
        for name, sh in sample_shapes:
            print(f"    {name}: {sh}")

    # ── 4. Config vs actual embedding dim ────────────────────────────────────
    sep("4. CONFIG CHECK")
    if shape_counts:
        most_common = max(shape_counts, key=shape_counts.get)
        actual_shape = eval(most_common)
        actual_dim = actual_shape[-1]          # last dim = feature dim
        cfg_dim    = cfg["endo_dim"]
        status = "OK" if actual_dim == cfg_dim else f"MISMATCH — update endo_dim to {actual_dim}"
        print(f"  config endo_dim  : {cfg_dim}")
        print(f"  actual embedding : {actual_shape}  →  dim = {actual_dim}")
        print(f"  Status           : {status}")

        if len(actual_shape) == 2:
            k = actual_shape[0]
            print(f"\n  Multi-frame detected: {k} frames per section")
            print(f"  MIL will pool over {k} frames/section "
                  f"(up to {2*k} frames for patients with both sections)")
    else:
        print(f"  No embedding files found in: {emb_dir}")
        print(f"  Check that emb_dir path is correct in config_picasso.py")

    # ── 5. Final summary ──────────────────────────────────────────────────────
    sep("5. READY-TO-TRAIN CHECK")
    checks = {
        "Label file exists"          : os.path.exists(label_path),
        "Embedding dir exists"       : os.path.exists(emb_dir),
        "At least 10 usable patients": (both_found + one_found) >= 10,
        "Events > 0"                 : n_event > 0,
    }
    all_pass = True
    for k, v in checks.items():
        status = "PASS" if v else "FAIL"
        if not v:
            all_pass = False
        print(f"  [{status}]  {k}")

    print()
    if all_pass:
        print("  All checks passed.  Run:  python main_picasso.py")
    else:
        print("  Fix the FAIL items above before training.")
    sep()


if __name__ == "__main__":
    audit(PICASSO_CONFIG)
