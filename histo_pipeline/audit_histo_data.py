# histo_pipeline/audit_histo_data.py
#
# Run BEFORE training to verify data is correctly wired.
#   python histo_pipeline/audit_histo_data.py

import os, sys, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from histo_pipeline.config_histo import HISTO_CONFIG
from histo_pipeline.patient_histo_dataset import filename_to_pat_id, normalize_pat_id, load_labels


def sep(title=""):
    print("\n" + "=" * 62)
    if title: print(f"  {title}"); print("=" * 62)


def audit(cfg):

    sep("1. FUSION LABEL FILE")
    fpath = cfg["fusion_label_xlsx"]
    assert os.path.exists(fpath), f"NOT FOUND: {fpath}"
    df = pd.read_excel(fpath)
    df["pat_id"] = df["Pat_ID"].apply(normalize_pat_id)
    df["event"]  = pd.to_numeric(df["Outcome"], errors="coerce")
    df["split"]  = pd.to_numeric(df[cfg["split_col"]], errors="coerce").fillna(-1).astype(int)
    print(f"  Rows          : {len(df)}")
    print(f"  Unique pat_id : {df['pat_id'].nunique()}")
    train_df = df[df["split"].isin(cfg["train_vals"]) & df["event"].isin([0.0, 1.0])]
    print(f"  Train pool    : {len(train_df)} patients")
    print(f"    Events      : {(train_df['event']==1).sum()}")
    print(f"    Censored    : {(train_df['event']==0).sum()}")
    test_df = df[df["split"].isin(cfg["test_vals"])]
    print(f"  Test pool     : {len(test_df)} patients")

    sep("2. TTE LABEL FILE")
    tte_path = cfg.get("tte_label_xlsx", "")
    if tte_path and os.path.exists(tte_path):
        tte = pd.read_excel(tte_path)
        print(f"  Found: {tte_path}")
        print(f"  Columns: {list(tte.columns)}")
        print(f"  Rows: {len(tte)}")
        print("  -> Cox loss ENABLED")
    else:
        print(f"  NOT FOUND: {tte_path}")
        print("  -> Will use BCE loss (binary outcome only)")
        print("  -> For Cox loss: place PICASSO_outcome_tte.xlsx in data/Picasso/histo/")

    sep("3. PATCH IMAGE DIRECTORY")
    patch_dir = cfg["histo_patch_dir"]
    if not os.path.isdir(patch_dir):
        print(f"  NOT FOUND: {patch_dir}")
        print("  Options:")
        print("    a) Copy patches to that folder")
        print("    b) Update histo_patch_dir in config_histo.py")
        print("       e.g. /mnt/d/Data/PICASSO_Histology/.../patch_neutrophils/images/")
        png_count = 0
    else:
        pngs = [f for f in os.listdir(patch_dir) if f.lower().endswith(".png")]
        png_count = len(pngs)
        print(f"  Patch images  : {png_count}")
        if pngs:
            # Unique patient IDs
            pat_ids = set()
            for f in pngs:
                pid = filename_to_pat_id(f)
                if pid: pat_ids.add(pid)
            print(f"  Unique patients: {len(pat_ids)}")
            samples = sorted(pngs)[:5]
            print(f"  Sample filenames:")
            for s in samples:
                print(f"    {s}  ->  pat_id={filename_to_pat_id(s)}")

    sep("4. EMBEDDING DIRECTORY")
    emb_dir = cfg["histo_emb_dir"]
    if not os.path.isdir(emb_dir):
        print(f"  NOT FOUND: {emb_dir}")
        if png_count > 0:
            print(f"  Run: python precompute_histo_embeddings.py")
        else:
            print(f"  First provide patch images, then run precompute_histo_embeddings.py")
    else:
        emb_files = [f for f in os.listdir(emb_dir)
                     if f.endswith(".pt") or f.endswith(".npy")]
        print(f"  Embedding files : {len(emb_files)}")
        if emb_files:
            # Sample shapes
            pat_ids_emb = set()
            shapes = {}
            for f in emb_files[:200]:   # sample first 200
                pid = filename_to_pat_id(f)
                if pid: pat_ids_emb.add(pid)
                fp = os.path.join(emb_dir, f)
                try:
                    t  = torch.load(fp, map_location="cpu", weights_only=False)
                    if hasattr(t, "shape"):
                        k = str(tuple(t.shape))
                        shapes[k] = shapes.get(k, 0) + 1
                except Exception:
                    pass
            print(f"  Unique patients : {len(pat_ids_emb)}")
            print(f"  Shape counts    : {shapes}")
            if shapes:
                most_common = max(shapes, key=shapes.get)
                actual_dim  = eval(most_common)[-1]
                cfg_dim     = cfg["histo_dim"]
                ok = actual_dim == cfg_dim
                print(f"  config histo_dim: {cfg_dim}")
                print(f"  actual dim      : {actual_dim}")
                print(f"  Status          : {'OK' if ok else f'MISMATCH -> update histo_dim to {actual_dim}'}")

    sep("5. PATIENT MATCHING")
    emb_dir = cfg["histo_emb_dir"]
    if os.path.isdir(emb_dir):
        emb_ids = set()
        for f in os.listdir(emb_dir):
            pid = filename_to_pat_id(f)
            if pid: emb_ids.add(pid)
        label_ids = set(train_df["pat_id"].tolist())
        matched   = label_ids & emb_ids
        print(f"  Label train patients  : {len(label_ids)}")
        print(f"  Embedding patients    : {len(emb_ids)}")
        print(f"  Matched (can train)   : {len(matched)}")
        unmatched = label_ids - emb_ids
        if unmatched:
            print(f"  No embeddings for    : {sorted(unmatched)[:10]}")

    sep("6. READY-TO-TRAIN CHECKLIST")
    checks = {
        "Fusion label file exists"  : os.path.exists(cfg["fusion_label_xlsx"]),
        "Train patients found"      : len(train_df) > 0,
        "Events > 0 in train"       : int((train_df["event"]==1).sum()) > 0,
        "Embedding dir exists"      : os.path.isdir(cfg["histo_emb_dir"]),
    }
    all_pass = True
    for k, v in checks.items():
        if not v: all_pass = False
        print(f"  [{'PASS' if v else 'FAIL'}]  {k}")
    print()
    if all_pass:
        print("  All required checks passed.")
        print("  Run:  python histo_pipeline/main_histo.py")
    else:
        print("  Fix FAIL items above before training.")
    sep()


if __name__ == "__main__":
    audit(HISTO_CONFIG)
