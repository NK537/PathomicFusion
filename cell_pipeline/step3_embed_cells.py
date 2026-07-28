# cell_pipeline/step3_embed_cells.py
#
# STEP 3 — Embed cell crops using Spatiopath (or UNI as fallback).
#
# Walks cell_patch_dir/{section_stem}/*.png, runs each crop through the
# foundation model, and saves a .pt tensor per cell.
#
# Output:
#   cell_embeddings/{section_stem}/{section_stem}_{cell_id}.pt
#   shape: (cell_emb_dim,)  e.g. (1024,) for UNI / check Spatiopath
#
# Usage:
#   python3 cell_pipeline/step3_embed_cells.py [--spatiopath] [--device cuda]

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cell_pipeline.config_cell import CELL_CONFIG

Image.MAX_IMAGE_PIXELS = None   # allow large images


# ── Model loaders ─────────────────────────────────────────────────────────────

def _load_spatiopath(device: str):
    """
    Load Spatiopath encoder.
    Requires: git clone https://gitlab.pasteur.fr/bia/projects/Spatiopath.git
              pip install -e Spatiopath/
    """
    sys.path.insert(0, "Spatiopath/")
    try:
        from spatiopath import get_spatiopath_encoder
    except ImportError:
        raise ImportError(
            "Spatiopath not installed.\n"
            "Run: git clone https://gitlab.pasteur.fr/bia/projects/Spatiopath.git\n"
            "     pip install -e Spatiopath/\n"
            "Or omit --spatiopath to use UNI as fallback."
        )
    print("Loading Spatiopath encoder...")
    model, transform = get_spatiopath_encoder(device=device)
    model = model.to(device).eval()
    print("  Spatiopath loaded.")
    return model, transform


def _load_uni(device: str):
    """
    Load UNI ViT-L/16 from local weights.
    Weights expected at: UNI/assets/ckpts/uni/pytorch_model.bin
    """
    import timm
    _here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(_here,        "UNI", "assets", "ckpts", "uni", "pytorch_model.bin"),
        os.path.join(os.getcwd(),  "UNI", "assets", "ckpts", "uni", "pytorch_model.bin"),
    ]
    ckpt = next((c for c in candidates if os.path.isfile(c)), None)
    if ckpt is None:
        raise FileNotFoundError(
            "UNI weights not found. Expected at:\n" +
            "\n".join(f"  {c}" for c in candidates)
        )
    print(f"Loading UNI weights from: {ckpt}")
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224, patch_size=16, init_values=1e-5,
        num_classes=0, dynamic_img_size=True,
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    print("  UNI loaded (embedding dim: 1024).")
    return model, transform


def load_model(device: str, use_spatiopath: bool = False):
    if use_spatiopath:
        return _load_spatiopath(device)
    return _load_uni(device)


# ── Embedding loop ────────────────────────────────────────────────────────────

def embed_all_cells(
    cell_patch_dir: str,
    cell_emb_dir:   str,
    use_spatiopath: bool = False,
    batch_size:     int  = 32,
    device:         str  = "cpu",
    skip_existing:  bool = True,
    fp16:           bool = True,
):
    os.makedirs(cell_emb_dir, exist_ok=True)
    model, transform = load_model(device, use_spatiopath)
    use_amp = (device == "cuda") and fp16

    # Collect all cell PNG paths that still need embedding
    to_process = []
    for section_stem in sorted(os.listdir(cell_patch_dir)):
        section_patch_dir = os.path.join(cell_patch_dir, section_stem)
        if not os.path.isdir(section_patch_dir):
            continue
        section_emb_dir = os.path.join(cell_emb_dir, section_stem)
        os.makedirs(section_emb_dir, exist_ok=True)

        for fname in os.listdir(section_patch_dir):
            if not fname.lower().endswith(".png"):
                continue
            stem     = os.path.splitext(fname)[0]
            img_path = os.path.join(section_patch_dir, fname)
            out_path = os.path.join(section_emb_dir, stem + ".pt")

            if skip_existing and os.path.exists(out_path):
                continue
            to_process.append((img_path, out_path))

    print(f"Cells to embed: {len(to_process)}")
    if not to_process:
        print("All cells already embedded.")
        return

    saved = failed = 0
    batch_imgs, batch_paths = [], []

    def flush(imgs, paths):
        nonlocal saved, failed
        try:
            xb = torch.stack(imgs).to(device)
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    emb = model(xb)
            else:
                emb = model(xb)
            emb = emb.float().detach().cpu()
            for i, p in enumerate(paths):
                torch.save(emb[i], p)
            saved += len(paths)
        except Exception as e:
            print(f"\n[WARN] batch failed: {e}")
            failed += len(paths)

    with torch.inference_mode():
        for img_path, out_path in tqdm(to_process, desc="Embedding cells"):
            try:
                img = Image.open(img_path).convert("RGB")
                x   = transform(img)
                batch_imgs.append(x)
                batch_paths.append(out_path)
            except Exception as e:
                print(f"\n[WARN] cannot open {img_path}: {e}")
                failed += 1
                continue

            if len(batch_imgs) == batch_size:
                flush(batch_imgs, batch_paths)
                batch_imgs, batch_paths = [], []

        if batch_imgs:
            flush(batch_imgs, batch_paths)

    print(f"\n=== DONE ===  saved={saved}  failed={failed}")

    # Print sample shape
    for root, _, files in os.walk(cell_emb_dir):
        for f in files:
            if f.endswith(".pt"):
                x = torch.load(os.path.join(root, f), map_location="cpu")
                print(f"Sample embedding shape: {x.shape}")
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatiopath", action="store_true",
                        help="Use Spatiopath instead of UNI")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-skip",    action="store_true")
    parser.add_argument("--no-fp16",    action="store_true")
    args = parser.parse_args()

    embed_all_cells(
        cell_patch_dir = CELL_CONFIG["cell_patch_dir"],
        cell_emb_dir   = CELL_CONFIG["cell_emb_dir"],
        use_spatiopath = args.spatiopath,
        batch_size     = args.batch_size,
        device         = args.device,
        skip_existing  = not args.no_skip,
        fp16           = not args.no_fp16,
    )
