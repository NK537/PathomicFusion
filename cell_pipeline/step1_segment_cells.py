# cell_pipeline/step1_segment_cells.py
#
# STEP 1 — Cell segmentation using HoVer-Net.
#
# For each section PNG in section_dir, detects individual cell nuclei and
# saves a JSON file with centroids, bounding boxes, and cell types.
#
# HoVer-Net setup (run once on server):
#   git clone https://github.com/vqdang/hover_net.git
#   cd hover_net && pip install -r requirements.txt
#   # Download weights from: https://drive.google.com/file/d/1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR
#   # Place at: hover_net/pretrained_models/hovernet_fast_pannuke_type_tf2pytorch.tar
#
# Usage:
#   python3 cell_pipeline/step1_segment_cells.py
#
# Output:
#   data/Picasso/cell/cell_masks/{section_stem}.json
#   JSON structure:
#     { "nuc": { "0": {"centroid": [x, y], "bbox": [y1,x1,y2,x2], "type": 1}, ... } }

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cell_pipeline.config_cell import CELL_CONFIG

# ── Try to import HoVer-Net ──────────────────────────────────────────────────
HOVERNET_AVAILABLE = False
try:
    sys.path.append("hover_net/")
    from models.hovernet.run_desc import proc_valid_step_output
    HOVERNET_AVAILABLE = True
except ImportError:
    pass

# ── Fallback: scikit-image nucleus detector ──────────────────────────────────
# Simpler, less accurate — good for testing the pipeline end-to-end
# before HoVer-Net weights are downloaded.
SKIMAGE_AVAILABLE = False
try:
    from skimage import filters, measure, morphology, color
    SKIMAGE_AVAILABLE = True
except ImportError:
    pass


def segment_with_skimage(img_path: str) -> dict:
    """
    Fallback nucleus detector using scikit-image Otsu thresholding.
    Not as accurate as HoVer-Net but requires no extra model weights.

    Returns dict in same format as HoVer-Net JSON output.
    """
    img   = np.array(Image.open(img_path).convert("RGB"))
    gray  = color.rgb2gray(img)
    thresh = filters.threshold_otsu(gray)
    binary = gray < thresh                              # nuclei are darker
    binary = morphology.remove_small_objects(binary, min_size=50)
    binary = morphology.remove_small_holes(binary, area_threshold=100)
    labeled = measure.label(binary)
    props   = measure.regionprops(labeled)

    nuc = {}
    for i, p in enumerate(props):
        cy, cx = p.centroid
        y1, x1, y2, x2 = p.bbox
        nuc[str(i)] = {
            "centroid": [float(cx), float(cy)],
            "bbox":     [int(y1), int(x1), int(y2), int(x2)],
            "type":     1,   # unknown type — HoVer-Net gives per-class labels
        }
    return {"nuc": nuc}


def segment_with_hovernet(img_path: str, model, device: str) -> dict:
    """
    Run HoVer-Net inference on a single image.
    Adjust this function to match the actual HoVer-Net API in the cloned repo.

    See: hover_net/run_utils/engine/infer_engine.py for the full inference pipeline.
    """
    import torch
    from torchvision import transforms

    img    = Image.open(img_path).convert("RGB")
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    # HoVer-Net post-processing (simplified — check hover_net repo for details)
    # The actual post-processing is in hover_net/models/hovernet/post_proc.py
    raise NotImplementedError(
        "Integrate HoVer-Net post-processing here.\n"
        "See hover_net/models/hovernet/post_proc.py  ->  process()\n"
        "It converts the NP/HV/TP maps to instance masks and centroids."
    )


def load_hovernet(device: str):
    """
    Load HoVer-Net model weights.
    Download weights from:
      https://drive.google.com/file/d/1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR
    Place at: hover_net/pretrained_models/hovernet_fast_pannuke_type_tf2pytorch.tar
    """
    import torch
    ckpt = "hover_net/pretrained_models/hovernet_fast_pannuke_type_tf2pytorch.tar"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"HoVer-Net weights not found at: {ckpt}\n"
            "Download from: https://drive.google.com/file/d/1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR\n"
            "Or use --fallback flag to use the scikit-image detector instead."
        )
    # Load using HoVer-Net's own loader — adjust import path as needed
    from models.hovernet.net_desc import create_model
    model = create_model(mode="fast", nr_types=6)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["desc"], strict=True)
    return model.to(device).eval()


def segment_all_sections(
    section_dir:  str,
    cell_mask_dir: str,
    use_fallback:  bool = False,
    skip_existing: bool = True,
    device:        str  = "cpu",
):
    import torch
    os.makedirs(cell_mask_dir, exist_ok=True)

    pngs = sorted(
        f for f in os.listdir(section_dir)
        if f.lower().endswith(".png")
    )
    print(f"Found {len(pngs)} section images in {section_dir}")

    if not pngs:
        print("No PNG files found. Check section_dir in config.")
        return

    # Choose segmentation method
    if use_fallback or not HOVERNET_AVAILABLE:
        if not SKIMAGE_AVAILABLE:
            raise ImportError("pip install scikit-image  (needed for fallback detector)")
        print("Using scikit-image fallback nucleus detector.")
        model = None
    else:
        print("Loading HoVer-Net model...")
        model = load_hovernet(device)
        print("  HoVer-Net loaded.")

    saved = skipped = failed = 0

    for fname in pngs:
        stem     = os.path.splitext(fname)[0]
        img_path = os.path.join(section_dir, fname)
        out_path = os.path.join(cell_mask_dir, stem + ".json")

        if skip_existing and os.path.exists(out_path):
            skipped += 1
            continue

        try:
            if model is None:
                result = segment_with_skimage(img_path)
            else:
                result = segment_with_hovernet(img_path, model, device)

            n_cells = len(result.get("nuc", {}))
            with open(out_path, "w") as f:
                json.dump(result, f)

            print(f"  {fname}  ->  {n_cells} cells  ->  {os.path.basename(out_path)}")
            saved += 1

        except Exception as e:
            print(f"  [WARN] {fname}: {e}")
            failed += 1

    print(f"\n=== DONE ===  saved={saved}  skipped={skipped}  failed={failed}")
    print(f"Cell masks saved to: {cell_mask_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fallback",      action="store_true",
                        help="Use scikit-image fallback instead of HoVer-Net")
    parser.add_argument("--no-skip",       action="store_true",
                        help="Re-process even if JSON already exists")
    parser.add_argument("--device",        default="cpu")
    args = parser.parse_args()

    segment_all_sections(
        section_dir   = CELL_CONFIG["section_dir"],
        cell_mask_dir = CELL_CONFIG["cell_mask_dir"],
        use_fallback  = args.fallback,
        skip_existing = not args.no_skip,
        device        = args.device,
    )
