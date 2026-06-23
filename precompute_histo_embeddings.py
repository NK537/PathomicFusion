import os
import sys

# ── Auto-inject the project venv so this script works regardless of which
#    Python interpreter is used to launch it.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _venv_name in (".pathomic", "pathomic", "venv", ".venv"):
    _sp = os.path.join(_HERE, _venv_name, "lib")
    if os.path.isdir(_sp):
        import glob as _glob
        for _d in _glob.glob(os.path.join(_sp, "python*", "site-packages")):
            if _d not in sys.path:
                sys.path.insert(0, _d)
        break

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Spatiopath model wrapper — adjust the import once you have the repo installed.
# Clone:  git clone https://gitlab.pasteur.fr/bia/projects/Spatiopath.git
# Install: cd Spatiopath && pip install -e .
# Then uncomment the import below and set SPATIOPATH_AVAILABLE = True.
SPATIOPATH_AVAILABLE = False
try:
    sys.path.append("Spatiopath/")
    from spatiopath import get_spatiopath_encoder   # adjust to actual API
    SPATIOPATH_AVAILABLE = True
except ImportError:
    pass

# Fallback: use UNI (already in repo) until Spatiopath is installed.
UNI_AVAILABLE = False
try:
    sys.path.append("UNI/")
    from uni import get_encoder as get_uni_encoder
    UNI_AVAILABLE = True
except ImportError:
    pass

# Embedding dim: 1024 for UNI / check Spatiopath paper for its dim.
# Update HISTO_EMBED_DIM after confirming the Spatiopath output size.
HISTO_EMBED_DIM = 1024


def _load_uni_direct(device):
    """
    Load UNI weights directly from the local repo — no HuggingFace download.
    Looks for pytorch_model.bin in UNI/assets/ckpts/uni/ relative to this script.
    """
    import timm
    from torchvision import transforms as T

    # Search for weights: try next to this script first, then CWD
    candidates = [
        os.path.join(_HERE, "UNI", "assets", "ckpts", "uni", "pytorch_model.bin"),
        os.path.join(os.getcwd(), "UNI", "assets", "ckpts", "uni", "pytorch_model.bin"),
    ]
    ckpt = None
    for c in candidates:
        if os.path.isfile(c):
            ckpt = c
            break

    if ckpt is None:
        raise FileNotFoundError(
            "UNI weights not found. Expected at:\n" +
            "\n".join(f"  {c}" for c in candidates)
        )

    print(f"  Loading UNI weights from: {ckpt}")
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return model, transform


def load_histo_model(device, use_spatiopath=False):
    """
    Load the histopathology foundation model.

    Priority order:
      1. Spatiopath  (if installed and use_spatiopath=True)
      2. UNI direct  (loads from local UNI/assets/ckpts/uni/pytorch_model.bin)
    """
    if use_spatiopath and SPATIOPATH_AVAILABLE:
        print("Loading Spatiopath encoder...")
        model, transform = get_spatiopath_encoder(device=device)
        print("  Spatiopath loaded.")
        return model, transform

    print("Loading UNI encoder from local weights...")
    model, transform = _load_uni_direct(device)
    print("  UNI loaded. Embedding dim: 1024")
    return model, transform


class HistoPatchDataset(Dataset):
    """
    Expected CSV columns:
        patient_id      - Picasso patient identifier
        patch_filename  - filename of the histopathology patch image

    Returns (tensor, save_path) per patch.
    """
    def __init__(self, patch_csv, patch_dir, out_dir, transform, skip_existing=True):
        df = pd.read_csv(patch_csv)
        self.patch_files = sorted(set(df["patch_filename"].astype(str).tolist()))
        self.patch_dir   = patch_dir
        self.out_dir     = out_dir
        self.transform   = transform
        os.makedirs(out_dir, exist_ok=True)

        if skip_existing:
            self.patch_files = [
                fn for fn in self.patch_files
                if not os.path.exists(
                    os.path.join(out_dir, os.path.splitext(fn)[0] + ".pt")
                )
            ]
        print(f"  Patches to embed: {len(self.patch_files)}")

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        fn       = self.patch_files[idx]
        img      = Image.open(os.path.join(self.patch_dir, fn)).convert("RGB")
        x        = self.transform(img)
        out_path = os.path.join(self.out_dir, os.path.splitext(fn)[0] + ".pt")
        return x, out_path


def precompute_histo_embeddings(
    patch_csv,
    patch_dir,
    out_dir,
    use_spatiopath=False,
    batch_size=64,
    num_workers=4,
    device=None,
    fp16=True,
    skip_existing=True,
):
    """
    Precompute histopathology foundation-model embeddings for all patches.

    Saves one .pt file per patch under out_dir.
    Default model: UNI (1024-D). Switch to Spatiopath by setting
    use_spatiopath=True once the repo is installed.

    Args:
        patch_csv:       CSV with columns [patient_id, patch_filename]
        patch_dir:       Root directory containing the raw patch images
        out_dir:         Root directory where .pt files will be written
        use_spatiopath:  Use Spatiopath instead of UNI when available
        batch_size:      Images per GPU batch
        fp16:            AMP half-precision on GPU
        skip_existing:   Skip patches whose .pt already exists
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, transform = load_histo_model(device, use_spatiopath)
    model.eval()

    ds = HistoPatchDataset(patch_csv, patch_dir, out_dir, transform, skip_existing)
    if len(ds) == 0:
        print("Nothing to embed — all patches already processed.")
        return

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=False,
    )

    use_amp = (device == "cuda") and fp16
    saved = failed = 0

    with torch.inference_mode():
        for xb, out_paths in tqdm(loader, desc="Histo embeddings"):
            try:
                xb = xb.to(device, non_blocking=True)

                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        emb = model(xb)
                else:
                    emb = model(xb)

                emb = emb.float().detach().cpu()

                for i, op in enumerate(out_paths):
                    parent = os.path.dirname(op)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    torch.save(emb[i], op)

                saved += len(out_paths)

            except Exception as e:
                print(f"\n[WARN] batch failed: {e}")
                failed += len(out_paths)

    print(f"\n=== DONE ===  saved={saved}  failed={failed}")

    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.endswith(".pt"):
                x = torch.load(os.path.join(root, f), map_location="cpu")
                print(f"Sample embedding shape: {x.shape}")
                return


if __name__ == "__main__":
    # ── Patch images are PNG files named:  02_003 Sigmoid 1E_16_146.png
    # ── Set patch_dir to wherever your images live, e.g.:
    #      /mnt/d/Data/PICASSO_Histology/PicassoHistologyOLD/Histo_Neutriphils/patch_neutrophils/images/
    #    or copy them to:
    #      data/Picasso/histo/patch_images/
    #
    # ── No CSV required — we scan patch_dir directly.
    #    out_dir receives one .pt per patch with the same stem name.

    import os, sys, re
    from tqdm import tqdm
    from PIL import Image

    patch_dir  = "data/Picasso/histo_new/sections/"
    out_dir    = "data/Picasso/histo_new/histo_embeddings/"
    batch_size = 32
    num_workers = 4
    fp16        = True
    skip_existing = True

    # --- validate patch_dir ---
    if not os.path.isdir(patch_dir):
        print(f"ERROR: patch_dir not found: {patch_dir}")
        print("Update patch_dir above to the correct path and re-run.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, transform = load_histo_model(device, use_spatiopath=False)
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    # Collect all PNG files
    all_pngs = sorted(f for f in os.listdir(patch_dir) if f.lower().endswith(".png"))
    print(f"Total PNG patches found: {len(all_pngs)}")

    if skip_existing:
        all_pngs = [
            f for f in all_pngs
            if not os.path.exists(os.path.join(out_dir, os.path.splitext(f)[0] + ".pt"))
        ]
        print(f"Patches remaining (skip_existing=True): {len(all_pngs)}")

    if not all_pngs:
        print("Nothing to embed — all patches already processed.")
        sys.exit(0)

    use_amp  = (device == "cuda") and fp16
    saved    = 0
    failed   = 0
    batch_imgs  = []
    batch_paths = []

    def flush_batch(imgs, paths):
        global saved, failed
        try:
            xb  = torch.stack(imgs).to(device, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    emb = model(xb)
            else:
                emb = model(xb)
            emb = emb.float().detach().cpu()
            for i, op in enumerate(paths):
                torch.save(emb[i], op)
            saved += len(paths)
        except Exception as e:
            print(f"\n[WARN] batch failed: {e}")
            failed += len(paths)

    with torch.inference_mode():
        for fname in tqdm(all_pngs, desc="Histo embeddings"):
            img_path = os.path.join(patch_dir, fname)
            out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".pt")
            try:
                img = Image.open(img_path).convert("RGB")
                x   = transform(img)
                batch_imgs.append(x)
                batch_paths.append(out_path)
            except Exception as e:
                print(f"\n[WARN] could not open {fname}: {e}")
                failed += 1
                continue

            if len(batch_imgs) == batch_size:
                flush_batch(batch_imgs, batch_paths)
                batch_imgs  = []
                batch_paths = []

        if batch_imgs:   # flush remainder
            flush_batch(batch_imgs, batch_paths)

    print(f"\n=== DONE ===  saved={saved}  failed={failed}  out_dir={out_dir}")

    # Show sample shape
    for f in os.listdir(out_dir):
        if f.endswith(".pt"):
            x = torch.load(os.path.join(out_dir, f), map_location="cpu")
            print(f"Sample embedding shape: {x.shape}  (expected: torch.Size([1024]))")
            break
