import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append("UNI/")  # Adjust if your UNI repo is in a different location

# UNI repo import (you already cloned UNI/)
from uni import get_encoder


class PatchCSVDataset(Dataset):
    """
    Loads patch images listed in patches_with_labels.csv and returns:
      - transformed tensor
      - output .pt path to save embedding
    """
    def __init__(self, patch_csv: str, patch_dir: str, out_dir: str, transform, skip_existing: bool = True):
        self.df = pd.read_csv(patch_csv)

        # Your column name:
        self.patch_col = "patch_filename"

        # Use unique patches only (csv can have repeats)
        patch_files = self.df[self.patch_col].astype(str).tolist()
        self.patch_files = sorted(set(patch_files))

        self.patch_dir = patch_dir
        self.out_dir = out_dir
        self.transform = transform
        self.skip_existing = skip_existing

        os.makedirs(out_dir, exist_ok=True)

        # Filter out those already embedded (optional)
        if skip_existing:
            keep = []
            for fn in self.patch_files:
                base = os.path.splitext(fn)[0]
                out_path = os.path.join(self.out_dir, base + ".pt")
                if not os.path.exists(out_path):
                    keep.append(fn)
            self.patch_files = keep

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        fn = self.patch_files[idx]
        img_path = os.path.join(self.patch_dir, fn)
        base = os.path.splitext(fn)[0]
        out_path = os.path.join(self.out_dir, base + ".pt")

        # Load + transform image into 224x224 tensor (UNI's default)
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)  # tensor (3,H,W)

        return x, out_path


def precompute_uni_embeddings_fast(
    patch_csv: str,
    patch_dir: str,
    out_dir: str,
    enc_name: str = "uni",  # UNI-1
    batch_size: int = 64,
    num_workers: int = 4,
    device: str | None = None,
    fp16: bool = True,
    skip_existing: bool = True,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(out_dir, exist_ok=True)

    # Load UNI encoder + transform (UNI handles resize/normalization)
    model, transform = get_encoder(enc_name=enc_name, device=device)
    model.eval()

    ds = PatchCSVDataset(
        patch_csv=patch_csv,
        patch_dir=patch_dir,
        out_dir=out_dir,
        transform=transform,
        skip_existing=skip_existing
    )

    if len(ds) == 0:
        print("No patches to embed (all exist already).")
        return

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False
    )

    use_amp = (device == "cuda") and fp16
    autocast_ctx = torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast

    saved = 0
    failed = 0

    # Progress bar
    pbar = tqdm(loader, desc=f"UNI embeddings ({enc_name})", total=len(loader))

    with torch.inference_mode():
        for xb, out_paths in pbar:
            try:
                xb = xb.to(device, non_blocking=True)  # (B,3,H,W)

                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        emb = model(xb)  # (B, D)
                else:
                    emb = model(xb)

                emb = emb.detach().cpu()

                # Save each embedding
                for i, op in enumerate(out_paths):
                    torch.save(emb[i], op)
                saved += len(out_paths)

                pbar.set_postfix(saved=saved, failed=failed)

            except Exception:
                # If a batch fails, fallback to per-item to isolate the bad file
                failed += len(out_paths)
                pbar.set_postfix(saved=saved, failed=failed)

    print("\n=== DONE ===")
    print(f"Saved embeddings: {saved}")
    print(f"Failed embeddings: {failed}")

    # Print a sample embedding dim
    # (Load any one file that was created)
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.endswith(".pt"):
                x = torch.load(os.path.join(root, f), map_location="cpu")
                print("Example embedding shape:", x.shape)
                return


if __name__ == "__main__":
    precompute_uni_embeddings_fast(
        patch_csv="data/TCGA_GBMLGG/patches_with_labels.csv",
        patch_dir="data/TCGA_GBMLGG/patches/",
        out_dir="data/TCGA_GBMLGG/uni_embeddings/",
        enc_name="uni",        # UNI-1
        batch_size=64,         # try 128 if VRAM allows
        num_workers=4,         # try 8 if you have CPU cores
        fp16=True,             # AMP on GPU
        skip_existing=True
    )