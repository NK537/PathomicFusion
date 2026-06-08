import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel


GASTRONET_HF_ID = "tgwboers/GastroNet-5M_Pretrained_Weights"
EMBED_DIM = 384   # ViT-small/16 CLS token


def build_gastronet_transform():
    """
    Standard ViT-s/16 preprocessing.
    GastroNet images are 512x512; resize to 224x224 for the ViT patch grid.
    ImageNet mean/std — consistent with the DINO pre-training setup.
    """
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_gastronet_model(device):
    print(f"Loading GastroNet-5M from HuggingFace: {GASTRONET_HF_ID}")
    model = ViTModel.from_pretrained(GASTRONET_HF_ID, add_pooling_layer=False)
    model.eval()
    model.to(device)
    print(f"  Loaded. Embedding dim: {EMBED_DIM}")
    return model


def extract_cls(model, pixel_values):
    """Returns CLS token: (B, 384)."""
    outputs = model(pixel_values=pixel_values)
    return outputs.last_hidden_state[:, 0, :]


class EndoFrameDataset(Dataset):
    """
    Reads frame filenames from a CSV.

    Expected CSV columns:
        patient_id     - Picasso patient identifier
        frame_filename - filename of the endoscopy frame image

    Returns (tensor, save_path) per frame.
    """
    def __init__(self, frame_csv, frame_dir, out_dir, transform, skip_existing=True):
        df = pd.read_csv(frame_csv)
        self.frame_files = sorted(set(df["frame_filename"].astype(str).tolist()))
        self.frame_dir   = frame_dir
        self.out_dir     = out_dir
        self.transform   = transform
        os.makedirs(out_dir, exist_ok=True)

        if skip_existing:
            self.frame_files = [
                fn for fn in self.frame_files
                if not os.path.exists(
                    os.path.join(out_dir, os.path.splitext(fn)[0] + ".pt")
                )
            ]
        print(f"  Frames to embed: {len(self.frame_files)}")

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        fn       = self.frame_files[idx]
        img      = Image.open(os.path.join(self.frame_dir, fn)).convert("RGB")
        x        = self.transform(img)
        out_path = os.path.join(self.out_dir, os.path.splitext(fn)[0] + ".pt")
        return x, out_path


def precompute_endo_embeddings(
    frame_csv,
    frame_dir,
    out_dir,
    hf_model_id=GASTRONET_HF_ID,
    batch_size=64,
    num_workers=4,
    device=None,
    fp16=True,
    skip_existing=True,
):
    """
    Precompute GastroNet-5M CLS-token embeddings (384-D) for every
    endoscopy frame listed in frame_csv.

    Saves one .pt file per frame (shape: torch.Size([384])) under out_dir,
    mirroring the layout used by precompute_uni_embeddings.py so the rest
    of the bimodal pipeline works without changes.

    Args:
        frame_csv:     CSV with columns [patient_id, frame_filename]
        frame_dir:     Root directory containing the raw frame images
        out_dir:       Root directory where .pt files will be written
        batch_size:    Images per GPU batch (lower if OOM)
        fp16:          AMP half-precision on GPU
        skip_existing: Skip frames whose .pt file already exists
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model     = load_gastronet_model(device)
    transform = build_gastronet_transform()

    ds = EndoFrameDataset(frame_csv, frame_dir, out_dir, transform, skip_existing)
    if len(ds) == 0:
        print("Nothing to embed — all frames already processed.")
        return

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=False,
    )

    use_amp = (device == "cuda") and fp16
    saved = failed = 0

    with torch.inference_mode():
        for xb, out_paths in tqdm(loader, desc="GastroNet-5M embeddings"):
            try:
                xb = xb.to(device, non_blocking=True)

                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        emb = extract_cls(model, xb)
                else:
                    emb = extract_cls(model, xb)

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
                print(f"Sample embedding shape: {x.shape}  (expected: [{EMBED_DIM}])")
                return


if __name__ == "__main__":
    precompute_endo_embeddings(
        frame_csv   = "data/Picasso/endo_frames.csv",
        frame_dir   = "data/Picasso/endo_frames/",
        out_dir     = "data/Picasso/endo_embeddings/",
        batch_size  = 64,
        num_workers = 4,
        fp16        = True,
        skip_existing = True,
    )
