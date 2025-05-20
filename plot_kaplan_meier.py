
import torch
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from models.CNN import CNNBranch
from models.MLP import MLPBranch
from fusion.attention_fusion import AttentionFusion
from pytorch_dataset_loader.patches_pytorch_dataset import PatchDataset
from pytorch_dataset_loader.pytorch_GeneDataset import GeneDataset
from torch.utils.data import DataLoader

# === Config ===
cancer_name = "GBMLGG"
checkpoint_path = f"best_model_{cancer_name}.pth"
csv_patch = "data/TCGA_GBMLGG/patches_with_labels.csv"
img_dir = "data/TCGA_GBMLGG/patches/"
gene_csv = "data/TCGA_GBMLGG/gene data/clean_gene_expression.csv"
label_csv = "data/TCGA_GBMLGG/merged_all_dataset_and_grade_data.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Datasets ===
patch_dataset = PatchDataset(csv_file=csv_patch, image_dir=img_dir)
gene_dataset = GeneDataset(gene_expression_csv=gene_csv, patient_labels_csv=label_csv)

# Use only validation set (e.g., split fixed like before)
from sklearn.model_selection import train_test_split
all_ids = patch_dataset.data_frame['TCGA_ID'].unique()
_, val_ids = train_test_split(all_ids, test_size=0.2, random_state=42)

val_dataset = PatchDataset(csv_file=csv_patch, image_dir=img_dir, subset_ids=val_ids)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# === Load Models ===
cnn_branch = CNNBranch(feature_dim=64).to(device)
mlp_branch = MLPBranch(input_dim=gene_dataset.gene_df.shape[1], feature_dim=64).to(device)
fusion_layer = AttentionFusion(input_dim=64, fusion_dim=128).to(device)

checkpoint = torch.load(checkpoint_path)
cnn_branch.load_state_dict(checkpoint['cnn'])
mlp_branch.load_state_dict(checkpoint['mlp'])
fusion_layer.load_state_dict(checkpoint['fusion'])

cnn_branch.eval()
mlp_branch.eval()
fusion_layer.eval()

# === Inference ===
all_risks, all_times, all_events = [], [], []

with torch.no_grad():
    for batch in val_loader:
        patches, surv_times, events, _, patient_ids = batch

        valid_patches, valid_gene_vectors = [], []

        for i, pid in enumerate(patient_ids):
            if pid in gene_dataset.gene_df.index:
                gene_vector = torch.tensor(gene_dataset.gene_df.loc[pid].values, dtype=torch.float32)
                valid_gene_vectors.append(gene_vector)
                valid_patches.append(patches[i])

        if not valid_patches:
            continue

        x_img = torch.stack(valid_patches).to(device)
        x_gene = torch.stack(valid_gene_vectors).to(device)

        img_feat = cnn_branch(x_img)
        gene_feat = mlp_branch(x_gene)
        risk = fusion_layer(img_feat, gene_feat)

        all_risks.extend(risk.cpu().numpy())
        all_times.extend(surv_times.cpu().numpy())
        all_events.extend(events.cpu().numpy())

# === Sanity check ===
print(f"Lengths -> times: {len(all_times)}, events: {len(all_events)}, risks: {len(all_risks)}")

# === Prepare DataFrame with min-aligned lengths ===
min_len = min(len(all_times), len(all_events), len(all_risks))
df = pd.DataFrame({
    "time": all_times[:min_len],
    "event": all_events[:min_len],
    "risk": all_risks[:min_len]
})
df["group"] = df["risk"] > df["risk"].median()

# === Kaplan-Meier Plot ===
kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 6))

for label, grouped_df in df.groupby("group"):
    name = "High Risk" if label else "Low Risk"
    kmf.fit(grouped_df["time"], grouped_df["event"], label=name)
    kmf.plot_survival_function()

plt.title("Kaplan–Meier Survival Curves by Predicted Risk Group")
plt.xlabel("Time (days)")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Log-Rank Test ===
group_high = df[df["group"] == True]
group_low = df[df["group"] == False]
results = logrank_test(group_high["time"], group_low["time"],
                       event_observed_A=group_high["event"],
                       event_observed_B=group_low["event"])

print("Log-rank test p-value:", results.p_value)
