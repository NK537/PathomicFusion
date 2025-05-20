import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from lifelines.utils import concordance_index

from models.CNN import CNNBranch
from models.MLP import MLPBranch
from pytorch_dataset_loader.patches_pytorch_dataset import PatchDataset
from pytorch_dataset_loader.pytorch_GeneDataset import GeneDataset
from fusion.attention_fusion import AttentionFusion
from COX.cox_loss import CustomCoxLoss
from train_for_cancer import train_for_cancer

# ------------------------------
# Dataset Configurations
# ------------------------------
cancers = {
    "GBMLGG": {
        "patch_csv": "data/TCGA_GBMLGG/patches_with_labels.csv",
        "patch_dir": "data/TCGA_GBMLGG/patches/",
        "gene_csv": "data/TCGA_GBMLGG/gene data/clean_gene_expression.csv",
        "label_csv": "data/TCGA_GBMLGG/merged_all_dataset_and_grade_data.csv"
    },
    # "KIRC": {
    #     "patch_csv": "data/TCGA_KIRC/patches_with_labels.csv",
    #     "patch_dir": "data/TCGA_KIRC/patches/",
    #     "gene_csv": "data/TCGA_KIRC/gene_expression.csv",
    #     "label_csv": "data/TCGA_KIRC/merged_all_dataset_and_grade_data.csv"
    # }
}

# ------------------------------
# Run for each Cancer
# ------------------------------
for cancer, data in cancers.items():
    train_for_cancer(cancer, data)
