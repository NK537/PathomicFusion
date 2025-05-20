import torch
from torch.utils.data import Dataset
import pandas as pd

class GeneDataset(Dataset):
    def __init__(self, gene_expression_csv, patient_labels_csv):
        """
        Args:
            gene_expression_csv (str): Path to the cleaned gene expression CSV (indexed by TCGA ID).
            patient_labels_csv (str): Path to the merged survival + grade labels CSV (TCGA ID matching).
        """
        # Load gene expression data
        self.gene_df = pd.read_csv(gene_expression_csv, index_col=0)
        
        # Load patient labels
        self.labels_df = pd.read_csv(patient_labels_csv)

        # Create lookup dictionaries
        self.labels_lookup = {
            row['TCGA ID']: {
                'Survival months': row['Survival months'],
                'censored': row['censored'],
                'grade': row['Grade']
            }
            for _, row in self.labels_df.iterrows()
        }

        # Only keep patients common in gene and labels
        self.common_patients = [pid for pid in self.gene_df.index if pid in self.labels_lookup]

    def __len__(self):
        return len(self.common_patients)

    def __getitem__(self, idx):
        # Get patient ID
        patient_id = self.common_patients[idx]
        
        # Get gene feature vector
        gene_features = torch.tensor(self.gene_df.loc[patient_id].values, dtype=torch.float32)

        # Get labels
        labels = self.labels_lookup[patient_id]
        survival_time = torch.tensor(labels['Survival months'], dtype=torch.float32)
        event = torch.tensor(labels['censored'], dtype=torch.float32)

        # Robust way to handle Grade field safely
        grade_raw = labels['grade']
        try:
            grade_clean = int(float(grade_raw)) if pd.notna(grade_raw) else 0
        except:
            grade_clean = 0  # Default to 0 if completely invalid
        grade = torch.tensor(grade_clean, dtype=torch.long)
        return gene_features, survival_time, event, grade
