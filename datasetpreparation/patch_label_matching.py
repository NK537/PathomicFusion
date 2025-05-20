import os
import pandas as pd

# Step 1: Load the patient feature table
patient_feature_df = pd.read_csv('../data/TCGA_GBMLGG/patient_feature_table.csv')  # Replace with actual path

# Step 2: Load gene dataset to get valid TCGA IDs
# gene_dataset_df = pd.read_csv('../data/TCGA_GBMLGG/FinalDataset/gene_expression.csv')  # Replace with actual path
gene_dataset_df = pd.read_csv('../data/TCGA_KIRC/clean_gene_expression.csv')  # Replace with actual path
gene_patient_ids = set(gene_dataset_df['TCGA ID'])

# Step 3: Build a quick lookup dictionary: TCGA ID → (survival_time, event, grade)
patient_feature_dict = {}
for idx, row in patient_feature_df.iterrows():
    if row['TCGA ID'] in gene_patient_ids:  # Only keep if also present in gene dataset
        patient_feature_dict[row['TCGA ID']] = {
            'Survival months': row['Survival months'],
            'censored': row['censored'],
            'grade': row['Grade']
        }

# Step 4: Path to your patches folder
patches_folder_path = '../data/TCGA_GBMLGG/patches/'  # Replace with your real path

# Step 5: Match patches to patients
patches_info = []
for filename in os.listdir(patches_folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        patient_id = filename[:12]  # Extract TCGA ID from patch filename
        if patient_id in patient_feature_dict:
            info = patient_feature_dict[patient_id]
            patches_info.append({
                'patch_filename': filename,
                'TCGA_ID': patient_id,
                'Survival months': info['Survival months'],
                'censored': info['censored'],
                'grade': info['grade']
            })

# Step 6: Save the full patch-label linking
patches_df = pd.DataFrame(patches_info)
# patches_df.to_csv('../data/TCGA_GBMLGG/FinalDataset/patches_with_labels.csv', index=False)
patches_df.to_csv('../data/TCGA_KIRC/patches_with_labels.csv', index=False)

print(patches_df.head())
print(f"Total patches linked with patient labels and gene dataset: {len(patches_df)}")
