import os
from collections import defaultdict
import pandas as pd

# Path to patches folder (/)
patches_folder_path = 'data/TCGA_GBMLGG/patches'  # path of image

# Initialize a dictionary to group patches by Patient TCGA ID
patient_patch_dict = defaultdict(list)

# Walk through the folder and extract TCGA IDs
for filename in os.listdir(patches_folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):  # Assuming patches are .png or .jpg
        patient_id = filename[:12]  # First 12 characters for TCGA ID
        patient_patch_dict[patient_id].append(filename)

# # Summary of how many patches per patient
# for patient_id, patches in patient_patch_dict.items():
#     print(f"Patient ID: {patient_id} — {len(patches)} patch(es)")


# Step 1: Load merged dataset
merged_df = pd.read_csv('data/TCGA_GBMLGG/merged_all_dataset_and_grade_data.csv')  # Replace with your actual path

# Step 2: Inspect column names
print(merged_df.columns.tolist())

# Step 3: Extract necessary columns
# Assume columns are named like ['TCGA ID', 'Survival_Time', 'Event', 'Grade']
patient_feature_df = merged_df[['TCGA ID', 'Survival months', 'censored', 'Grade']]

# Step 4: Standardize TCGA IDs (make sure no extra spaces, all upper case)
patient_feature_df['TCGA ID'] = patient_feature_df['TCGA ID'].str.strip().str.upper()

# Step 5: View sample data
print(patient_feature_df.head())

# Step 6: Save for later use (optional)
patient_feature_df.to_csv('data/TCGA_GBMLGG/patient_feature_table.csv', index=False)