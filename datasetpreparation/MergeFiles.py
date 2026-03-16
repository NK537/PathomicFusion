import pandas as pd

# Load the two datasets
all_dataset = pd.read_csv('data/TCGA_GBMLGG/all_dataset.csv')
grade_data = pd.read_csv('data/TCGA_GBMLGG/grade_data.csv')

all_dataset.shape
grade_data.shape

print("all_dataset dataset shape:", all_dataset.shape)
print("grade_data dataset shape:", grade_data.shape)

# Identify the correct column for Sample ID
# Adjust if necessary based on your actual file structure
all_dataset_sample_col = 'TCGA ID' if 'TCGA ID' in all_dataset.columns else all_dataset.columns[0]
grade_data_sample_col = 'TCGA ID' if 'TCGA ID' in grade_data.columns else grade_data.columns[0]

# Merge the datasets on the Sample ID
merged_df = pd.merge(
    all_dataset,
    grade_data,
    left_on=all_dataset_sample_col,
    right_on=grade_data_sample_col,
    how='left',  # Use 'inner' if you want only matched records
    suffixes=('', '_grade')
)

# # Save the merged result (optional)
# merged_df.to_csv('merged_all_dataset_and_grade_data.csv', index=False)

# # Display basic info
# print("Merged dataset shape:", merged_df.shape)
# print("Merged dataset columns:", merged_df.columns.tolist())

# # If you want to preview
# print(merged_df.head())
