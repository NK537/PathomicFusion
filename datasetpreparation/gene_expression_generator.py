import pandas as pd

# Step 1: Load the uploaded file
gene_df = pd.read_csv('data/TCGA_GBMLGG/gene data/concatenated_mrna_expression_transposed.csv')

# Step 2: Extract TCGA ID from Sample ID
gene_df['TCGA ID'] = gene_df['SAMPLE_ID'].apply(lambda x: x[:12])

# Step 3: Set TCGA ID as index
gene_df = gene_df.set_index('TCGA ID')

# Step 4: Drop the Sample ID column
# Drop Unnamed: 0 and SAMPLE_ID columns
gene_df = gene_df.drop(columns=['Unnamed: 0', 'SAMPLE_ID'])

# Fill missing gene expressions with 0
gene_df = gene_df.fillna(0)

# Drop duplicate TCGA IDs (keep first occurrence)
gene_df = gene_df[~gene_df.index.duplicated(keep='first')]

# Step 5: Save clean gene expression matrix
gene_df.to_csv('data/TCGA_GBMLGG/gene data/clean_gene_expression.csv')

gene_df.head()  # Show a sample
