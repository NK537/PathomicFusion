import pandas as pd

# Load the two TXT files (update the file paths as necessary)
mrna_expr1 = pd.read_csv('data/TCGA_GBMLGG/mRNA_Expression_z-Scores_RNA_Seq_RSEM.txt', sep='\t', index_col=0)
mrna_expr2 = pd.read_csv('data/TCGA_GBMLGG/mRNA_Expression_Zscores_RSEM.txt', sep='\t', index_col=0)

# Display first few rows to understand the structure
print("First mRNA Expression File Index and Columns:", mrna_expr1.index, mrna_expr1.columns)
print("Second mRNA Expression File Index and Columns:", mrna_expr2.index, mrna_expr2.columns)

# Transpose both datasets to use rows as columns
mrna_expr1_transposed = mrna_expr1.T
mrna_expr2_transposed = mrna_expr2.T



# Display first few rows to understand the structure
print("First mRNA Expression Transpose:", mrna_expr1_transposed)
print("Second mRNA Expression Transpose",mrna_expr1_transposed)



# Concatenate the two transposed datasets vertically
concatenated_mrna_df = pd.concat([mrna_expr1_transposed, mrna_expr2_transposed], ignore_index=False)

# Save the concatenated result to CSV
concatenated_mrna_df.to_csv('data/TCGA_GBMLGG/concatenated_mrna_expression_transposed.csv')

print(f"Concatenated {concatenated_mrna_df.shape[0]} samples and {concatenated_mrna_df.shape[1]} genes successfully!")