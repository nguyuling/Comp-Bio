"""
    NAME: NGU YU LING
    MATRIC NO: A23CS0149
"""

#! Data Preprocessing

# data loading
import pandas as pd
actual_df = pd.read_csv('raw data/actual.csv')
train_df = pd.read_csv('raw data/data_set_ALL_AML_train.csv')
independent_df = pd.read_csv('raw data/data_set_ALL_AML_independent.csv')

# extract gene accession number (from train only, since it's acc no. are the same for independent)
gene_accessions = train_df['Gene Accession Number'].values

# extract gene expression values (train & independent)
train_cols = train_df.columns[2:]  # skip first two columns (gene desc, gene accession no.)
independent_cols = independent_df.columns[2:] 
train_expression_cols = [col for col in train_cols if 'call' not in col.lower()]
train_expression = train_df[train_expression_cols].copy()
independent_expression_cols = [col for col in independent_cols if 'call' not in col.lower()]
independent_expression = independent_df[independent_expression_cols].copy()

# combine train and independent data
combined_expression = pd.concat([train_expression, independent_expression], axis=1)

# set gene accession numbers as row index
combined_expression.index = gene_accessions

# transpose so rows = samples + gene expression values, columns = genes accession number
expression_transposed = combined_expression.T

# set patient IDs (int) as row index
patient_ids = [int(col) for col in expression_transposed.index]
expression_transposed.index = patient_ids
expression_transposed.index.name = 'patient'
actual_df['patient'] = actual_df['patient'].astype(int)
expression_transposed.index = expression_transposed.index.astype(int)

# merge label with features
expression_with_patient = expression_transposed.reset_index()
merged_data = pd.merge(expression_with_patient, actual_df, on='patient', how='left')

# sort by patient ID in numerical order
merged_data = merged_data.sort_values('patient').reset_index(drop=True)
merged_data.to_csv('data.csv', index=False)