"""
    NAME: NGU YU LING
    MATRIC NO: A23CS0149
"""

#! Part 1.1: Data Preprocessing

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


#! Part 1.2: Data Exploration - Target Classes & Label Distribution

# data loading
data = pd.read_csv('data.csv')

# data shape
print(f"\nDataset shape: {data.shape}")
print(f"  - Number of samples (rows): {data.shape[0]}")
print(f"  - Number of features + target (columns): {data.shape[1]}")
print(f"  - Number of genes (features): {data.shape[1] - 2}")

# class distribution
target_column = 'cancer'
class_counts = data[target_column].value_counts().sort_index()
class_percentages = (class_counts / len(data) * 100).round(2)
print(f"\nClass distribution:")
for class_label in sorted(data[target_column].unique()):
    count = class_counts[class_label]
    percentage = class_percentages[class_label]
    print(f"  {class_label}: {count} samples ({percentage}%)")

# bar chart with counts and percentages
import matplotlib.pyplot as plt
colors = [ '#3895D3', '#072F5F']
bars = plt.bar(class_counts.index, class_counts.values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
plt.style.use('dark_background')
plt.title('Dataset Class Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Cancer Type', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    percentage = (height / len(data) * 100)
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)} ({percentage:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=12)
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')