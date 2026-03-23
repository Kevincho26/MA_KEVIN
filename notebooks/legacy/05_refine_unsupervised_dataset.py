# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: jupytext,-kernelspec,-language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %%
"""
Unsupervised Feature Refinement Module
--------------------------------------
This script refines the unsupervised dataset generated from the base pipeline by:
- Loading the unsupervised-ready dataset
- Dropping redundant or complementary variables
- Scaling selected numerical columns
- Exporting the refined dataset for downstream unsupervised models
"""

# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler

# %%
from src.utils.paths import PROCESSED_DIR

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# %%
input_path = PROCESSED_DIR / "unsupervised_base_dataset.csv"

# %%
# Load dataset prepared for unsupervised learning
df = pd.read_csv(input_path).copy()
print(f"Loaded unsupervised base dataset from: {input_path}")

# %%
# Drop redundant or complementary variables
columns_to_remove = ["_δCV", "_δND", "_share_RSE_internal"]
df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)

# %%
# Standardize relevant numerical columns for unsupervised models
scaled_columns = ["_δAV", "_δPV", "_share_RSE_external"]
scaled_columns = [col for col in scaled_columns if col in df.columns]

# %%
if scaled_columns:
    scaler = StandardScaler()
    df[scaled_columns] = scaler.fit_transform(df[scaled_columns])

# %%
# Save refined dataset for unsupervised learning models
output_path = PROCESSED_DIR / "unsupervised_modeling_dataset.csv"
df.to_csv(output_path, index=False)

# %%
# Final confirmation
print("\n----- Dataset Summary (Unsupervised Refinement) -----")
print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Data types:\n", df.dtypes.value_counts())
print("\nDataset successfully saved to:", output_path)
