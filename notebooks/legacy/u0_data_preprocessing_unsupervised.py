"""
Unsupervised Feature Refinement Module
--------------------------------------
This script refines the unsupervised dataset generated from the base pipeline by:
- Loading the unsupervised-ready dataset
- Dropping redundant or complementary variables
- Scaling selected numerical columns
- Exporting the refined dataset for downstream unsupervised models
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "notebooks").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not find repository root.")


REPO_ROOT = find_repo_root()
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

input_path = PROCESSED_DIR / "preprocessed_unsupervised.csv"

# Load dataset prepared for unsupervised learning
df = pd.read_csv(input_path).copy()
print(f"Loaded unsupervised base dataset from: {input_path}")

# Drop redundant or complementary variables
columns_to_remove = ["_δCV", "_δND", "_share_RSE_internal"]
df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)

# Standardize relevant numerical columns for unsupervised models
scaled_columns = ["_δAV", "_δPV", "_share_RSE_external"]
scaled_columns = [col for col in scaled_columns if col in df.columns]

if scaled_columns:
    scaler = StandardScaler()
    df[scaled_columns] = scaler.fit_transform(df[scaled_columns])

# Save refined dataset for unsupervised learning models
output_path = PROCESSED_DIR / "processed_unsupervised.csv"
df.to_csv(output_path, index=False)

# Final confirmation
print("\n----- Dataset Summary (Unsupervised Refinement) -----")
print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Data types:\n", df.dtypes.value_counts())
print("\nDataset successfully saved to:", output_path)
