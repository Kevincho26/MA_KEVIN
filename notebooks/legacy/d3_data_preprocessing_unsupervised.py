"""
Unsupervised Data Preprocessing Module
----------------------------------------
This script prepares the cleaned dataset from `d0_data_preprocessing`
for unsupervised learning models by:
- Dropping target and identifiers
- Encoding categorical product profile variables with ordinal meaning
- Encoding contextual variables with ordinal values (high > low > 0)
- Exporting the result for downstream clustering and dimensionality reduction
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add the parent directory to path to allow import of preprocessing script
sys.path.append(str(Path(__file__).resolve().parents[1]))

# === 1. Load Merged Clean Dataset ===
from d0_data_preprocessing import merged_df as base_df

# Copy to avoid modifying original
df = base_df.copy()

# === 2. Drop target and irrelevant variables ===
drop_columns = ["_Success_qual", "_Gen_ID", "Context_Others", "has_context"]
df.drop(columns=drop_columns, inplace=True, errors="ignore")

# === 3. Encode Product Profile Variation (ordinal: CV < AV < PV) ===
variation_map = {"CV": 0, "AV": 1, "PV": 2}
profile_vars = [
    "_PP_Claim_Variation",
    "_Provider_Benefit_Variation",
    "_Customer_Benefit_Variation",
]
for col in profile_vars:
    if col in df.columns:
        df[col] = df[col].map(variation_map).astype(int)

# === 4. Encode Contextual Variables (ordinal: 0 < low < high) ===
contextual_vars = [
    "Competition",
    "Customer",
    "Legal",
    "Market",
    "Organization",
    "Politics",
    "Product_development",
    "Society",
    "Technology",
]
context_map = {"0": 0, "low": 1, "high": 2}
for col in contextual_vars:
    if col in df.columns:
        df[col] = df[col].astype(str).map(context_map).astype(int)

# === 5. Export dataset ===
output_path = (
    r"C:\\Master Thesis Repositories\\MAChoque\\data\\processed\\preprocessed_unsupervised.csv"
)
df.to_csv(output_path, index=False)

# === 6. Summary ===
print("\n----- Unsupervised Preprocessing Complete -----")
print(f"Final shape: {df.shape}")
print("Variables:", df.columns.tolist())
