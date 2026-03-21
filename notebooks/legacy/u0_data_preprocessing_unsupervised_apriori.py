"""
Apriori Feature Refinement Module
--------------------------------
This script refines the unsupervised dataset generated from the base pipeline by:
- Loading the unsupervised-ready dataset
- Binarizing product profile variation columns
- Discretizing continuous technical variables into low/med/high bins
- Binning generation into early/mid/late
- One-hot encoding contextual ordinal features
- Exporting a binary dataset for Apriori and association rules
"""

from pathlib import Path

import pandas as pd


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "notebooks").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not find repository root.")


REPO_ROOT = find_repo_root()
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

input_path = PROCESSED_DIR / "unsupervised_base_dataset.csv"

# Load dataset prepared for unsupervised learning
df = pd.read_csv(input_path).copy()
print(f"Loaded unsupervised base dataset from: {input_path}")

# Binarize product profile variation columns
variation_cols = [
    "_PP_Claim_Variation",
    "_Provider_Benefit_Variation",
    "_Customer_Benefit_Variation",
]

variation_value_map = {0: "CV", 1: "AV", 2: "PV"}

for col in variation_cols:
    if col in df.columns:
        mapped = df[col].map(variation_value_map)
        for label in ["CV", "AV", "PV"]:
            df[f"{col}_{label}"] = (mapped == label).astype(int)
        df.drop(columns=[col], inplace=True)

# Discretize continuous technical variables into binary bins
continuous_cols = [
    "_δCV",
    "_δAV",
    "_δPV",
    "_δND",
    "_share_RSE_external",
    "_share_RSE_internal",
]

for col in continuous_cols:
    if col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_low"] = (series < 0.33).astype(int)
        df[f"{col}_med"] = ((series >= 0.33) & (series < 0.66)).astype(int)
        df[f"{col}_high"] = (series >= 0.66).astype(int)
        df.drop(columns=[col], inplace=True)

# Convert generation to binned categories
if "_Generation" in df.columns:
    generation = pd.to_numeric(df["_Generation"], errors="coerce")
    bins = [0, 5, 10, 20]
    labels = ["early", "mid", "late"]
    generation_cat = pd.cut(generation, bins=bins, labels=labels)

    for label in labels:
        df[f"Generation_{label}"] = (generation_cat == label).astype(int)

    df.drop(columns=["_Generation"], inplace=True)

# One-hot encode remaining contextual ordinal columns
contextual_cols = [col for col in df.columns if not col.startswith("_")]

for col in contextual_cols:
    dummies = pd.get_dummies(df[col].astype(str), prefix=col)
    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

# Final check: ensure all variables are binary (0/1)
non_binary_cols = [col for col in df.columns if not set(df[col].dropna().unique()).issubset({0, 1})]
if non_binary_cols:
    print("Warning: The following columns are not strictly binary:", non_binary_cols)

# Export binarized dataset
output_path = PROCESSED_DIR / "processed_unsupervised_apriori.csv"
df.to_csv(output_path, index=False)

print("\n----- Apriori Dataset Summary -----")
print("Shape:", df.shape)
print("Binary columns:", df.columns.tolist()[:10], "...")
print("Dataset saved to:", output_path)
