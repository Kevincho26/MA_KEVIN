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

# %%
# Load dataset prepared for unsupervised learning
from src.data.loaders import load_unsupervised_base_dataset
from src.features.engineering import (
    bin_numeric_column_to_indicators,
    expand_mapped_indicator_columns,
    one_hot_encode_columns,
)
from src.utils.paths import PROCESSED_DIR

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

df = load_unsupervised_base_dataset().copy()

# %%
# Binarize product profile variation columns
variation_cols = [
    "_PP_Claim_Variation",
    "_Provider_Benefit_Variation",
    "_Customer_Benefit_Variation",
]

# %%
variation_value_map = {0: "CV", 1: "AV", 2: "PV"}

for col in variation_cols:
    df = expand_mapped_indicator_columns(
        df,
        source_column=col,
        value_map=variation_value_map,
        labels=["CV", "AV", "PV"],
    )

# %%
# Discretize continuous technical variables into binary bins
continuous_cols = [
    "_δCV",
    "_δAV",
    "_δPV",
    "_δND",
    "_share_RSE_external",
    "_share_RSE_internal",
]

# %%
for col in continuous_cols:
    df = bin_numeric_column_to_indicators(
        df,
        source_column=col,
        bins=[float("-inf"), 0.33, 0.66, float("inf")],
        labels=["low", "med", "high"],
        prefix=col,
        right=False,
    )

# %%
# Convert generation to binned categories
df = bin_numeric_column_to_indicators(
    df,
    source_column="_Generation",
    bins=[0, 5, 10, 20],
    labels=["early", "mid", "late"],
    prefix="Generation",
)

# %%
# One-hot encode remaining contextual ordinal columns
contextual_cols = [col for col in df.columns if not col.startswith("_")]

# %%
df = one_hot_encode_columns(df, contextual_cols)

# %%
# Final check: ensure all variables are binary (0/1)
non_binary_cols = [col for col in df.columns if not set(df[col].dropna().unique()).issubset({0, 1})]
if non_binary_cols:
    print("Warning: The following columns are not strictly binary:", non_binary_cols)

# %%
# Export binarized dataset
output_path = PROCESSED_DIR / "association_rules_dataset.csv"
df.to_csv(output_path, index=False)

# %%
print("\n----- Apriori Dataset Summary -----")
print("Shape:", df.shape)
print("Binary columns:", df.columns.tolist()[:10], "...")
print("Dataset saved to:", output_path)
