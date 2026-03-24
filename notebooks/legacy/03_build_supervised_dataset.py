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
Supervised Data Preprocessing Module
-------------------------------------
This script prepares the cleaned dataset from `01_build_base_dataset`
for supervised learning models by:
- Dropping non-predictive and irrelevant columns
- Encoding categorical product profile variables with ordinal meaning
- Encoding contextual variables with ordinal values (high > low > 0)
- Binarizing the target variable `_Success_qual`
- Exporting the result for downstream supervised models
"""

# %%
import pandas as pd

# %%
# === 1. Load Merged Clean Dataset ===
from src.data.loaders import load_base_dataset
from src.features.preprocessing import drop_columns_if_present
from src.utils.paths import PROCESSED_DIR

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

df = load_base_dataset().copy()

# %%
# === 2. Drop non-informative or irrelevant columns ===
columns_to_drop = ["_Gen_ID", "Context_Others", "has_context"]
df = drop_columns_if_present(df, columns_to_drop)

# %%
# === 3. Ordinal Encode Product Profile Variation Columns ===
variation_map = {"CV": 0, "AV": 1, "PV": 2}
variation_cols = [
    "_PP_Claim_Variation",
    "_Customer_Benefit_Variation",
    "_Provider_Benefit_Variation",
]

# %%
for col in variation_cols:
    df[col] = df[col].map(variation_map).astype(int)

# %%
# === 4. Ordinal Encode Contextual Variables (high > low > 0) ===
contextual_cols = df.columns.difference(
    variation_cols
    + ["_Generation", "_Success_qual"]
    + [c for c in df.columns if c.startswith("_δ")]
    + [c for c in df.columns if "RSE" in c]
)

# %%
contextual_map = {"0": 0, "low": 1, "high": 2}
for col in contextual_cols:
    df[col] = df[col].map(contextual_map).astype(int)

# %%
# === 5. Binarize the target variable ===
df["_Success_qual"] = df["_Success_qual"].apply(lambda x: 1 if x == "successful" else 0)

# %%
# === 6. Export Supervised-Ready Dataset ===
output_path = PROCESSED_DIR / "supervised_modeling_dataset.csv"
df.to_csv(output_path, index=False)
print(f"Dataset successfully saved to: {output_path}")

# %%
# === 7. Optional Summary ===
print("\n----- Supervised Dataset Summary -----")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Target distribution:")
print(df["_Success_qual"].value_counts(normalize=True).rename("proportion"))
