# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: jupytext,-kernelspec,-language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # Data Preprocessing

# %%
import pandas as pd

from src.utils.paths import PROCESSED_DIR, RAW_DIR

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
sge_path = RAW_DIR / "250119_SGE.csv"
context_path = RAW_DIR / "250123_Contextual.csv"
sge_df = pd.read_csv(sge_path)
contextual_df = pd.read_csv(context_path)

# %%
# Remove irrelevant columns from both datasets
contextual_df.rename(columns={"_Gen_ID_Indirekt": "_Gen_ID"}, inplace=True)
columns_to_drop = [
    "_success_quant",
    "Unnamed: 0",
    "_Variant",
    "_Go_to_market",
    "_#SubSys",
    "_Case_Study",
]
columns_to_drop2 = ["_Factor_Dimension", "Unnamed: 4"]
sge_df = sge_df.drop(columns=columns_to_drop, errors="ignore")
contextual_df = contextual_df.drop(columns=columns_to_drop2, errors="ignore")

# %%
# Pivot the table so each Gen_ID has one row and each Factor one column with value high, low, or 0
contextual_pivot = contextual_df.pivot_table(
    index="_Gen_ID",
    columns="_Factor_Factor",
    values="_Factor_Expression",
    aggfunc=lambda x: x.iloc[0] if not x.empty else 0,
)

# Fill missing values with 0 (meaning no expression present for that factor)
contextual_pivot = contextual_pivot.fillna("0").reset_index()

# %%
# Merge with SGE dataset
merged_df = sge_df.merge(contextual_pivot, on="_Gen_ID", how="left")

# %%
# Identify contextual columns
contextual_columns = contextual_pivot.columns.difference(["_Gen_ID"])

# Define technical columns (excluding ID and success)
technical_columns = [
    "_Generation",
    "_PP_Claim_Variation",
    "_Provider_Benefit_Variation",
    "_Customer_Benefit_Variation",
    "_δCV",
    "_δAV",
    "_δPV",
    "_δND",
    "_share_RSE_external",
    "_share_RSE_internal",
]

# %%
# Drop rows with missing values in any technical variable
merged_df = merged_df.dropna(subset=technical_columns)

# %%
# ----- Create binary variable 'has_context' based on missing contextual information -----
# If all contextual variables are NaN, then the product has no context info
merged_df["has_context"] = merged_df[contextual_columns].notna().any(axis=1).astype(int)

# Now replace remaining NaNs (within valid context) with "0"
merged_df[contextual_columns] = merged_df[contextual_columns].fillna("0")

# %%
# Convert technical variation columns to categorical
variation_columns = [
    "_PP_Claim_Variation",
    "_Provider_Benefit_Variation",
    "_Customer_Benefit_Variation",
]
for col in variation_columns:
    merged_df[col] = merged_df[col].astype("category")

# %%
# Convert technical variation columns to categorical
variation_columns = [
    "_PP_Claim_Variation",
    "_Provider_Benefit_Variation",
    "_Customer_Benefit_Variation",
]
for col in variation_columns:
    merged_df[col] = merged_df[col].astype("category")

# %%
# Normalize _Success_qual
if "_Success_qual" in merged_df.columns:
    merged_df["_Success_qual"] = (
        merged_df["_Success_qual"]
        .apply(lambda x: "successful" if x == "successful" else "unsuccessful")
        .astype("category")
    )

# %%
# Normalize percentage columns
percentage_columns = [
    "_δCV",
    "_δAV",
    "_δPV",
    "_δND",
    "_share_RSE_external",
    "_share_RSE_internal",
]
for col in percentage_columns:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].astype(str).str.rstrip("%").astype(float) / 100.0

# %%
# Ensure _Generation is numeric
if "_Generation" in merged_df.columns:
    merged_df["_Generation"] = pd.to_numeric(merged_df["_Generation"], errors="coerce")

# %%
# Set threshold for low representation (e.g., < 10% of dataset size)
threshold = 0.1 * len(merged_df)

# Identify low coverage contextual columns
low_coverage_factors = [
    col for col in contextual_columns if (merged_df[col] != "0").sum() < threshold
]

# Create 'Context_Others' binary column: 1 if any of the low coverage factors is active
merged_df["Context_Others"] = (
    merged_df[low_coverage_factors]
    .apply(lambda row: any(val != "0" for val in row), axis=1)
    .astype(int)
)

# Drop the low coverage columns
merged_df.drop(columns=low_coverage_factors, inplace=True)

# Print which factors were grouped into 'Context_Others'
print("Contextual factors grouped into 'Context_Others':")
print(low_coverage_factors)

# %%
# Export clean dataset for EDA
output_path = PROCESSED_DIR / "base_dataset.csv"
merged_df.to_csv(output_path, index=False)
print(f"Dataset successfully saved to: {output_path}")

# %%
# Display dataset summary (relevant information only)
print("\n----- Final Dataset Summary -----")
print("Total rows:", merged_df.shape[0])
print("Total columns:", merged_df.shape[1])
print("Target class distribution:")
print(merged_df["_Success_qual"].value_counts(normalize=True).rename("proportion"))
print(
    "\nContextual columns:",
    [
        col
        for col in merged_df.columns
        if col not in variation_columns + percentage_columns + ["_Generation", "_Success_qual"]
    ],
)
