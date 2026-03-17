import os
from pathlib import Path

import pandas as pd

# Ensure consistent working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load Datasets
sge_path = Path("../data/raw/250119_SGE.csv")
context_path = Path("../data/raw/250123_Contextual.csv")
sge_df = pd.read_csv(sge_path)
contextual_df = pd.read_csv(context_path)

# Clean contextual dataset
contextual_df.rename(columns={"_Gen_ID_Indirekt": "_Gen_ID"}, inplace=True)
columns_to_drop = [
    "_success_quant",
    "Unnamed: 0",
    "_Variant",
    "_Go_to_market",
    "_#SubSys",
    "_Case_Study",
]
columns_to_drop2 = ["_Factor_Level", "Unnamed: 4"]
sge_df = sge_df.drop(columns=columns_to_drop, errors="ignore")
contextual_df = contextual_df.drop(columns=columns_to_drop2, errors="ignore")

# Pivot contextual data
contextual_df["_Factor_Expression"] = contextual_df["_Factor_Expression"].astype(str).fillna("")
contextual_pivot = contextual_df.pivot_table(
    index="_Gen_ID",
    columns=["_Factor_Factor", "_Factor_Dimension"],
    values="_Factor_Expression",
    aggfunc=lambda x: " | ".join(map(str, x)),
)
contextual_pivot.columns = [
    f"{factor}_{dimension}" for factor, dimension in contextual_pivot.columns
]
contextual_pivot.reset_index(inplace=True)

# Merge datasets
merged_df = sge_df.merge(contextual_pivot, on="_Gen_ID", how="left")
merged_df.drop(columns=["_Gen_ID"], inplace=True)

# Fill contextual missing values and drop remaining NaNs
contextual_columns = merged_df.columns.difference(sge_df.columns)
merged_df[contextual_columns] = merged_df[contextual_columns].fillna("0")
merged_df = merged_df.dropna()

# Drop target variable
if "_Success_qual" in merged_df.columns:
    merged_df.drop(columns=["_Success_qual"], inplace=True)

# Binarize variation type columns
variation_mapping = {"PV": "PV", "AV": "AV", "CV": "CV"}
for col in [
    "_PP_Claim_Variation",
    "_Provider_Benefit_Variation",
    "_Customer_Benefit_Variation",
]:
    for vtype in ["PV", "AV", "CV"]:
        merged_df[f"{col}_{vtype}"] = merged_df[col].apply(lambda x: 1 if x == vtype else 0)
    merged_df.drop(columns=[col], inplace=True)

# Discretize percentage/ratio columns into categories
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
        merged_df[f"{col}_low"] = (merged_df[col] < 0.33).astype(int)
        merged_df[f"{col}_med"] = ((merged_df[col] >= 0.33) & (merged_df[col] < 0.66)).astype(int)
        merged_df[f"{col}_high"] = (merged_df[col] >= 0.66).astype(int)
        merged_df.drop(columns=[col], inplace=True)

# Convert _Generation to category (binned)
if "_Generation" in merged_df.columns:
    merged_df["_Generation"] = pd.to_numeric(merged_df["_Generation"], errors="coerce")
    bins = [0, 5, 10, 20]
    labels = ["early", "mid", "late"]
    merged_df["_Generation_cat"] = pd.cut(merged_df["_Generation"], bins=bins, labels=labels)
    for label in labels:
        merged_df[f"Generation_{label}"] = (merged_df["_Generation_cat"] == label).astype(int)
    merged_df.drop(columns=["_Generation", "_Generation_cat"], inplace=True)

# Ordinal encoding to binarized dummy variables for contextual factors
excluded_cols = merged_df.columns[
    merged_df.columns.str.startswith("_")
    & ~merged_df.columns.str.contains("_PV|_AV|_CV|_low|_med|_high|Generation_")
]
contextual_columns = [col for col in merged_df.columns if col not in excluded_cols]

for col in contextual_columns:
    merged_df[col] = merged_df[col].astype(str)
    dummies = pd.get_dummies(merged_df[col], prefix=col)
    merged_df = pd.concat([merged_df.drop(columns=[col]), dummies], axis=1)

# Final check: ensure all variables are binary (0/1)
non_binary_cols = [
    col for col in merged_df.columns if not set(merged_df[col].unique()).issubset({0, 1})
]
if non_binary_cols:
    print("Warning: The following columns are not strictly binary:", non_binary_cols)

# Export binarized dataset
output_path = (
    r"C:\\Master Thesis Repositories\\MAChoque\\data\\processed\\processed_unsupervised_apriori.csv"
)
merged_df.to_csv(output_path, index=False)

print("\n----- Apriori Dataset Summary -----")
print("Shape:", merged_df.shape)
print("Binary columns:", merged_df.columns.tolist()[:10], "...")  # display first few
print("Dataset saved to:", output_path)
