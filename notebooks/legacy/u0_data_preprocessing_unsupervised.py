import os
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Ensure consistent working directory when run as script (not notebook)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load Datasets
sge_path = Path("../data/raw/250119_SGE.csv")
context_path = Path("../data/raw/250123_Contextual.csv")
sge_df = pd.read_csv(sge_path)
contextual_df = pd.read_csv(context_path)

# Clean contextual dataset: drop irrelevant columns and standardize IDs
contextual_df.rename(columns={"_Gen_ID_Indirekt": "_Gen_ID"}, inplace=True)
columns_to_drop = [
    "_success_quant",
    "Unnamed: 0",
    "_Variant",
    "_Go_to_market",
    "_#SubSys",
    "_Case_Study",
    "_Success_qual",  # remove target variable here
]
columns_to_drop2 = ["_Factor_Level", "Unnamed: 4"]
sge_df = sge_df.drop(columns=columns_to_drop, errors="ignore")
contextual_df = contextual_df.drop(columns=columns_to_drop2, errors="ignore")

# Transform contextual data: pivot expression values into columns
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

# Merge SGE and contextual datasets
merged_df = sge_df.merge(contextual_pivot, on="_Gen_ID", how="left")
merged_df.drop(columns=["_Gen_ID"], inplace=True)

# Fill missing contextual values and remove rows with any remaining NaNs
contextual_columns = merged_df.columns.difference(sge_df.columns)
merged_df[contextual_columns] = merged_df[contextual_columns].fillna(0)
merged_df = merged_df.dropna()

# Encode technical variation factors
variation_mapping = {"PV": 2, "AV": 1, "CV": 0}
for col in [
    "_PP_Claim_Variation",
    "_Provider_Benefit_Variation",
    "_Customer_Benefit_Variation",
]:
    merged_df[col] = merged_df[col].map(variation_mapping)

# Ensure _Generation is numeric
merged_df["_Generation"] = pd.to_numeric(merged_df["_Generation"], errors="coerce")

# Encode contextual factors ordinally
excluded_cols = [
    "_Generation",
    "_PP_Claim_Variation",
    "_Provider_Benefit_Variation",
    "_Customer_Benefit_Variation",
    "_#SubSys",
    "_δCV",
    "_δAV",
    "_δPV",
    "_δND",
    "_share_RSE_external",
    "_share_RSE_internal",
]
contextual_columns = [col for col in merged_df.columns if col not in excluded_cols]
merged_df[contextual_columns] = merged_df[contextual_columns].fillna("0").astype(str)

# Optional: preserve a copy for EDA/visuals with categorical labels (EDA only, not for modeling)
contextual_copy = merged_df[contextual_columns].copy()

contextual_factor_mapping = {"0": 0, "low": 1, "high": 2}
for col in contextual_columns:
    merged_df[col] = merged_df[col].map(contextual_factor_mapping).fillna(0).astype(int)

# Process percentage columns to float format (decimal)
percentage_columns = [
    "_δCV",
    "_δAV",
    "_δPV",
    "_δND",
    "_share_RSE_external",
    "_share_RSE_internal",
]
percentage_columns = [col for col in percentage_columns if col in merged_df.columns]
for col in percentage_columns:
    merged_df[col] = merged_df[col].astype(str).str.rstrip("%").astype(float) / 100.0

# Drop redundant or complementary variables
columns_to_remove = ["_δCV", "_δND", "_share_RSE_internal"]
merged_df.drop(columns=[col for col in columns_to_remove if col in merged_df.columns], inplace=True)

# Standardize relevant numerical columns for Unsupervised Models
scaled_columns = ["_δAV", "_δPV", "_share_RSE_external"]
scaler = StandardScaler()
merged_df[scaled_columns] = scaler.fit_transform(merged_df[scaled_columns])

# Save processed dataset for unsupervised learning models
output_path = (
    r"C:\\Master Thesis Repositories\\MAChoque\\data\\processed\\processed_unsupervised.csv"
)
merged_df.to_csv(output_path, index=False)

# Final confirmation
print("\n----- Dataset Summary (Unsupervised) -----")
print("Shape:", merged_df.shape)
print("Missing values:", merged_df.isnull().sum().sum())
print("Data types:\n", merged_df.dtypes.value_counts())
print("\nDataset successfully saved to:", output_path)
