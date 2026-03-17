import os
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ensure consistent working directory when run as script (not notebook)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define the paths to the datasets
sge_path = Path("../data/raw/250119_SGE.csv")  # SGE factors dataset
context_path = Path("../data/raw/250123_Contextual.csv")  # Contextual factors dataset

# Load both datasets
try:
    sge_df = pd.read_csv(sge_path)
    contextual_df = pd.read_csv(context_path)
    print("Both datasets loaded successfully!")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# Rename the ID column to join the datasets
contextual_df.rename(columns={"_Gen_ID_Indirekt": "_Gen_ID"}, inplace=True)

# Columns that are not needed or irrelevant
columns_to_drop = [
    "_success_quant",
    "Unnamed: 0",
    "_Variant",
    "_Go_to_market",
    "_#SubSys",
    "_Case_Study",
]  # Add or Remove columns if needed
columns_to_drop2 = ["_Factor_Level", "Unnamed: 4"]

# Drop the irrelevant columns
sge_df = sge_df.drop(columns=columns_to_drop, axis=1)
contextual_df = contextual_df.drop(columns=columns_to_drop2, axis=1)

# Ensure '_Factor_Expression' is of type string and fill NaN with an empty string
contextual_df["_Factor_Expression"] = contextual_df["_Factor_Expression"].astype(str).fillna("")

# Pivot the table ensuring multiple expressions for the same '_Gen_ID' are concatenated correctly
contextual_pivot = contextual_df.pivot_table(
    index="_Gen_ID",
    columns=["_Factor_Factor", "_Factor_Dimension"],
    values="_Factor_Expression",
    aggfunc=lambda x: " | ".join(map(str, x)),  # Convert each element to string before joining
)

# Flatten the multi-level column index
contextual_pivot.columns = [
    f"{factor}_{dimension}" for factor, dimension in contextual_pivot.columns
]

# Reset the index to prepare for merging
contextual_pivot.reset_index(inplace=True)

# Merge the contextual factors dataset with the main dataset
merged_df = sge_df.merge(contextual_pivot, on="_Gen_ID", how="left")

merged_df = merged_df.drop(columns=["_Gen_ID"])

# Check the structure of the final merged table
print(merged_df.head())

missing_values = merged_df.isnull().sum()
missing_values = missing_values[missing_values > 0]
print(missing_values)

# Identify contextual factor columns (those that are not in the original SGE dataset)
contextual_columns = merged_df.columns.difference(sge_df.columns)

# Fill missing values only in contextual factor columns
merged_df[contextual_columns] = merged_df[contextual_columns].fillna(0)

# Drop rows with missing values in all columns
merged_df = merged_df.dropna()

# Display the number of remaining rows to confirm
print(f"Number of rows after dropping rows with missing values: {len(merged_df)}")

# get the number of missing data points per column
missing_values_count = merged_df.isnull().sum()

# look at the # of missing points in the first ten columns (SGE factors)
missing_values_count[0:11]

# Section 1: Target Variable and Variation Factors Encoding

# Step 1: Convert target variable (_Success_qual) to binary
# Mapping: "successful" -> 1, "indifferent" and "unsuccessful" -> 0
merged_df["_Success_qual"] = merged_df["_Success_qual"].apply(
    lambda x: 1 if x == "successful" else 0
)

# Step 2: Ordinal Encoding for variation factors
# Hierarchy mapping: PV=2, AV=1, CV=0
variation_mapping = {"PV": 2, "AV": 1, "CV": 0}
variation_columns = [
    "_PP_Claim_Variation",
    "_Provider_Benefit_Variation",
    "_Customer_Benefit_Variation",
]
for col in variation_columns:
    merged_df[col] = merged_df[col].map(variation_mapping)

# Step 3: Ensure that `_Generation` remains numerical
merged_df["_Generation"] = pd.to_numeric(merged_df["_Generation"], errors="coerce")

# Section 2: Encoding Contextual Factors

# Step 1: Ordinal Encoding for contextual factors
# Identify contextual factor columns dynamically SGE factors
excluded_cols = [
    "_Success_qual",
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

# Fill missing values in contextual factor columns with "0"
merged_df[contextual_columns] = merged_df[contextual_columns].fillna("0")
# Convert all values to string to avoid mapping errors
merged_df[contextual_columns] = merged_df[contextual_columns].astype(str)

# Define the mapping for contextual factors: "0" -> 0, "low" -> 1, "high" -> 2
contextual_factor_mapping = {"0": 0, "low": 1, "high": 2}
for col in contextual_columns:
    merged_df[col] = merged_df[col].map(contextual_factor_mapping).fillna(0).astype(int)


# Section 3: Processing Percentage Columns and Standardization

# Step 1: Identify percentage columns
percentage_columns = [
    "_δCV",
    "_δAV",
    "_δPV",
    "_δND",
    "_share_RSE_external",
    "_share_RSE_internal",
]
# Verify that these columns exist in the dataset
percentage_columns = [col for col in percentage_columns if col in merged_df.columns]

# Step 2: Convert percentage strings to decimal values
for col in percentage_columns:
    merged_df[col] = merged_df[col].astype(str).str.rstrip("%").astype(float) / 100.0

# Step 3: Remove redundant complementary variables
columns_to_remove = ["_δCV", "_δND", "_share_RSE_internal"]
columns_to_remove = [
    col for col in columns_to_remove if col in merged_df.columns
]  # Ensure columns exist
merged_df.drop(columns=columns_to_remove, axis=1, inplace=True)

# Step 4: Define numerical columns to scale (for logistic regression)
scaled_columns = ["_δAV", "_δPV", "_share_RSE_external"]
scaled_columns = [col for col in scaled_columns if col in merged_df.columns]

# Step 5: Apply standardization using StandardScaler
scaler = StandardScaler()
merged_df[scaled_columns] = scaler.fit_transform(merged_df[scaled_columns])

# Define the output file path
output_path = r"C:\Master Thesis Repositories\MAChoque\data\processed\processed_dataset.csv"

# Save the dataset as a CSV file
merged_df.to_csv(output_path, index=False)

# Confirm the operation
print(f"Dataset successfully saved to: {output_path}")

# Step 1: Separate the target variable (_Success_qual) from the features
if "_Success_qual" in merged_df.columns:
    X = merged_df.drop("_Success_qual", axis=1)  # Features (all columns except the target)
    y = merged_df["_Success_qual"]  # Target variable (binary)
else:
    raise ValueError("The target variable '_Success_qual' is missing from the dataset.")

# Step 2: Confirm the separation
print("Features and target variable separated successfully.")

# Step 3: Display shapes to confirm the structure
print("Features shape (X):", X.shape)
print("Target shape (y):", y.shape)

# Step 4: Ensure that the dataset does not contain missing values before modeling
if X.isnull().sum().sum() > 0:
    print(
        "Warning: There are missing values in the features dataset (X). Consider handling them before modeling."
    )
