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

input_path = PROCESSED_DIR / "base_dataset.csv"

# === 1. Load Merged Clean Dataset ===
df = pd.read_csv(input_path).copy()
print(f"Loaded base dataset from: {input_path}")

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
output_path = PROCESSED_DIR / "preprocessed_unsupervised.csv"
df.to_csv(output_path, index=False)
print(f"Dataset successfully saved to: {output_path}")

# === 6. Summary ===
print("\n----- Unsupervised Preprocessing Complete -----")
print(f"Final shape: {df.shape}")
print("Variables:", df.columns.tolist())
