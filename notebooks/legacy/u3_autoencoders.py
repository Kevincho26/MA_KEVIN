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
# # Autoencoder

# %%
# Autoencoder for Unsupervised Pattern Detection
# -----------------------------------------------
# This script uses an autoencoder to explore reconstruction errors for
# potential pattern discovery or anomaly detection in the dataset.

# === 1. Imports ===
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "notebooks").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not find repository root.")


REPO_ROOT = find_repo_root()
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# === 2. Load Preprocessed Unsupervised Dataset ===
csv_path = PROCESSED_DIR / "unsupervised_modeling_dataset.csv"
df = pd.read_csv(csv_path)

# === 2.1 Load full product info with Gen_ID ===
original_data_path = PROCESSED_DIR / "base_dataset.csv"
df_original = pd.read_csv(original_data_path)

# === 3. Standardize Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# === 4. Train/Test Split (for reconstruction validation only) ===
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# === 5. Define Autoencoder Architecture ===
input_dim = X_train.shape[1]
encoding_dim = 8  # Compression layer size

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation="relu")(input_layer)
decoded = Dense(input_dim, activation="linear")(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# === 6. Train Autoencoder ===
autoencoder.fit(
    X_train,
    X_train,
    epochs=100,
    batch_size=8,
    shuffle=True,
    validation_data=(X_test, X_test),
    verbose=0,
)

# === 7. Compute Reconstruction Errors ===
X_pred = autoencoder.predict(X_scaled)
reconstruction_error = np.mean(np.square(X_scaled - X_pred), axis=1)

# === 8. Analyze and Visualize ===
df_error = pd.DataFrame({"Reconstruction_Error": reconstruction_error})

plt.figure(figsize=(8, 5))
sns.histplot(df_error["Reconstruction_Error"], bins=30, kde=True, color="steelblue")
plt.title("Distribution of Reconstruction Errors")
plt.xlabel("Reconstruction Error")
plt.ylabel("Count")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# === 9. Identify High-Error Samples (Potential Outliers) ===
threshold = np.percentile(reconstruction_error, 95)
df_error["Outlier"] = df_error["Reconstruction_Error"] > threshold

# === 10. Match with _Gen_ID from full dataset ===
df_error["_Gen_ID"] = (
    df_original["_Gen_ID"] if "_Gen_ID" in df_original.columns else df_original.index
)

top_outliers = df_error.sort_values(by="Reconstruction_Error", ascending=False).head(10)

print("\nTop samples with highest reconstruction error:")
print(top_outliers[["Reconstruction_Error", "_Gen_ID", "Outlier"]])
print(f"\nSuggested anomaly threshold (95th percentile): {threshold:.4f}")

# === 11. Visualize Top Outliers ===
plt.figure(figsize=(10, 6))
sns.barplot(data=top_outliers, y="_Gen_ID", x="Reconstruction_Error", palette="Blues_d")
plt.title("Top 10 Products by Reconstruction Error")
plt.xlabel("Reconstruction Error")
plt.ylabel("Product (_Gen_ID)")
plt.grid(True, axis="x", linestyle="--", alpha=0.5)
for index, row in enumerate(top_outliers.itertuples()):
    plt.text(
        row.Reconstruction_Error + 0.01,
        index,
        f"{row.Reconstruction_Error:.3f}",
        va="center",
        ha="left",
    )
plt.tight_layout()
plt.show()

# %%
# NOTE: Sobre la aleatoriedad en autoencoders
# --------------------------------------------
# Este modelo incluye aleatoriedad inherente debido a la inicialización de pesos,
# el orden aleatorio del entrenamiento y operaciones internas del framework TensorFlow.
# Aunque es posible fijar seeds para obtener resultados más consistentes, no se ha hecho
# en esta implementación porque el objetivo de la tesis es presentar una única versión final
# del modelo y no comparar múltiples ejecuciones. Los resultados obtenidos representan un
# comportamiento válido y esperado del autoencoder bajo condiciones normales de entrenamiento.
