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
# ## 1. DBSCAN

# %%
# DBSCAN Clustering Model
# ------------------------
# This script applies DBSCAN to the PCA-reduced data to identify natural clusters and outliers.

# === 1. Imports ===

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.utils.paths import PROCESSED_DIR

# === 2. Load Preprocessed Unsupervised Dataset ===
csv_path = PROCESSED_DIR / "unsupervised_modeling_dataset.csv"
df = pd.read_csv(csv_path)

# === 3. Standardize ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# === 4. PCA for 2D Projection ===
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# === 5. Determine eps using Nearest Neighbors (k=5) ===
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(X_pca)
distances, indices = neigh.kneighbors(X_pca)
distance_desc = np.sort(distances[:, -1])

plt.figure(figsize=(8, 4))
plt.plot(distance_desc)
plt.title("k-Distance Graph (k=5) for DBSCAN")
plt.xlabel("Samples sorted by distance")
plt.ylabel("5-NN distance")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 6. Apply DBSCAN ===
# NOTE: Adjust eps based on elbow point from k-distance graph
db = DBSCAN(eps=0.8, min_samples=5)
labels = db.fit_predict(X_pca)

# === 7. Evaluate Clustering ===
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"\nDBSCAN found {n_clusters} clusters and {n_noise} noise points.")

if n_clusters > 1:
    silhouette = silhouette_score(X_pca, labels)
    print(f"Silhouette Score: {silhouette:.2f}")
else:
    print("Silhouette Score: Not defined (only one cluster)")

# === 8. Plot Results ===
df_plot = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_plot["Cluster"] = labels

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_plot, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=70, edgecolor="k", alpha=0.8
)
plt.title("DBSCAN Clustering on PCA-reduced Data")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# DBSCAN Clustering Model
# ------------------------
# This script applies DBSCAN to the PCA-reduced data to identify natural clusters and outliers.

# === 1. Imports ===
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# === 2. Load Preprocessed Unsupervised Dataset ===
csv_path = PROCESSED_DIR / "unsupervised_modeling_dataset.csv"
df = pd.read_csv(csv_path)

# === 3. Standardize ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# === 4. PCA for 2D Projection ===
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# === 5. Determine eps using Nearest Neighbors (k=5) ===
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(X_pca)
distances, indices = neigh.kneighbors(X_pca)
distance_desc = np.sort(distances[:, -1])

plt.figure(figsize=(8, 4))
plt.plot(distance_desc)
plt.title("k-Distance Graph (k=5) for DBSCAN")
plt.xlabel("Samples sorted by distance")
plt.ylabel("5-NN distance")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 6. Apply DBSCAN (Final Model: eps=0.8 based on best silhouette score) ===
db = DBSCAN(eps=0.8, min_samples=5)
labels = db.fit_predict(X_pca)

# === 7. Evaluate Clustering ===
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"\nDBSCAN found {n_clusters} clusters and {n_noise} noise points.")

if n_clusters > 1:
    silhouette = silhouette_score(X_pca, labels)
    print(f"Silhouette Score: {silhouette:.2f}")
else:
    print("Silhouette Score: Not defined (only one cluster)")

# === 8. Plot Results ===
df_plot = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_plot["Cluster"] = labels

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_plot, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=70, edgecolor="k", alpha=0.8
)
plt.title("DBSCAN Clustering on PCA-reduced Data")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 9. Show Cluster Distribution ===
cluster_counts = df_plot["Cluster"].value_counts().sort_index()
print("\nDistribution of Samples per Cluster:")
print(cluster_counts)

# %%
# DBSCAN Clustering Model
# ------------------------
# This script applies DBSCAN to the PCA-reduced data to identify natural clusters and outliers.

# === 1. Imports ===
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# === 2. Load Preprocessed Unsupervised Dataset ===
csv_path = PROCESSED_DIR / "unsupervised_modeling_dataset.csv"
df = pd.read_csv(csv_path)

# === 3. Standardize ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# === 4. PCA for 2D Projection ===
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# === 5. Determine eps using Nearest Neighbors (k=5) ===
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(X_pca)
distances, indices = neigh.kneighbors(X_pca)
distance_desc = np.sort(distances[:, -1])

plt.figure(figsize=(8, 4))
plt.plot(distance_desc)
plt.title("k-Distance Graph (k=5) for DBSCAN")
plt.xlabel("Samples sorted by distance")
plt.ylabel("5-NN distance")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 6. Apply DBSCAN (Final Model: eps=0.8 based on best silhouette score) ===
db = DBSCAN(eps=0.8, min_samples=5)
labels = db.fit_predict(X_pca)

# === 7. Evaluate Clustering ===
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"\nDBSCAN found {n_clusters} clusters and {n_noise} noise points.")

if n_clusters > 1:
    silhouette = silhouette_score(X_pca, labels)
    print(f"Silhouette Score: {silhouette:.2f}")
else:
    print("Silhouette Score: Not defined (only one cluster)")

# === 8. Plot Results ===
df_plot = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_plot["Cluster"] = labels

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_plot, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=70, edgecolor="k", alpha=0.8
)
plt.title("DBSCAN Clustering on PCA-reduced Data")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 9. Show Cluster Distribution ===
cluster_counts = df_plot["Cluster"].value_counts().sort_index()
print("\nDistribution of Samples per Cluster:")
print(cluster_counts)

# === 10. Add Cluster to Original Dataset and Show _Success_qual by Cluster ===
original_data_path = PROCESSED_DIR / "base_dataset.csv"
df_original = pd.read_csv(original_data_path)
df_original["Cluster"] = labels

if "_Success_qual" in df_original.columns:
    cross_tab = pd.crosstab(df_original["Cluster"], df_original["_Success_qual"], normalize="index")
    print("\nSuccess Distribution per Cluster (relative):")
    print(cross_tab)

    cross_tab.plot(kind="bar", stacked=True, colormap="Set2", figsize=(8, 5), edgecolor="k")
    plt.title("Distribution of Success per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Proportion")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
