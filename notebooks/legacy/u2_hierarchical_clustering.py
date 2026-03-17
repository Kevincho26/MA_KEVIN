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
# # Hierarchical Clustering
#

# %%
# Hierarchical Clustering Model
# -----------------------------
# This script applies Agglomerative Hierarchical Clustering to the PCA-reduced dataset.
# It visualizes the dendrogram and evaluates the clustering structure.

# === 1. Imports ===
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# === 2. Load Preprocessed Unsupervised Dataset ===
csv_path = (
    r"C:\\Master Thesis Repositories\\MAChoque\\data\\processed\\preprocessed_unsupervised.csv"
)
df = pd.read_csv(csv_path)

# === 3. Standardize ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# === 4. PCA for 2D Projection ===
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# === 5. Perform Hierarchical Clustering ===
linked = linkage(X_pca, method="ward")

# === 6. Plot Dendrogram ===
plt.figure(figsize=(10, 6))
dendrogram(linked, truncate_mode="level", p=5)
plt.title("Hierarchical Clustering Dendrogram (truncated)")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# === 7. Extract Flat Clusters ===
n_clusters = 4  # You can adjust this number based on dendrogram analysis
cluster_labels = fcluster(linked, n_clusters, criterion="maxclust")

# === 8. Evaluate Clustering ===
silhouette = silhouette_score(X_pca, cluster_labels)
print(f"\nHierarchical Clustering with {n_clusters} clusters")
print(f"Silhouette Score: {silhouette:.2f}")

# === 9. Visualize Clusters ===
df_plot = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_plot["Cluster"] = cluster_labels

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_plot, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=70, edgecolor="k", alpha=0.8
)
plt.title("Hierarchical Clustering on PCA-reduced Data")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 10. Add Cluster to Original Dataset and Show _Success_qual by Cluster ===
original_data_path = r"C:\\Master Thesis Repositories\\MAChoque\\data\\processed\\preprocessed.csv"
df_original = pd.read_csv(original_data_path)
df_original["Cluster"] = cluster_labels

if "_Success_qual" in df_original.columns:
    cross_tab = pd.crosstab(df_original["Cluster"], df_original["_Success_qual"], normalize="index")
    print("\nSuccess Distribution per Cluster (relative):")
    print(cross_tab)

    ax = cross_tab.plot(kind="bar", stacked=True, colormap="Set2", figsize=(8, 5), edgecolor="k")
    plt.title("Distribution of Success per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Proportion")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)

    # Add percentage labels
    for c in ax.containers:
        labels = [f"{v.get_height():.0%}" if v.get_height() > 0 else "" for v in c]
        ax.bar_label(c, labels=labels, label_type="center", fontsize=9, color="black")

    plt.legend(title="_Success_qual", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

# === 11. Show Gen_ID, Success_qual, and Context by Cluster ===
print("\nSample Overview by Cluster:")
print(df_original[["_Gen_ID", "_Success_qual", "Cluster"]].head(10))

# === 12. Cluster Descriptive Analysis ===
print("\nCluster Descriptive Analysis (mean values):")
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_descriptive = pd.DataFrame(X_scaled, columns=numeric_cols)
df_descriptive["Cluster"] = cluster_labels
descriptive_summary = df_descriptive.groupby("Cluster").mean().round(2)
print(descriptive_summary)

# === 13. Visualize Cluster Profiles ===
descriptive_summary.T.plot(kind="bar", figsize=(12, 6), colormap="tab10")
plt.title("Cluster Profiles by Feature Means")
plt.ylabel("Standardized Mean Value")
plt.xlabel("Feature")
plt.xticks(rotation=45, ha="right")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# %% [markdown]
# NOTE: Use the cluster profile chart to visually interpret differences between clusters.
# Each bar shows the standardized mean of a feature within each cluster.
# This aids interpretation for reporting (e.g., "Cluster 2 is strong in Competition and Product Development").
# The interpretation itself is not included in the code and should be written manually in the documentation.
