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

# %%
import d0_data_preprocessing as d0
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load preprocessed dataset
df = d0.merged_df.copy()

# %% [markdown]
# ## 1. Dataset Overview

# %%
# Print shape and types
print("--- Dataset Overview ---")
print("Total rows:", df.shape[0])
print("Total columns:", df.shape[1])
print("\nData types and non-null counts:")
print(df.info())

# Summary statistics
summary = df.describe(include="all").transpose()
print("\nSummary Statistics:")
print(summary)

# Visualize class distribution
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(6, 5))
ax = sns.countplot(data=df, x="_Success_qual", palette=["#1f77b4", "#ff7f0e"], edgecolor="black")

# Add percentage + count inside the bars
total = len(df)
for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        label = f"{height / total:.1%} ({height})"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            label,
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            weight="bold",
        )

plt.title("Distribution of Innovation Success", fontsize=16, weight="bold", pad=15)
plt.xlabel("Success Label", fontsize=13)
plt.ylabel("Count", fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Print distribution to console
dist = df["_Success_qual"].value_counts(normalize=True).rename("proportion")
print("\nTarget class distribution:")
print(dist)

# %% [markdown]
# ## 2. Contextual Factor Richness

# %%
# Identify contextual columns (excluding technical ones)
contextual_cols = [
    col
    for col in df.columns
    if col not in ["_Gen_ID", "_Generation", "_Success_qual", "has_context", "Context_Others"]
    and not col.startswith("_")
]

# Count non-zero values per product across contextual factors
df["contextual_richness"] = df[contextual_cols].apply(lambda row: sum(row != "0"), axis=1)

# Print basic statistics
print("\nContextual Factor Richness Summary:")
print("Min:", df["contextual_richness"].min())
print("Max:", df["contextual_richness"].max())
print("Mean:", round(df["contextual_richness"].mean(), 2))
print("Median:", df["contextual_richness"].median())

# Visualize histogram
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 6))
bins_range = np.arange(
    df["contextual_richness"].min() - 0.5, df["contextual_richness"].max() + 1.5, 1
)
counts, bins, patches = plt.hist(
    df["contextual_richness"], bins=bins_range, color="#1f77b4", edgecolor="black", align="mid"
)

# Add count labels inside bars
for count, patch in zip(counts, patches):
    if count > 0:
        plt.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() / 2,
            f"{int(count)}",
            ha="center",
            va="center",
            color="white",
            weight="bold",
            fontsize=11,
        )

plt.title("Distribution of Contextual Factor Richness", fontsize=16, weight="bold")
plt.xlabel("Number of Active Contextual Factors", fontsize=13)
plt.ylabel("Number of Products", fontsize=13)
plt.xticks(np.arange(df["contextual_richness"].min(), df["contextual_richness"].max() + 1))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Correlation Analysis and Distribution of Technical Features
#

# %%
# Select only continuous numeric technical variables (excluding IDs, success, contextual flags)
numeric_vars = [
    "_Generation",
    "_δCV",
    "_δAV",
    "_δPV",
    "_δND",
    "_share_RSE_external",
    "_share_RSE_internal",
]

# Compute correlation matrix
corr_matrix = df[numeric_vars].corr()

# Set up plot style
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 11},
)
plt.title("Correlation Matrix of Numerical Technical Features", fontsize=16, weight="bold")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# falta analisis de colinealidad.

# %%
# Define numeric technical variables to visualize
numeric_features = ["_δCV", "_δAV", "_δPV", "_δND", "_share_RSE_external", "_share_RSE_internal"]

# Set visual style
sns.set_theme(style="whitegrid", font_scale=1.2)

# Create subplots grid
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 12))
axes = axes.flatten()

# Plot histogram with KDE overlay for each numeric variable
for i, var in enumerate(numeric_features):
    sns.histplot(data=df, x=var, kde=True, ax=axes[i], color="#1f77b4", edgecolor="black")
    axes[i].set_title(f"Distribution of {var}", fontsize=13)
    axes[i].set_xlabel(var, fontsize=11)
    axes[i].set_ylabel("Count", fontsize=11)

# Remove unused subplots
for j in range(len(numeric_features), len(axes)):
    fig.delaxes(axes[j])

# Layout adjustment
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4.  Distribution of Product Profile Technical Variation and Success
#

# %%
# Define technical variation columns
variation_columns = [
    "_PP_Claim_Variation",
    "_Provider_Benefit_Variation",
    "_Customer_Benefit_Variation",
]

# Set style
sns.set_theme(style="whitegrid", font_scale=1.2)
palette = sns.color_palette("Blues", n_colors=len(variation_columns))

# Create subplot layout
fig, axes = plt.subplots(1, len(variation_columns), figsize=(18, 5))

# Generate countplots for each categorical technical variable
for ax, col, color in zip(axes, variation_columns, palette):
    order = df[col].value_counts().index
    sns.countplot(data=df, x=col, order=order, palette=[color], edgecolor="black", ax=ax)
    ax.set_title(f"Distribution of {col}", fontsize=14, weight="bold")
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", label_type="edge", fontsize=10, padding=3)

plt.tight_layout()
plt.show()

# %%
# Set up layout
sns.set_theme(style="whitegrid", font_scale=1.2)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot each variation type vs. success
for ax, col in zip(axes, variation_columns):
    crosstab = pd.crosstab(df[col], df["_Success_qual"], normalize="index")
    crosstab.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=["#66c2a5", "#b3b3b3"],
        edgecolor="black",
        width=0.8,
        legend=False,
    )

    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + height / 2,
                    f"{height:.1%}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="black",
                )

    ax.set_title(f"Success Rate by {col}", fontsize=13)
    ax.set_xlabel(col.replace("_", " ").replace("Variation", "Variation").strip(), fontsize=11)
    ax.set_ylabel("Proportion", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# Add global legend outside the plot
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, labels, title="Success", loc="center right", bbox_to_anchor=(1.04, 0.5), fontsize=11
)
plt.tight_layout(rect=[0, 0, 0.97, 1])
plt.show()

# %% [markdown]
# ## 6. SGE Numeric Technical Features vs. Success

# %%
# Define numeric technical features (excluding contextual richness and ID)
numeric_vars = ["_δCV", "_δAV", "_δPV", "_δND", "_share_RSE_external", "_share_RSE_internal"]

# Set style
sns.set_theme(style="whitegrid", font_scale=1.2)

# Create boxplots comparing distributions by success label
plt.figure(figsize=(16, 10))
for i, col in enumerate(numeric_vars):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(data=df, x="_Success_qual", y=col, palette="Set2")
    plt.title(f"{col} by Success", fontsize=13)
    plt.xlabel("")
    plt.ylabel(col)
plt.tight_layout()
plt.show()

# Create violin plots for same variables
plt.figure(figsize=(16, 10))
for i, col in enumerate(numeric_vars):
    plt.subplot(3, 3, i + 1)
    sns.violinplot(data=df, x="_Success_qual", y=col, palette="Set3", inner="box")
    plt.title(f"{col} Distribution by Success", fontsize=13)
    plt.xlabel("")
    plt.ylabel(col)
plt.tight_layout()
plt.show()

# KDE plots to show density distribution
plt.figure(figsize=(16, 10))
for i, col in enumerate(numeric_vars):
    plt.subplot(3, 3, i + 1)
    for label, color in zip(df["_Success_qual"].unique(), ["#2ca02c", "#d62728"]):
        subset = df[df["_Success_qual"] == label]
        sns.kdeplot(subset[col], label=label, fill=True, alpha=0.3, linewidth=2)
    plt.title(f"{col} KDE by Success", fontsize=13)
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Aggregated Contextual Variables Vs Success (Not meaningful)

# %%
# ---- Boxplot, Violin and KDE for contextual_richness ----
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x="_Success_qual", y="contextual_richness", palette="pastel")
plt.title("Contextual Richness by Success", fontsize=14)
plt.xlabel("")
plt.ylabel("contextual_richness")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.violinplot(data=df, x="_Success_qual", y="contextual_richness", palette="Set3", inner="box")
plt.title("Contextual Richness Distribution by Success", fontsize=14)
plt.xlabel("")
plt.ylabel("contextual_richness")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
for label, color in zip(df["_Success_qual"].unique(), ["#2ca02c", "#d62728"]):
    subset = df[df["_Success_qual"] == label]
    sns.kdeplot(subset["contextual_richness"], label=label, fill=True, alpha=0.3, linewidth=2)
plt.title("Contextual Richness KDE by Success", fontsize=14)
plt.xlabel("contextual_richness")
plt.ylabel("Density")
plt.legend(title="Success")
plt.tight_layout()
plt.show()

# ---- Context_Others bar plot ----
crosstab = pd.crosstab(df["Context_Others"], df["_Success_qual"], normalize="index")
crosstab.plot(kind="bar", stacked=True, figsize=(6, 5), colormap="Pastel1", edgecolor="black")
plt.title("Success Rate by Context_Others", fontsize=14, weight="bold")
plt.xlabel("Context_Others")
plt.ylabel("Proportion")
plt.xticks(ticks=[0, 1], labels=["No Other Factors", "Has Other Factors"], rotation=0)
plt.legend(title="Success", loc="upper right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Individual Contextual Variables vs Succes. Chi² & Cramér’s V Analysis (Contextual Variables)

# %% [markdown]
# ###  7.1  Individual Contextual Variables – Distribution & Success Rate

# %%
# Define contextual factor columns (excluding aggregated columns)
contextual_factors = [
    col
    for col in df.columns
    if col
    not in [
        "_Gen_ID",
        "_Generation",
        "_Success_qual",
        "has_context",
        "Context_Others",
        "contextual_richness",
    ]
    and not col.startswith("_")
]

# Distribution of each contextual factor
sns.set_theme(style="whitegrid", font_scale=1.2)
n_cols = 3
n_rows = int(np.ceil(len(contextual_factors) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(contextual_factors):
    order = df[col].value_counts().index
    sns.countplot(data=df, x=col, order=order, palette="pastel", ax=axes[i], edgecolor="black")
    axes[i].set_title(f"Distribution of {col}", fontsize=13)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    for container in axes[i].containers:
        axes[i].bar_label(container, fmt="%d", label_type="edge", fontsize=10, padding=3)

for j in range(len(contextual_factors), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Define contextual factors
contextual_factors = [
    col
    for col in df.columns
    if col
    not in [
        "_Gen_ID",
        "_Generation",
        "_Success_qual",
        "has_context",
        "Context_Others",
        "contextual_richness",
    ]
    and not col.startswith("_")
]

# Set plotting style
sns.set_theme(style="whitegrid", font_scale=1.2)

# Stacked bar plots of success rate per contextual factor
n_cols = 3
n_rows = int(np.ceil(len(contextual_factors) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
axes = axes.flatten()

for i, factor in enumerate(contextual_factors):
    crosstab = pd.crosstab(df[factor], df["_Success_qual"], normalize="index")
    crosstab.plot(
        kind="bar",
        stacked=True,
        ax=axes[i],
        color=["#66c2a5", "#b3b3b3"],
        edgecolor="black",
        width=0.8,
        legend=False,
    )
    axes[i].set_title(f"Success Rate by {factor}", fontsize=13)
    axes[i].set_xlabel(factor)
    axes[i].set_ylabel("Proportion")
    axes[i].tick_params(axis="x", rotation=0)

# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Add single legend to the right of the last plot
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    ["successful", "unsuccessful"],
    title="Success",
    loc="center right",
    bbox_to_anchor=(1.12, 0.5),
)

plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.show()

# %% [markdown]
# ### 7.2  Chi² & Cramér’s V Analysis

# %%
from scipy.stats import chi2_contingency

# Define contextual factors (excluding aggregated variables)
contextual_factors = [
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

chi2_results = []

# Calculate Chi-square and Cramer's V
for factor in contextual_factors:
    contingency_table = pd.crosstab(df[factor], df["_Success_qual"])
    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        n = contingency_table.to_numpy().sum()
        phi2 = chi2 / n
        r, k = contingency_table.shape
        cramers_v = np.sqrt(phi2 / min(k - 1, r - 1))
        chi2_results.append({"Factor": factor, "Cramers_V": cramers_v, "p_value": p})

# Convert to DataFrame and sort
results_df = pd.DataFrame(chi2_results)
results_df.sort_values("Cramers_V", ascending=False, inplace=True)

# Plot configuration
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 6))
barplot = sns.barplot(
    data=results_df, x="Cramers_V", y="Factor", palette="Blues_d", edgecolor="black"
)

# Add labels centered inside each bar
for index, row in results_df.reset_index().iterrows():
    x_value = row["Cramers_V"] / 2
    y_value = index
    label = f"V={row['Cramers_V']:.2f}\np={row['p_value']:.4f}"
    barplot.text(x_value, y_value, label, color="black", ha="center", va="center", fontsize=10)

# Titles and layout
plt.title("Cramér's V for Contextual Factors vs. Success", fontsize=16, weight="bold")
plt.xlabel("Cramér's V")
plt.ylabel("Contextual Factor")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Information Value and Weight of Evidence (WoE) for Contextual Factors

# %%
# Define contextual factors (excluding aggregated variables)
contextual_factors = [
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


def calculate_woe_iv(data, feature, target):
    epsilon = 0.0001
    lst = []
    for val in data[feature].unique():
        total = len(data[data[feature] == val])
        good = len(data[(data[feature] == val) & (data[target] == "successful")])
        bad = len(data[(data[feature] == val) & (data[target] == "unsuccessful")])
        lst.append([val, total, good, bad])

    dset = pd.DataFrame(lst, columns=["Value", "Total", "Good", "Bad"])
    dset["Distr_Good"] = dset["Good"] / dset["Good"].sum()
    dset["Distr_Bad"] = dset["Bad"] / dset["Bad"].sum()
    dset["WoE"] = np.log((dset["Distr_Good"] + epsilon) / (dset["Distr_Bad"] + epsilon))
    dset["IV"] = (dset["Distr_Good"] - dset["Distr_Bad"]) * dset["WoE"]

    iv = dset["IV"].sum()
    return iv


# Calculate IV for each contextual factor
iv_results = []
for factor in contextual_factors:
    try:
        iv = calculate_woe_iv(df, factor, "_Success_qual")
        iv_results.append({"Factor": factor, "Information_Value": iv})
    except Exception as e:
        print(f"Error with {factor}: {e}")

# Create DataFrame and sort
iv_df = pd.DataFrame(iv_results).sort_values(by="Information_Value", ascending=False)

# Plot configuration
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 6))
barplot = sns.barplot(
    data=iv_df, x="Information_Value", y="Factor", palette="Blues_d", edgecolor="black"
)

# Add labels to the right of each bar
for index, row in iv_df.reset_index(drop=True).iterrows():
    x_value = row["Information_Value"]
    y_value = index
    label = f"IV={row['Information_Value']:.3f}"
    barplot.text(x_value + 0.01, y_value, label, color="black", ha="left", va="center", fontsize=10)

# Titles and layout
plt.title("Information Value (IV) for Contextual Factors", fontsize=16, weight="bold")
plt.xlabel("Information Value (IV)")
plt.ylabel("Contextual Factor")
plt.xlim(0, iv_df["Information_Value"].max() + 0.1)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. PCA

# %% [markdown]
# ### 9.1 PCA with only Technical Features.

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Select technical features
technical_cols = [
    "_Generation",
    "_δCV",
    "_δAV",
    "_δPV",
    "_δND",
    "_share_RSE_external",
    "_share_RSE_internal",
]
X_tech = df[technical_cols]

# Standardize technical features
scaler = StandardScaler()
X_scaled_tech = scaler.fit_transform(X_tech)

# Apply PCA
pca_tech = PCA(n_components=None)
X_pca_tech = pca_tech.fit_transform(X_scaled_tech)

# Variance explained
explained_var = pca_tech.explained_variance_ratio_ * 100
cum_var = np.cumsum(explained_var)

# Scree Plot with values
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 6))
components = np.arange(1, len(explained_var) + 1)
plt.plot(components, explained_var, marker="o", label="Individual Variance")
plt.plot(components, cum_var, marker="s", linestyle="--", label="Cumulative Variance")

# Add value annotations
for i, (x, y1, y2) in enumerate(zip(components, explained_var, cum_var)):
    plt.text(x, y1 + 1.2, f"{y1:.1f}%", ha="center", va="bottom", fontsize=10, color="blue")
    plt.text(x, y2 + 1.2, f"{y2:.1f}%", ha="center", va="bottom", fontsize=10, color="gray")

plt.xticks(components)
plt.title("Scree Plot – PCA on Technical Variables", fontsize=16, weight="bold")
plt.xlabel("Principal Component", fontsize=13)
plt.ylabel("Explained Variance (%)", fontsize=13)
plt.legend()
plt.tight_layout()
plt.show()

# 2D PCA Projection
pca_df_tech = pd.DataFrame(X_pca_tech[:, :2], columns=["PC1", "PC2"])
pca_df_tech["_Success_qual"] = df["_Success_qual"].values

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=pca_df_tech,
    x="PC1",
    y="PC2",
    hue="_Success_qual",
    palette={"successful": "#1f77b4", "unsuccessful": "#ff7f0e"},
    edgecolor="black",
    s=100,
    alpha=0.8,
)
plt.title("PCA Projection – Technical Variables", fontsize=16, weight="bold")
plt.xlabel(f"PC1 ({explained_var[0]:.1f}% var)", fontsize=13)
plt.ylabel(f"PC2 ({explained_var[1]:.1f}% var)", fontsize=13)
plt.legend(title="Success", loc="best")
plt.tight_layout()
plt.show()

# Loadings
loadings_tech = pd.DataFrame(
    pca_tech.components_.T,
    columns=[f"PC{i + 1}" for i in range(len(technical_cols))],
    index=technical_cols,
)

print("\nPCA Loadings – Technical Variables:")
print(loadings_tech.round(3))

# Heatmap of Loadings
plt.figure(figsize=(10, 6))
sns.heatmap(
    loadings_tech.iloc[:, :5],  # limit to first 5 components for readability
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f",
    cbar_kws={"label": "Loading Value"},
)
plt.title("PCA Loadings Heatmap – Technical Variables", fontsize=16, weight="bold")
plt.xlabel("Principal Components", fontsize=13)
plt.ylabel("Technical Variables", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 9.2 PCA with evolutionary data (bot technical and contextual)

# %%
# Define technical and contextual columns
technical_cols = [
    "_Generation",
    "_δCV",
    "_δAV",
    "_δPV",
    "_δND",
    "_share_RSE_external",
    "_share_RSE_internal",
]
contextual_cols = [
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

# Ordinal encoding for contextual variables
ordinal_map = {"0": 0, "low": 1, "high": 2}
df_encoded = df.copy()
for col in contextual_cols:
    df_encoded[col] = df_encoded[col].map(ordinal_map)

# Combine features
combined_cols = technical_cols + contextual_cols
X_combined = df_encoded[combined_cols]

# Standardize features
scaler = StandardScaler()
X_scaled_combined = scaler.fit_transform(X_combined)

# Apply PCA
pca_combined = PCA(n_components=None)
X_pca_combined = pca_combined.fit_transform(X_scaled_combined)

# Explained variance
explained_var = pca_combined.explained_variance_ratio_ * 100
cum_var = np.cumsum(explained_var)

# Scree Plot
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 6))
components = np.arange(1, len(explained_var) + 1)
plt.plot(components, explained_var, marker="o", label="Individual Variance")
plt.plot(components, cum_var, marker="s", linestyle="--", label="Cumulative Variance")

# Annotate values
for i, (x, y1, y2) in enumerate(zip(components, explained_var, cum_var)):
    plt.text(x, y1 + 1.2, f"{y1:.1f}%", ha="center", fontsize=10, color="blue")
    plt.text(x, y2 + 1.2, f"{y2:.1f}%", ha="center", fontsize=10, color="gray")

plt.xticks(components)
plt.title("Scree Plot – PCA on Technical + Contextual Features", fontsize=16, weight="bold")
plt.xlabel("Principal Component", fontsize=13)
plt.ylabel("Explained Variance (%)", fontsize=13)
plt.legend()
plt.tight_layout()
plt.show()

# 2D Projection
pca_df_combined = pd.DataFrame(X_pca_combined[:, :2], columns=["PC1", "PC2"])
pca_df_combined["_Success_qual"] = df["_Success_qual"].values

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=pca_df_combined,
    x="PC1",
    y="PC2",
    hue="_Success_qual",
    palette={"successful": "#1f77b4", "unsuccessful": "#ff7f0e"},
    edgecolor="black",
    s=100,
    alpha=0.8,
)
plt.title("PCA Projection – Technical + Contextual Features", fontsize=16, weight="bold")
plt.xlabel(f"PC1 ({explained_var[0]:.1f}% var)", fontsize=13)
plt.ylabel(f"PC2 ({explained_var[1]:.1f}% var)", fontsize=13)
plt.legend(title="Success", loc="best")
plt.tight_layout()
plt.show()

# Loadings heatmap
loadings_combined = pd.DataFrame(
    pca_combined.components_.T,
    columns=[f"PC{i + 1}" for i in range(len(combined_cols))],
    index=combined_cols,
)

plt.figure(figsize=(12, 6))
sns.heatmap(
    loadings_combined.iloc[:, :5],
    annot=True,
    cmap="Blues",
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "Loading strength"},
)
plt.title("PCA Loadings Heatmap – Technical + Contextual Features", fontsize=16, weight="bold")
plt.xlabel("Principal Components", fontsize=13)
plt.ylabel("Features", fontsize=13)
plt.tight_layout()
plt.show()

# Print loadings
print("\nPCA Loadings – Technical + Contextual Features:")
print(loadings_combined.round(3))

# %% [markdown]
# ### 9.3 PCA Comparison

# %%
# --- Define columns ---
technical_cols = [
    "_Generation",
    "_δCV",
    "_δAV",
    "_δPV",
    "_δND",
    "_share_RSE_external",
    "_share_RSE_internal",
]
contextual_cols = [
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

# Ordinal encoding for contextual variables
ordinal_map = {"0": 0, "low": 1, "high": 2}
df_encoded = df.copy()
for col in contextual_cols:
    df_encoded[col] = df_encoded[col].map(ordinal_map)

# --- PCA 1: Technical only ---
X_tech = df[technical_cols]
X_scaled_tech = StandardScaler().fit_transform(X_tech)
pca_tech = PCA()
X_pca_tech = pca_tech.fit_transform(X_scaled_tech)
loadings_tech = pd.DataFrame(
    pca_tech.components_.T,
    columns=[f"Tech_PC{i + 1}" for i in range(len(technical_cols))],
    index=technical_cols,
)

# --- PCA 2: Technical + Contextual ---
X_mix = df_encoded[technical_cols + contextual_cols]
X_scaled_mix = StandardScaler().fit_transform(X_mix)
pca_mix = PCA()
X_pca_mix = pca_mix.fit_transform(X_scaled_mix)
loadings_mix = pd.DataFrame(
    pca_mix.components_.T,
    columns=[f"Mix_PC{i + 1}" for i in range(X_mix.shape[1])],
    index=technical_cols + contextual_cols,
)

# --- Scree Plot Comparison ---
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca_tech.explained_variance_ratio_) * 100, label="Technical PCA", marker="o")
plt.plot(
    np.cumsum(pca_mix.explained_variance_ratio_) * 100,
    label="Mixed PCA",
    marker="s",
    linestyle="--",
)
plt.xticks(np.arange(1, max(X_mix.shape[1], len(technical_cols)) + 1))
plt.title("Cumulative Explained Variance Comparison", fontsize=16, weight="bold")
plt.xlabel("Number of Components", fontsize=13)
plt.ylabel("Cumulative Explained Variance (%)", fontsize=13)
plt.legend()
plt.tight_layout()
plt.show()

# --- Loadings Comparison Table (PC1 & PC2 only) ---
loadings_summary = pd.concat(
    [loadings_tech[["Tech_PC1", "Tech_PC2"]], loadings_mix[["Mix_PC1", "Mix_PC2"]]], axis=1
)
print("\nPCA Loadings Comparison (PC1 & PC2):")
print(loadings_summary.round(3))

# --- Projection Comparison Plots with Independent Axes ---
pca_df_tech = pd.DataFrame(X_pca_tech[:, :2], columns=["PC1", "PC2"])
pca_df_tech["_Success_qual"] = df["_Success_qual"].values
pca_df_mix = pd.DataFrame(X_pca_mix[:, :2], columns=["PC1", "PC2"])
pca_df_mix["_Success_qual"] = df["_Success_qual"].values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Technical projection
sns.scatterplot(
    data=pca_df_tech,
    x="PC1",
    y="PC2",
    hue="_Success_qual",
    palette={"successful": "#1f77b4", "unsuccessful": "#ff7f0e"},
    edgecolor="black",
    alpha=0.8,
    s=100,
    ax=ax1,
)
ax1.set_title("PCA Projection – Technical")
ax1.set_xlabel(f"PC1 ({pca_tech.explained_variance_ratio_[0] * 100:.1f}% var)")
ax1.set_ylabel(f"PC2 ({pca_tech.explained_variance_ratio_[1] * 100:.1f}% var)")
ax1.get_legend().remove()

# Mixed projection
sns.scatterplot(
    data=pca_df_mix,
    x="PC1",
    y="PC2",
    hue="_Success_qual",
    palette={"successful": "#1f77b4", "unsuccessful": "#ff7f0e"},
    edgecolor="black",
    alpha=0.8,
    s=100,
    ax=ax2,
)
ax2.set_title("PCA Projection – Mixed")
ax2.set_xlabel(f"PC1 ({pca_mix.explained_variance_ratio_[0] * 100:.1f}% var)")
ax2.set_ylabel(f"PC2 ({pca_mix.explained_variance_ratio_[1] * 100:.1f}% var)")
ax2.get_legend().remove()

# Shared legend
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, title="Success", loc="lower center", ncol=2)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

# %% [markdown]
# ## 10. Feature Importance

# %% [markdown]
# Corregir la grafica de barrras

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define features (technical + contextual)
technical_cols = [
    "_Generation",
    "_δCV",
    "_δAV",
    "_δPV",
    "_δND",
    "_share_RSE_external",
    "_share_RSE_internal",
]
contextual_cols = [
    "Competition",
    "Customer",
    "Legal",
    "Market",
    "Organization",
    "Politics",
    "Product_development",
    "Society",
    "Technology",
    "Context_Others",
]
meta_cols = ["has_context"]

# Ordinal encode contextual features
ordinal_map = {"0": 0, "low": 1, "high": 2}
df_encoded = df.copy()
for col in contextual_cols:
    df_encoded[col] = df_encoded[col].map(ordinal_map)

# Prepare data for classification
X = df_encoded[technical_cols + contextual_cols + meta_cols]
y = df_encoded["_Success_qual"]

# Encode target
y_encoded = y.astype("category").cat.codes  # 1 = successful, 0 = unsuccessful

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(
    by="Importance", ascending=False
)

# Plot
plt.figure(figsize=(10, 7))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
plt.title("Feature Importance (Random Forest)", fontsize=16, weight="bold")
plt.xlabel("Importance Score", fontsize=13)
plt.ylabel("Feature", fontsize=13)
plt.tight_layout()
plt.show()

# Print table
print("\nFeature Importance Table:")
print(importance_df.to_string(index=False, float_format="{:.3f}".format))

# %%
from sklearn.metrics import accuracy_score, f1_score

y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# %% [markdown]
# ## 11. Cluster Tendency & Dimensionality Insights

# %%
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import uniform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Use both technical and encoded contextual variables for clustering
technical_cols = [
    "_Generation",
    "_δCV",
    "_δAV",
    "_δPV",
    "_δND",
    "_share_RSE_external",
    "_share_RSE_internal",
]
contextual_cols = [
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
ordinal_map = {"0": 0, "low": 1, "high": 2}
df_encoded = df.copy()
for col in contextual_cols:
    df_encoded[col] = df_encoded[col].map(ordinal_map)

features = technical_cols + contextual_cols
X = df_encoded[features]
X_scaled = StandardScaler().fit_transform(X)


# ----------------------
# 2. Hopkins Statistic to assess cluster tendency
# ----------------------
def hopkins(X):
    d = X.shape[1]
    n = len(X)
    m = int(0.1 * n)
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)

    rand_X = sample(range(0, n), m)
    ujd = []
    wjd = []

    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(
            [uniform(np.amin(X, axis=0), np.amax(X, axis=0))], 2, return_distance=True
        )
        w_dist, _ = nbrs.kneighbors([X[rand_X[j]]], 2, return_distance=True)
        ujd.append(u_dist[0][1])
        wjd.append(w_dist[0][1])

    H = sum(ujd) / (sum(ujd) + sum(wjd))
    return H


hopkins_stat = hopkins(X_scaled)
print(f"Hopkins Statistic: {hopkins_stat:.4f}")

# ----------------------
# 3. Elbow Method
# ----------------------
sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, sse, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.title("Elbow Method for Optimal k", fontsize=14, weight="bold")
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------
# 4. Silhouette Analysis
# ----------------------
silhouette_scores = []
k_vals = range(2, 11)

for k in k_vals:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(8, 5))
plt.plot(k_vals, silhouette_scores, marker="s", color="darkgreen")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis for Optimal k", fontsize=14, weight="bold")
plt.xticks(k_vals)
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- Define number of clusters ---
k = 5  # or 5, based on previous analysis

# --- Run KMeans clustering (for visualization only) ---
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled_mix)  # X_scaled_mix ya está estandarizado desde antes

# --- PCA for 2D visualization ---
pca_vis = PCA(n_components=2)
pca_2d = pca_vis.fit_transform(X_scaled_mix)

# --- Create dataframe for plotting ---
pca_df = pd.DataFrame(pca_2d, columns=["PC1", "PC2"])
pca_df["Cluster"] = clusters
pca_df["_Success_qual"] = df["_Success_qual"].values
pca_df["Generation"] = df["_Generation"].values  # por si deseas analizar eso también

# --- Plot 1: Clustering con KMeans ---
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10",
    s=80,
    alpha=0.9,
    edgecolor="black",
)
plt.title("Clustering Visualization with k-Means (k=4)", fontsize=15)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

# --- Plot 2: Success overlay sobre clustering ---
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="_Success_qual",
    style="Cluster",
    s=100,
    palette={"successful": "#1f77b4", "unsuccessful": "#ff7f0e"},
    edgecolor="black",
)
plt.title("Success Overlay on Clustered Data (PCA space)", fontsize=15)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Success / Cluster")
plt.tight_layout()
plt.show()

# --- (Opcional) Plot 3: Color por generación ---
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Generation",
    style="Cluster",
    palette="coolwarm",
    s=100,
    edgecolor="black",
)
plt.title("Generation Overlay on Clustered Data (PCA space)", fontsize=15)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()
