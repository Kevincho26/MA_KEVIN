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
# ## 1. Logistic Regression: Baseline Model
#

# %%
"""
Logistic Regression Model for Predicting Product Success (_Success_qual)
------------------------------------------------------------------------
This notebook performs binary classification using Logistic Regression,
with data processed from 03_build_supervised_dataset.
"""

# === 1. Imports and Setup ===
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression

from src.data.loaders import load_supervised_modeling_dataset
from src.features.preprocessing import drop_columns_if_present, split_features_and_target
from src.models.evaluation import evaluate_binary_classifier
from src.models.splitting import scale_train_test, split_supervised_data

# === 2. Load Supervised Modeling Dataset ===
df = load_supervised_modeling_dataset()
print("Loaded supervised modeling dataset")

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)

X_train, X_test, y_train, y_test = split_supervised_data(X, y)

# === 4. Scale Features ===
X_train_scaled, X_test_scaled, scaler = scale_train_test(X_train, X_test)

# === 5. Train Logistic Regression Model ===
model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# === 6. Evaluate Model Performance ===
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix",
    roc_title="ROC Curve",
)

# === 7. Analyze Coefficients ===
coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_[0]}).sort_values(
    by="Coefficient", ascending=False
)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Coefficient", y="Feature", data=coef_df, palette="coolwarm")
plt.axvline(0, color="gray", linestyle="--")
plt.title("Logistic Regression Coefficients")

# Annotate values on bars
for i in ax.containers:
    ax.bar_label(i, fmt="%.2f", label_type="edge", fontsize=9, padding=3)

plt.tight_layout()
plt.show()

print("\nTop influencing features:")
print(coef_df.head())

# %% [markdown]
# ## 2. Logistic Regression: L1 (Lazo) Penalization

# %%
# === 5. Train L1-Regularized Logistic Regression Model ===
model = LogisticRegression(
    penalty="l1", solver="liblinear", class_weight="balanced", random_state=42, max_iter=1000
)
model.fit(X_train_scaled, y_train)

# === 6. Evaluate Model Performance ===
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (L1)",
    roc_title="ROC Curve (L1)",
)

# === 7. Analyze Coefficients ===
coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_[0]}).sort_values(
    by="Coefficient", ascending=False
)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Coefficient", y="Feature", data=coef_df, palette="coolwarm")
plt.axvline(0, color="gray", linestyle="--")
plt.title("L1-Regularized Coefficients")

# Annotate values on bars
for i in ax.containers:
    ax.bar_label(i, fmt="%.2f", label_type="edge", fontsize=9, padding=3)

plt.tight_layout()
plt.show()

print("\nTop influencing features:")
print(coef_df.head())

# %% [markdown]
# ## 3. Logistic Regression: Weight class penalization

# %%
# === 5. Train Logistic Regression with Custom Class Weights ===
custom_weights = {0: 5, 1: 1}  # Increased penalty for class 0
model = LogisticRegression(class_weight=custom_weights, random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# === 6. Evaluate Model Performance ===
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (Custom Weights: 0=5, 1=1)",
    roc_title="ROC Curve (Custom Weights: 0=5, 1=1)",
)

# === 7. Analyze Coefficients ===
coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_[0]}).sort_values(
    by="Coefficient", ascending=False
)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Coefficient", y="Feature", data=coef_df, palette="coolwarm")
plt.axvline(0, color="gray", linestyle="--")
plt.title("Logistic Regression Coefficients (Custom Weights: 0=5, 1=1)")

# Annotate values on bars
for i in ax.containers:
    ax.bar_label(i, fmt="%.2f", label_type="edge", fontsize=9, padding=3)

plt.tight_layout()
plt.show()

print("\nTop influencing features:")
print(coef_df.head())

# %% [markdown]
# ## 4. Logistic Regression: SMOTE

# %%
from imblearn.over_sampling import SMOTE

# === 5. Apply SMOTE to training data ===
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# === 6. Train Logistic Regression (No Class Weights) ===
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)

# === 7. Evaluate Model Performance ===
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (SMOTE)",
    roc_title="ROC Curve (SMOTE)",
)

# === 8. Analyze Coefficients ===
coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_[0]}).sort_values(
    by="Coefficient", ascending=False
)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Coefficient", y="Feature", data=coef_df, palette="coolwarm")
plt.axvline(0, color="gray", linestyle="--")
plt.title("Logistic Regression Coefficients (SMOTE)")

# Annotate values on bars
for i in ax.containers:
    ax.bar_label(i, fmt="%.2f", label_type="edge", fontsize=9, padding=3)

plt.tight_layout()
plt.show()

print("\nTop influencing features:")
print(coef_df.head())
