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
with data processed from d2_data_preprocessing_supervised.
"""

# === 1. Imports and Setup ===
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === 2. Load Preprocessed Supervised Dataset ===
csv_path = r"C:\\Master Thesis Repositories\\MAChoque\\data\\processed\\preprocessed_supervised.csv"
df = pd.read_csv(csv_path)

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X = df.drop(columns=["_Success_qual"] + colinear_vars, errors="ignore")
y = df["_Success_qual"]  # already binarized

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 4. Scale Features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 5. Train Logistic Regression Model ===
model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# === 6. Evaluate Model Performance ===
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["unsuccessful", "successful"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

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

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["unsuccessful", "successful"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (L1)")
plt.show()

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (L1)")
plt.legend()
plt.grid(True)
plt.show()

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

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["unsuccessful", "successful"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Custom Weights: 0=5, 1=1)")
plt.show()

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Custom Weights: 0=5, 1=1)")
plt.legend()
plt.grid(True)
plt.show()

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

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["unsuccessful", "successful"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (SMOTE)")
plt.show()

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (SMOTE)")
plt.legend()
plt.grid(True)
plt.show()

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
