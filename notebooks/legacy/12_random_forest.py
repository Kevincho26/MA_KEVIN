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
# ## 1. Random Forest: Baseline
#
#

# %%
"""
Random Forest Model (v1 - Basic)
----------------------------------
This version uses a default Random Forest classifier to evaluate base performance
without class balancing or resampling.
"""

# === 1. Imports and Setup ===
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.utils.paths import PROCESSED_DIR

# === 2. Load Preprocessed Supervised Dataset ===
csv_path = PROCESSED_DIR / "supervised_modeling_dataset.csv"
df = pd.read_csv(csv_path)

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X = df.drop(columns=["_Success_qual"] + colinear_vars, errors="ignore")
y = df["_Success_qual"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 4. Train Basic Random Forest ===
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# === 5. Evaluate Performance ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["unsuccessful", "successful"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Random Forest Basic)")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Random Forest Basic)")
plt.legend()
plt.grid(True)
plt.show()

# === 6. Feature Importance ===
importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm")
plt.title("Feature Importance (Random Forest Basic)")

# Annotate values on bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9, padding=3)

plt.tight_layout()
plt.show()

print("\nTop influencing features:")
print(importance_df.head())

# %% [markdown]
# ## 2. Random Forest: Class Weight Balanced

# %%
# === 1. Imports and Setup ===
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# === 2. Load Preprocessed Supervised Dataset ===
csv_path = PROCESSED_DIR / "supervised_modeling_dataset.csv"
df = pd.read_csv(csv_path)

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X = df.drop(columns=["_Success_qual"] + colinear_vars, errors="ignore")
y = df["_Success_qual"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 4. Train Random Forest with Class Weight Balanced ===
model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight="balanced")
model.fit(X_train, y_train)

# === 5. Evaluate Performance ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["unsuccessful", "successful"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Random Forest v2 - Balanced)")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Random Forest v2 - Balanced)")
plt.legend()
plt.grid(True)
plt.show()

# === 6. Feature Importance ===
importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm")
plt.title("Feature Importance (Random Forest v2 - Balanced)")

# Annotate values on bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9, padding=3)

plt.tight_layout()
plt.show()

print("\nTop influencing features:")
print(importance_df.head())

# %% [markdown]
# ## 3. Random Forest: SMOTE

# %%
# === 1. Imports and Setup ===
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# === 2. Load Preprocessed Supervised Dataset ===
csv_path = PROCESSED_DIR / "supervised_modeling_dataset.csv"
df = pd.read_csv(csv_path)

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X = df.drop(columns=["_Success_qual"] + colinear_vars, errors="ignore")
y = df["_Success_qual"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 4. Apply SMOTE on Training Data Only ===
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# === 5. Train Random Forest on Resampled Data ===
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train_resampled, y_train_resampled)

# === 6. Evaluate Performance ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["unsuccessful", "successful"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Random Forest SMOTE)")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Random Forest SMOTE)")
plt.legend()
plt.grid(True)
plt.show()

# === 7. Feature Importance ===
importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm")
plt.title("Feature Importance (Random Forest SMOTE)")

# Annotate values on bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9, padding=3)

plt.tight_layout()
plt.show()

print("\nTop influencing features:")
print(importance_df.head())

# %% [markdown]
# ## 4. Random Forest: SMOTE + Class Weight + Threshold

# %%
# This version applies SMOTE to oversample the minority class, uses class_weight='balanced' in RandomForestClassifier, and adjusts the decision threshold.


# === 1. Imports and Setup ===

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# === 2. Load Preprocessed Supervised Dataset ===
csv_path = PROCESSED_DIR / "supervised_modeling_dataset.csv"
df = pd.read_csv(csv_path)

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X = df.drop(columns=["_Success_qual"] + colinear_vars, errors="ignore")
y = df["_Success_qual"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 4. Apply SMOTE on Training Data Only ===
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# === 5. Train Random Forest on Resampled Data with Class Weight ===
model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight="balanced")
model.fit(X_train_resampled, y_train_resampled)

# === 6. Predict Probabilities and Adjust Threshold ===
y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.6  # lowered to increase sensitivity to class 0
y_pred = (y_prob >= threshold).astype(int)

# === 7. Evaluate Performance ===
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["unsuccessful", "successful"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Random Forest SMOTE + Class Weight + Threshold)")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Random Forest SMOTE + Class Weight + Threshold)")
plt.legend()
plt.grid(True)
plt.show()

# === 8. Feature Importance ===
importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm")
plt.title("Feature Importance (Random Forest SMOTE + Class Weight + Threshold)")

# Annotate values on bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9, padding=3)

plt.tight_layout()
plt.show()

print("\nTop influencing features:")
print(importance_df.head())
