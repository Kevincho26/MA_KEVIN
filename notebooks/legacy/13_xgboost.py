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
# ## 1. XGBoost: Baseline

# %%
# This version trains a basic XGBoost classifier without class weighting, SMOTE, or threshold adjustment.


# === 1. Imports and Setup ===
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data.loaders import load_supervised_modeling_dataset

# === 2. Load Preprocessed Supervised Dataset ===
df = load_supervised_modeling_dataset()

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X = df.drop(columns=["_Success_qual"] + colinear_vars, errors="ignore")
y = df["_Success_qual"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 4. Train XGBoost Classifier ===
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(X_train, y_train)

# === 5. Predict and Evaluate ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["unsuccessful", "successful"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (XGBoost Basic)")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (XGBoost Basic)")
plt.legend()
plt.grid(True)
plt.show()

# === 6. Feature Importance ===
importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm")
plt.title("Feature Importance (XGBoost Basic)")

# Annotate values on bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9, padding=3)

plt.tight_layout()
plt.show()

print("\nTop influencing features:")
print(importance_df.head())

# %% [markdown]
# ## 2. XGBoost: Balanced

# %%
# This version uses `scale_pos_weight` to handle class imbalance based on the training set distribution.


# === 1. Imports and Setup ===
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# === 2. Load Preprocessed Supervised Dataset ===
df = load_supervised_modeling_dataset()

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X = df.drop(columns=["_Success_qual"] + colinear_vars, errors="ignore")
y = df["_Success_qual"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 4. Compute Class Weight for Balance ===
class_counts = Counter(y_train)
scale_pos_weight = class_counts[0] / class_counts[1]

# === 5. Train XGBoost Classifier with Balance ===
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=scale_pos_weight,
)
model.fit(X_train, y_train)

# === 6. Predict and Evaluate ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["unsuccessful", "successful"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (XGBoost Balanced)")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (XGBoost Balanced)")
plt.legend()
plt.grid(True)
plt.show()

# === 7. Feature Importance ===
importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm")
plt.title("Feature Importance (XGBoost Balanced)")

# Annotate values on bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9, padding=3)

plt.tight_layout()
plt.show()

print("\nTop influencing features:")
print(importance_df.head())

# %%
# This version uses class_weight approximation and adjusts the decision threshold to improve detection of the minority class.

# === 1. Imports and Setup ===
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# === 2. Load Preprocessed Supervised Dataset ===
df = load_supervised_modeling_dataset()

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X = df.drop(columns=["_Success_qual"] + colinear_vars, errors="ignore")
y = df["_Success_qual"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 4. Train XGBoost Classifier with Class Weight Adjustment ===
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
model = XGBClassifier(
    use_label_encoder=False, eval_metric="logloss", scale_pos_weight=ratio, random_state=42
)
model.fit(X_train, y_train)

# === 5. Predict and Evaluate with Threshold Adjustment ===
threshold = 0.6  # Manually selected threshold
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["unsuccessful", "successful"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (XGBoost Balanced + Threshold)")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (XGBoost Balanced + Threshold)")
plt.legend()
plt.grid(True)
plt.show()

# === 6. Feature Importance ===
importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm")
plt.title("Feature Importance (XGBoost Balanced + Threshold)")

# Annotate values on bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9, padding=3)

plt.tight_layout()
plt.show()

print("\nTop influencing features:")
print(importance_df.head())

# %% [markdown]
# ## 3. XGBoost: SMOTE

# %%
# This version applies SMOTE to balance the training data and trains an XGBoost classifier without using class weights or threshold adjustment.

# === 1. Imports and Setup ===
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# === 2. Load Preprocessed Supervised Dataset ===
df = load_supervised_modeling_dataset()

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X = df.drop(columns=["_Success_qual"] + colinear_vars, errors="ignore")
y = df["_Success_qual"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 4. Apply SMOTE to Training Data ===
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# === 5. Train XGBoost Classifier ===
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(X_train_res, y_train_res)

# === 6. Predict and Evaluate ===
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["unsuccessful", "successful"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (XGBoost SMOTE)")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (XGBoost SMOTE)")
plt.legend()
plt.grid(True)
plt.show()

# === 7. Feature Importance ===
importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm")
plt.title("Feature Importance (XGBoost SMOTE)")

# Annotate values on bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9, padding=3)

plt.tight_layout()
plt.show()

print("\nTop influencing features:")
print(importance_df.head())

# %% [markdown]
# ## 4. XGBoost: SMOTE + Class Weight + Threshold

# %%
# This version applies SMOTE to balance the training data and trains an XGBoost classifier with class_weight approximation and threshold adjustment.

# === 1. Imports and Setup ===
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# === 2. Load Preprocessed Supervised Dataset ===
df = load_supervised_modeling_dataset()

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X = df.drop(columns=["_Success_qual"] + colinear_vars, errors="ignore")
y = df["_Success_qual"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 4. Apply SMOTE to Training Data ===
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# === 5. Train XGBoost Classifier with Class Weight ===
# Simulate class weight effect via scale_pos_weight (class 0 is minority)
ratio = y_train_res.value_counts()[0] / y_train_res.value_counts()[1]
model = XGBClassifier(
    use_label_encoder=False, eval_metric="logloss", scale_pos_weight=ratio, random_state=42
)
model.fit(X_train_res, y_train_res)

# === 6. Predict and Evaluate with Threshold Adjustment ===
y_prob = model.predict_proba(X_test)[:, 1]
thresh = 0.6  # Adjust threshold
y_pred = (y_prob >= thresh).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["unsuccessful", "successful"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (XGBoost SMOTE + Class Weight + Threshold)")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (XGBoost SMOTE + Class Weight + Threshold)")
plt.legend()
plt.grid(True)
plt.show()

# === 7. Feature Importance ===
importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm")
plt.title("Feature Importance (XGBoost SMOTE + Class Weight + Threshold)")

# Annotate values on bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9, padding=3)

plt.tight_layout()
plt.show()

print("\nTop influencing features:")
print(importance_df.head())
