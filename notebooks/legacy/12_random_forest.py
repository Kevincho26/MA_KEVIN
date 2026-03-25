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

from src.data.loaders import load_supervised_modeling_dataset
from src.features.preprocessing import drop_columns_if_present, split_features_and_target
from src.models.evaluation import evaluate_binary_classifier
from src.models.interpretation import (
    plot_feature_importance,
    prepare_feature_importance_df,
)

# === 2. Load Preprocessed Supervised Dataset ===
df = load_supervised_modeling_dataset()

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 4. Train Basic Random Forest ===
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# === 5. Evaluate Performance ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (Random Forest Basic)",
    roc_title="ROC Curve (Random Forest Basic)",
)

# === 6. Feature Importance ===
importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)

plot_feature_importance(
    importance_df,
    title="Feature Importance (Random Forest Basic)",
)


# %% [markdown]
# ## 2. Random Forest: Class Weight Balanced

# %%
# === 1. Imports and Setup ===
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# === 2. Load Preprocessed Supervised Dataset ===
df = load_supervised_modeling_dataset()

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 4. Train Random Forest with Class Weight Balanced ===
model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight="balanced")
model.fit(X_train, y_train)

# === 5. Evaluate Performance ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (Random Forest v2 - Balanced)",
    roc_title="ROC Curve (Random Forest v2 - Balanced)",
)

# === 6. Feature Importance ===
importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)

plot_feature_importance(
    importance_df,
    title="Feature Importance (Random Forest v2 - Balanced)",
)


# %% [markdown]
# ## 3. Random Forest: SMOTE

# %%
# === 1. Imports and Setup ===
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# === 2. Load Preprocessed Supervised Dataset ===
df = load_supervised_modeling_dataset()

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)

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

auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (Random Forest SMOTE)",
    roc_title="ROC Curve (Random Forest SMOTE)",
)

# === 7. Feature Importance ===
importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)

plot_feature_importance(
    importance_df,
    title="Feature Importance (Random Forest SMOTE)",
)


# %% [markdown]
# ## 4. Random Forest: SMOTE + Class Weight + Threshold

# %%
# This version applies SMOTE to oversample the minority class, uses class_weight='balanced' in RandomForestClassifier, and adjusts the decision threshold.


# === 1. Imports and Setup ===

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# === 2. Load Preprocessed Supervised Dataset ===
df = load_supervised_modeling_dataset()

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)

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
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (Random Forest SMOTE + Class Weight + Threshold)",
    roc_title="ROC Curve (Random Forest SMOTE + Class Weight + Threshold)",
)

# === 8. Feature Importance ===
importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)

plot_feature_importance(
    importance_df,
    title="Feature Importance (Random Forest SMOTE + Class Weight + Threshold)",
)
