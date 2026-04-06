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
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.loaders import load_supervised_modeling_dataset
from src.features.preprocessing import drop_columns_if_present, split_features_and_target
from src.models.evaluation import evaluate_binary_classifier
from src.models.interpretation import plot_feature_importance, prepare_feature_importance_df
from src.models.splitting import split_supervised_data
from src.models.train import train_xgboost

# === 2. Load Preprocessed Supervised Dataset ===
df = load_supervised_modeling_dataset()

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

# === 4. Train XGBoost Classifier ===
model = train_xgboost(
    X_train,
    y_train,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)

# === 5. Predict and Evaluate ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (XGBoost Basic)",
    roc_title="ROC Curve (XGBoost Basic)",
)

# === 6. Feature Importance ===
importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (XGBoost Basic)",
)

# %% [markdown]
# ## 2. XGBoost: Balanced

# %%
# This version uses `scale_pos_weight` to handle class imbalance based on the training set distribution.

# === 2. Load Preprocessed Supervised Dataset ===
df = load_supervised_modeling_dataset()

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

# === 4. Compute Class Weight for Balance ===
class_counts = Counter(y_train)
scale_pos_weight = class_counts[0] / class_counts[1]

# === 5. Train XGBoost Classifier with Balance ===
model = train_xgboost(
    X_train,
    y_train,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=scale_pos_weight,
)

# === 6. Predict and Evaluate ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (XGBoost Balanced)",
    roc_title="ROC Curve (XGBoost Balanced)",
)

# === 7. Feature Importance ===
importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (XGBoost Balanced)",
)

# %%
# This version uses class_weight approximation and adjusts the decision threshold to improve detection of the minority class.

# === 2. Load Preprocessed Supervised Dataset ===
df = load_supervised_modeling_dataset()

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

# === 4. Train XGBoost Classifier with Class Weight Adjustment ===
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
model = train_xgboost(
    X_train,
    y_train,
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=ratio,
    random_state=42,
)

# === 5. Predict and Evaluate with Threshold Adjustment ===
threshold = 0.6  # Manually selected threshold
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (XGBoost Balanced + Threshold)",
    roc_title="ROC Curve (XGBoost Balanced + Threshold)",
)

# === 6. Feature Importance ===
importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (XGBoost Balanced + Threshold)",
)

# %% [markdown]
# ## 3. XGBoost: SMOTE

# %%
from imblearn.over_sampling import SMOTE

# This version applies SMOTE to balance the training data and trains an XGBoost classifier without using class weights or threshold adjustment.

# === 2. Load Preprocessed Supervised Dataset ===
df = load_supervised_modeling_dataset()

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

# === 4. Apply SMOTE to Training Data ===
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# === 5. Train XGBoost Classifier ===
model = train_xgboost(
    X_train_res,
    y_train_res,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)

# === 6. Predict and Evaluate ===
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (XGBoost SMOTE)",
    roc_title="ROC Curve (XGBoost SMOTE)",
)

# === 7. Feature Importance ===
importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (XGBoost SMOTE)",
)

# %% [markdown]
# ## 4. XGBoost: SMOTE + Class Weight + Threshold

# %%
# This version applies SMOTE to balance the training data and trains an XGBoost classifier with class_weight approximation and threshold adjustment.

# === 2. Load Preprocessed Supervised Dataset ===
df = load_supervised_modeling_dataset()

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

# === 4. Apply SMOTE to Training Data ===
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# === 5. Train XGBoost Classifier with Class Weight ===
# Simulate class weight effect via scale_pos_weight (class 0 is minority)
ratio = y_train_res.value_counts()[0] / y_train_res.value_counts()[1]
model = train_xgboost(
    X_train_res,
    y_train_res,
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=ratio,
    random_state=42,
)

# === 6. Predict and Evaluate with Threshold Adjustment ===
y_prob = model.predict_proba(X_test)[:, 1]
thresh = 0.6  # Adjust threshold
y_pred = (y_prob >= thresh).astype(int)
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (XGBoost SMOTE + Class Weight + Threshold)",
    roc_title="ROC Curve (XGBoost SMOTE + Class Weight + Threshold)",
)

# === 7. Feature Importance ===
importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (XGBoost SMOTE + Class Weight + Threshold)",
)
