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

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

from src.data.loaders import load_supervised_modeling_dataset
from src.features.preprocessing import drop_columns_if_present, split_features_and_target
from src.models.evaluation import evaluate_binary_classifier
from src.models.interpretation import plot_feature_importance, prepare_feature_importance_df
from src.models.splitting import split_supervised_data
from src.models.train import train_xgboost
from src.models.tuning import prepare_cv_results_df, tune_xgboost_random_search

df = load_supervised_modeling_dataset()

colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

model = train_xgboost(
    X_train,
    y_train,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (XGBoost Basic)",
    roc_title="ROC Curve (XGBoost Basic)",
)

importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (XGBoost Basic)",
)

# %% [markdown]
# ## 2. XGBoost: Random Search Tuning

# %%
class_counts = Counter(y_train)
scale_pos_weight = class_counts[0] / class_counts[1]

param_distributions = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
}

search = tune_xgboost_random_search(
    X_train,
    y_train,
    param_distributions=param_distributions,
    estimator_params={
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight,
    },
    n_iter=15,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    random_state=42,
)

print("Best params:", search.best_params_)
print("Best CV score:", search.best_score_)

cv_results_df = prepare_cv_results_df(search, top_n=10)
display(cv_results_df[["rank_test_score", "mean_test_score", "std_test_score", "params"]])

model = search.best_estimator_

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (XGBoost Tuned)",
    roc_title="ROC Curve (XGBoost Tuned)",
)

importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (XGBoost Tuned)",
)

# %%
# This version uses class_weight approximation and adjusts the decision threshold to improve detection of the minority class.

df = load_supervised_modeling_dataset()

colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
model = train_xgboost(
    X_train,
    y_train,
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=ratio,
    random_state=42,
)

threshold = 0.6
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (XGBoost Balanced + Threshold)",
    roc_title="ROC Curve (XGBoost Balanced + Threshold)",
)

importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (XGBoost Balanced + Threshold)",
)

# %% [markdown]
# ## 3. XGBoost: SMOTE

# %%
from imblearn.over_sampling import SMOTE

df = load_supervised_modeling_dataset()

colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

model = train_xgboost(
    X_train_res,
    y_train_res,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (XGBoost SMOTE)",
    roc_title="ROC Curve (XGBoost SMOTE)",
)

importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (XGBoost SMOTE)",
)

# %% [markdown]
# ## 4. XGBoost: SMOTE + Class Weight + Threshold

# %%
df = load_supervised_modeling_dataset()

colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

ratio = y_train_res.value_counts()[0] / y_train_res.value_counts()[1]
model = train_xgboost(
    X_train_res,
    y_train_res,
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=ratio,
    random_state=42,
)

y_prob = model.predict_proba(X_test)[:, 1]
thresh = 0.6
y_pred = (y_prob >= thresh).astype(int)
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (XGBoost SMOTE + Class Weight + Threshold)",
    roc_title="ROC Curve (XGBoost SMOTE + Class Weight + Threshold)",
)

importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (XGBoost SMOTE + Class Weight + Threshold)",
)
