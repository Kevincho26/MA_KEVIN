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

# %%
"""
Random Forest Model (v1 - Basic)
----------------------------------
This version uses a default Random Forest classifier to evaluate base performance without class balancing or resampling.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display

from src.data.loaders import load_supervised_modeling_dataset
from src.features.preprocessing import drop_columns_if_present, split_features_and_target
from src.models.evaluation import evaluate_binary_classifier
from src.models.interpretation import plot_feature_importance, prepare_feature_importance_df
from src.models.splitting import split_supervised_data
from src.models.train import train_random_forest
from src.models.tuning import prepare_cv_results_df, tune_random_forest_random_search

df = load_supervised_modeling_dataset()

colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

model = train_random_forest(X_train, y_train, random_state=42)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (Random Forest Basic)",
    roc_title="ROC Curve (Random Forest Basic)",
)

importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (Random Forest Basic)",
)

# %% [markdown]
# ## 2. Random Forest: Random Search Tuning

# %%
param_distributions = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 4, 6, 8, 10],
    "min_samples_split": [2, 4, 6, 10],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_features": ["sqrt", "log2", None],
}

search = tune_random_forest_random_search(
    X_train,
    y_train,
    param_distributions=param_distributions,
    estimator_params={
        "random_state": 42,
        "class_weight": "balanced",
    },
    n_iter=20,
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
    confusion_matrix_title="Confusion Matrix (Random Forest Tuned)",
    roc_title="ROC Curve (Random Forest Tuned)",
)

importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (Random Forest Tuned)",
)

# %% [markdown]
# ## 3. Random Forest: SMOTE

# %%
from imblearn.over_sampling import SMOTE

df = load_supervised_modeling_dataset()

colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model = train_random_forest(
    X_train_resampled,
    y_train_resampled,
    random_state=42,
    n_estimators=100,
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (Random Forest SMOTE)",
    roc_title="ROC Curve (Random Forest SMOTE)",
)

importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (Random Forest SMOTE)",
)

# %% [markdown]
# ## 4. Random Forest: SMOTE + Class Weight + Threshold

# %%
df = load_supervised_modeling_dataset()

colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model = train_random_forest(
    X_train_resampled,
    y_train_resampled,
    random_state=42,
    n_estimators=100,
    class_weight="balanced",
)

y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.6
y_pred = (y_prob >= threshold).astype(int)

auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (Random Forest SMOTE + Class Weight + Threshold)",
    roc_title="ROC Curve (Random Forest SMOTE + Class Weight + Threshold)",
)

importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (Random Forest SMOTE + Class Weight + Threshold)",
)
