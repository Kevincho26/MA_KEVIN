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
from IPython.display import display

from src.data.loaders import load_supervised_modeling_dataset
from src.features.preprocessing import drop_columns_if_present, split_features_and_target
from src.models.evaluation import evaluate_binary_classifier
from src.models.interpretation import plot_coefficients, prepare_coefficient_df
from src.models.splitting import scale_train_test, split_supervised_data
from src.models.train import train_logistic_regression
from src.models.tuning import prepare_cv_results_df, tune_logistic_regression_grid_search

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
model = train_logistic_regression(
    X_train_scaled,
    y_train,
    class_weight="balanced",
    random_state=42,
    max_iter=1000,
)

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
coef_df = prepare_coefficient_df(X.columns, model.coef_[0])
plot_coefficients(
    coef_df,
    title="AQUI VA TU TITULO ACTUAL",
)

# %% [markdown]
# ## 2. Logistic Regression: Grid Search Tuning

# %%
param_grid = [
    {
        "penalty": ["l1"],
        "solver": ["liblinear"],
        "C": [0.01, 0.1, 1.0, 10.0],
    },
    {
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
        "C": [0.01, 0.1, 1.0, 10.0],
    },
]

search = tune_logistic_regression_grid_search(
    X_train_scaled,
    y_train,
    param_grid=param_grid,
    estimator_params={
        "class_weight": "balanced",
        "random_state": 42,
        "max_iter": 1000,
    },
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
)

print("Best params:", search.best_params_)
print("Best CV score:", search.best_score_)

cv_results_df = prepare_cv_results_df(search, top_n=10)
display(cv_results_df[["rank_test_score", "mean_test_score", "std_test_score", "params"]])

model = search.best_estimator_

# === Evaluate tuned model ===
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (Grid Search)",
    roc_title="ROC Curve (Grid Search)",
)

coef_df = prepare_coefficient_df(X.columns, model.coef_[0])
plot_coefficients(
    coef_df,
    title="AQUI VA TU TITULO ACTUAL",
)

# %% [markdown]
# ## 3. Logistic Regression: Weight class penalization

# %%
custom_weights = {0: 5, 1: 1}
model = train_logistic_regression(
    X_train_scaled,
    y_train,
    class_weight=custom_weights,
    random_state=42,
    max_iter=1000,
)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (Custom Weights: 0=5, 1=1)",
    roc_title="ROC Curve (Custom Weights: 0=5, 1=1)",
)

coef_df = prepare_coefficient_df(X.columns, model.coef_[0])
plot_coefficients(
    coef_df,
    title="AQUI VA TU TITULO ACTUAL",
)

# %% [markdown]
# ## 4. Logistic Regression: SMOTE

# %%
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

model = train_logistic_regression(
    X_train_resampled,
    y_train_resampled,
    random_state=42,
    max_iter=1000,
)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (SMOTE)",
    roc_title="ROC Curve (SMOTE)",
)

coef_df = prepare_coefficient_df(X.columns, model.coef_[0])
plot_coefficients(
    coef_df,
    title="AQUI VA TU TITULO ACTUAL",
)
