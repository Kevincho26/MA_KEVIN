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
# ## 1. Decision Tree: baseline

# %%
"""
Decision Tree Model (v1 - Basic)
----------------------------------
This version uses a default Decision Tree Classifier without any class balancing or pruning.
The goal is to understand natural decision boundaries and extract interpretable rules.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.tree import plot_tree

from src.data.loaders import load_supervised_modeling_dataset
from src.features.preprocessing import drop_columns_if_present, split_features_and_target
from src.models.evaluation import evaluate_binary_classifier
from src.models.interpretation import plot_feature_importance, prepare_feature_importance_df
from src.models.splitting import split_supervised_data
from src.models.train import train_decision_tree
from src.models.tuning import prepare_cv_results_df, tune_decision_tree_grid_search

df = load_supervised_modeling_dataset()

colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

model = train_decision_tree(X_train, y_train, random_state=42)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (Decision Tree Basic)",
    roc_title="ROC Curve (Decision Tree Basic)",
)

importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (Decision Tree Basic)",
)

plt.figure(figsize=(18, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["unsuccessful", "successful"],
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.title("Decision Tree Structure (Basic)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Decision Tree: Balanced and Pruned

# %%
"""
Decision Tree Model (v2 - Balanced + Pruned)
---------------------------------------------
This version applies class_weight='balanced' and limits max_depth to reduce overfitting and address class imbalance.
"""

df = load_supervised_modeling_dataset()

colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

model = train_decision_tree(
    X_train,
    y_train,
    random_state=42,
    class_weight="balanced",
    max_depth=4,
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (Decision Tree v2)",
    roc_title="ROC Curve (Decision Tree v2)",
)

importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (Decision Tree v2)",
)

plt.figure(figsize=(18, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["unsuccessful", "successful"],
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.title("Decision Tree Structure (v2 - Pruned + Balanced)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Decision Tree: Grid Search Tuning

# %%
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, None],
    "min_samples_split": [2, 4, 6, 10],
    "min_samples_leaf": [1, 2, 3, 5],
}

search = tune_decision_tree_grid_search(
    X_train,
    y_train,
    param_grid=param_grid,
    estimator_params={
        "random_state": 42,
        "class_weight": "balanced",
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

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (Decision Tree Tuned)",
    roc_title="ROC Curve (Decision Tree Tuned)",
)

importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (Decision Tree Tuned)",
)

plt.figure(figsize=(18, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["unsuccessful", "successful"],
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.title("Decision Tree Structure (Tuned)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Decision Tree: SMOTE

# %%
from imblearn.over_sampling import SMOTE

df = load_supervised_modeling_dataset()

colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

model = train_decision_tree(X_train_sm, y_train_sm, random_state=42, max_depth=4)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (Decision Tree SMOTE)",
    roc_title="ROC Curve (Decision Tree SMOTE)",
)

importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (Decision Tree SMOTE)",
)

plt.figure(figsize=(18, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["unsuccessful", "successful"],
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.title("Decision Tree Structure (SMOTE)")
plt.tight_layout()
plt.show()
