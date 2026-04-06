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

# === 1. Imports and Setup ===
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree

from src.data.loaders import load_supervised_modeling_dataset
from src.features.preprocessing import drop_columns_if_present, split_features_and_target
from src.models.evaluation import evaluate_binary_classifier
from src.models.interpretation import plot_feature_importance, prepare_feature_importance_df
from src.models.splitting import split_supervised_data
from src.models.train import train_decision_tree

# === 2. Load Preprocessed Supervised Dataset ===
df = load_supervised_modeling_dataset()

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

# === 4. Train Basic Decision Tree ===
model = train_decision_tree(X_train, y_train, random_state=42)

# === 5. Evaluate Performance ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (Decision Tree Basic)",
    roc_title="ROC Curve (Decision Tree Basic)",
)

# === 6. Feature Importance ===
importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (Decision Tree Basic)",
)

# === 7. Visualize Tree Structure ===
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
# Ver como se puede ajustar la grafica para la documentacion.

# %% [markdown]
# ## 2. Decision Tree: Balanced and Pruned

# %%
"""
Decision Tree Model (v2 - Balanced + Pruned)
---------------------------------------------
This version applies class_weight='balanced' and limits max_depth to reduce overfitting and address class imbalance.
"""

# === 2. Load Preprocessed Supervised Dataset ===
df = load_supervised_modeling_dataset()

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

# === 4. Train Balanced & Pruned Decision Tree ===
model = train_decision_tree(
    X_train,
    y_train,
    random_state=42,
    class_weight="balanced",
    max_depth=4,
)

# === 5. Evaluate Performance ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (Decision Tree v2)",
    roc_title="ROC Curve (Decision Tree v2)",
)

# === 6. Feature Importance ===
importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (Decision Tree v2)",
)

# === 7. Visualize Tree Structure ===
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

# %%
"""
Decision Tree Model (v3 - Balanced + Pruned + Tuned)
-----------------------------------------------------
This version applies class_weight='balanced', limits max_depth, and adds min_samples_leaf to reduce overfitting,
and help the tree better generalize, especially for minority class predictions.
"""

# === 2. Load Preprocessed Supervised Dataset ===
df = load_supervised_modeling_dataset()

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

# === 4. Train Balanced, Pruned & Tuned Decision Tree ===
model = train_decision_tree(
    X_train,
    y_train,
    random_state=42,
    class_weight="balanced",
    max_depth=4,
    min_samples_leaf=3,
)

# === 5. Evaluate Performance ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (Decision Tree v3)",
    roc_title="ROC Curve (Decision Tree v3)",
)

# === 6. Feature Importance ===
importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (Decision Tree v3)",
)

# === 7. Visualize Tree Structure ===
plt.figure(figsize=(18, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["unsuccessful", "successful"],
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.title("Decision Tree Structure (v3 - Pruned + Balanced + min_samples_leaf=3)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Decision Tree: SMOTE

# %%
"""
Decision Tree Model (v3 - SMOTE Balanced)
-----------------------------------------------------
This version applies SMOTE to oversample the minority class before training, to improve performance on imbalanced classification.
"""
from imblearn.over_sampling import SMOTE

# === 2. Load Preprocessed Supervised Dataset ===
df = load_supervised_modeling_dataset()

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X, y = split_features_and_target(df, "_Success_qual")
X = drop_columns_if_present(X, colinear_vars)
X_train, X_test, y_train, y_test = split_supervised_data(X, y)

# === 4. Apply SMOTE to Training Data ===
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# === 5. Train Decision Tree on SMOTE Data ===
model = train_decision_tree(X_train_sm, y_train_sm, random_state=42, max_depth=4)

# === 6. Evaluate Performance ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc_score = evaluate_binary_classifier(
    y_test,
    y_pred,
    y_prob,
    confusion_matrix_title="Confusion Matrix (Decision Tree SMOTE)",
    roc_title="ROC Curve (Decision Tree SMOTE)",
)

# === 7. Feature Importance ===
importance_df = prepare_feature_importance_df(X.columns, model.feature_importances_)
plot_feature_importance(
    importance_df,
    title="Feature Importance (Decision Tree SMOTE)",
)

# === 8. Visualize Tree Structure ===
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

# %%
