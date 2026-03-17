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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# === 2. Load Preprocessed Supervised Dataset ===
csv_path = r"C:\\Master Thesis Repositories\\MAChoque\\data\\processed\\preprocessed_supervised.csv"
df = pd.read_csv(csv_path)

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X = df.drop(columns=["_Success_qual"] + colinear_vars, errors="ignore")
y = df["_Success_qual"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 4. Train Basic Decision Tree ===
model = DecisionTreeClassifier(random_state=42)
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
plt.title("Confusion Matrix (Decision Tree Basic)")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Decision Tree Basic)")
plt.legend()
plt.grid(True)
plt.show()

# === 6. Feature Importance ===
importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm")
plt.title("Feature Importance (Decision Tree Basic)")

# Annotate values on bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9, padding=3)

plt.tight_layout()
plt.show()

print("\nTop influencing features:")
print(importance_df.head())

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
from sklearn.tree import DecisionTreeClassifier, plot_tree

# === 2. Load Preprocessed Supervised Dataset ===
csv_path = r"C:\\Master Thesis Repositories\\MAChoque\\data\\processed\\preprocessed_supervised.csv"
df = pd.read_csv(csv_path)

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X = df.drop(columns=["_Success_qual"] + colinear_vars, errors="ignore")
y = df["_Success_qual"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 4. Train Balanced & Pruned Decision Tree ===
model = DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=4)
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
plt.title("Confusion Matrix (Decision Tree v2)")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Decision Tree v2)")
plt.legend()
plt.grid(True)
plt.show()

# === 6. Feature Importance ===
importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm")
plt.title("Feature Importance (Decision Tree v2)")

# Annotate values on bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9, padding=3)

plt.tight_layout()
plt.show()

print("\nTop influencing features:")
print(importance_df.head())

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
from sklearn.tree import DecisionTreeClassifier, plot_tree

# === 2. Load Preprocessed Supervised Dataset ===
csv_path = r"C:\\Master Thesis Repositories\\MAChoque\\data\\processed\\preprocessed_supervised.csv"
df = pd.read_csv(csv_path)

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X = df.drop(columns=["_Success_qual"] + colinear_vars, errors="ignore")
y = df["_Success_qual"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 4. Train Balanced, Pruned & Tuned Decision Tree ===
model = DecisionTreeClassifier(
    random_state=42, class_weight="balanced", max_depth=4, min_samples_leaf=3
)
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
plt.title("Confusion Matrix (Decision Tree v3)")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Decision Tree v3)")
plt.legend()
plt.grid(True)
plt.show()

# === 6. Feature Importance ===
importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm")
plt.title("Feature Importance (Decision Tree v3)")

# Annotate values on bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9, padding=3)

plt.tight_layout()
plt.show()

print("\nTop influencing features:")
print(importance_df.head())

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
#

# %%
"""
Decision Tree Model (v3 - SMOTE Balanced)
-----------------------------------------------------
This version applies SMOTE to oversample the minority class before training,
to improve performance on imbalanced classification.
"""

# === 1. Imports and Setup ===
import matplotlib.pyplot as plt
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
from sklearn.tree import DecisionTreeClassifier, plot_tree

# === 2. Load Preprocessed Supervised Dataset ===
csv_path = r"C:\\Master Thesis Repositories\\MAChoque\\data\\processed\\preprocessed_supervised.csv"
df = pd.read_csv(csv_path)

# === 3. Prepare Features and Target ===
colinear_vars = ["_δND", "_share_RSE_internal"]
X = df.drop(columns=["_Success_qual"] + colinear_vars, errors="ignore")
y = df["_Success_qual"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 4. Apply SMOTE to Training Data ===
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# === 5. Train Decision Tree on SMOTE Data ===
model = DecisionTreeClassifier(random_state=42, max_depth=4)
model.fit(X_train_sm, y_train_sm)

# === 6. Evaluate Performance ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["unsuccessful", "successful"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Decision Tree SMOTE)")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Decision Tree SMOTE)")
plt.legend()
plt.grid(True)
plt.show()

# === 7. Feature Importance ===
importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm")
plt.title("Feature Importance (Decision Tree SMOTE)")

# Annotate values on bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9, padding=3)

plt.tight_layout()
plt.show()

print("\nTop influencing features:")
print(importance_df.head())

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
