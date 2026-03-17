from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Define the relative path to the dataset
data_path = Path("../data/raw/250119_SGE.csv")

try:
    # Load the CSV file into a pandas DataFrame
    dataset = pd.read_csv(data_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    # Handle the case when the file is not found
    print("File not found. Please verify the relative path and your current working directory.")
except Exception as e:
    # Handle any other errors that might occur during file loading
    print(f"An error occurred: {e}")

# General summary of the dataset
print(dataset.info())

# Check unique values for each column
print(dataset.nunique())

# Check for missing values in the dataset
print(dataset.isnull().sum())

# Columns that may not be relevant for Logistic Regression
columns_to_drop = [
    "_Gen_ID",
    "_success_quant",
    "Unnamed: 0",
    "_Variant",
    "_Go_to_market",
]  # Add other irrelevant columns if needed

# Drop the irrelevant columns
dataset = dataset.drop(columns=columns_to_drop, axis=1)

# get the number of missing data points per column
missing_values_count = dataset.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:17]

# Drop rows with missing values in all columns
dataset = dataset.dropna()

# Display the number of remaining rows to confirm
print(f"Number of rows after dropping rows with missing values: {len(dataset)}")

# get the number of missing data points per column
missing_values_count = dataset.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:17]

# Step 1: Drop `_Gen_ID` for model training (but keep it for reference)
# gen_id_column = dataset['_Gen_ID']  # Save for reference later
# dataset.drop(['_Gen_ID'], axis=1, inplace=True)

# Step 2: Convert target variable (_Success_qual) to binary
# successful = 1, indifferent and unsuccessful = 0
dataset["_Success_qual"] = dataset["_Success_qual"].apply(lambda x: 1 if x == "successful" else 0)

# Step 3: One-Hot Encoding for `_Case_Study`
dataset = pd.get_dummies(dataset, columns=["_Case_Study"], prefix="Case")

# Step 4: Label Encoding for variation columns with hierarchical order
# Define label encoder
label_encoder = LabelEncoder()

# Encode variation columns with hierarchy: PV=2, AV=1, CV=0
variation_columns = [
    "_PP_Claim_Variation",
    "_Provider_Benefit_Variation",
    "_Customer_Benefit_Variation",
]
for col in variation_columns:
    dataset[col] = label_encoder.fit_transform(dataset[col])

# Step 5: Ensure `_Generation` and `_Go_to_market` remain numerical
dataset["_Generation"] = pd.to_numeric(dataset["_Generation"], errors="coerce")

# Display the first rows of the cleaned and encoded dataset
print(dataset.head())

from sklearn.preprocessing import StandardScaler

# Columns with percentages that need to be converted
percentage_columns = [
    "_δCV",
    "_δAV",
    "_δPV",
    "_δND",
    "_share_RSE_external",
    "_share_RSE_internal",
]

# Remove '%' and convert to float
for col in percentage_columns:
    dataset[col] = (
        dataset[col].str.rstrip("%").astype(float) / 100.0
    )  # Convert percentages to decimals

# Combine complementary variables and drop redundant ones
# Keep '_δAV' and '_δPV', drop '_δCV' and '_δND' since it is complementary
dataset.drop("_δCV", axis=1, inplace=True)
dataset.drop("_δND", axis=1, inplace=True)

# Similarly, keep only '_share_RSE_external' and drop '_share_RSE_internal'
dataset.drop("_share_RSE_internal", axis=1, inplace=True)

# Update the list of columns to scale, excluding dropped variables
scaled_columns = ["_δAV", "_δPV", "_share_RSE_external"]

# Apply standardization to the selected numerical columns
scaler = StandardScaler()
dataset[scaled_columns] = scaler.fit_transform(dataset[scaled_columns])

# Display the first rows of the dataset after normalization
print(dataset.head())

# Separate the target variable (_Success_qual) from the features
X = dataset.drop("_Success_qual", axis=1)  # Features (all columns except the target)
y = dataset["_Success_qual"]  # Target variable (binary)

# Save the cleaned dataset for future use
cleaned_data_path = "../data/processed/cleaned_datasetpy.csv"  # Update this path if necessary
dataset.to_csv(cleaned_data_path, index=False)

# Confirm the separation and save
print("Features and target variable separated.")
print(f"Cleaned dataset saved to: {cleaned_data_path}")

# Display shapes to confirm the structure
print("Features shape (X):", X.shape)
print("Target shape (y):", y.shape)

# EDA Analysis

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Resumen estadístico y tipos de datos
print("Dataset Info:")
print(dataset.info())  # Displays dataset structure and column types
print("\nSummary Statistics:")
print(dataset.describe())  # Displays summary statistics for numerical columns

# 2. Distribución de la variable objetivo
plt.figure(figsize=(6, 4))
y.value_counts().plot(
    kind="bar", color=["skyblue", "orange"]
)  # Bar plot for target variable distribution
plt.title("Class Distribution in Target Variable (_Success_qual)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# 3. Distribución de variables numéricas
# Updated numeric columns after removing `_share_RSE_internal`, `_δCV` and '_δND'
numeric_columns = ["_δAV", "_δPV", "_share_RSE_external"]

# Plot histograms for numerical columns
dataset[numeric_columns].hist(bins=20, figsize=(12, 10), color="teal")
plt.suptitle("Distributions of Numerical Features")
plt.show()

# 4. Heatmap de correlación
plt.figure(figsize=(10, 8))
# Compute correlation matrix only for numeric columns (to exclude categorical ones)
correlation_matrix = dataset[numeric_columns].corr()
sns.heatmap(
    correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
)  # Correlation heatmap
plt.title("Correlation Matrix")
plt.show()

# VIF Calculation for Multicollinearity

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Seleccionar las variables a analizar
features = [
    "_δAV",
    "_δPV",
    "_share_RSE_external",
]  # Incluir _δAV y _δPV junto con _share_RSE_external
X_vif = dataset[features]

# Calcular VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

# Mostrar los resultados del VIF
print("VIF values for selected features:")
print(vif_data)

# EDA Analysis Categpry Distribution. Analysis the freuqency of categorical variables

# Analyze categorical variables after encoding
# '_Case_Study' was one-hot encoded, so we'll handle it by analyzing its new columns
categorical_columns = [
    col
    for col in dataset.columns
    if col.startswith("Case_")
    or col
    in [
        "_PP_Claim_Variation",
        "_Provider_Benefit_Variation",
        "_Customer_Benefit_Variation",
    ]
]

for col in categorical_columns:
    print(f"Value counts for {col}:")
    print(dataset[col].value_counts())
    print("\n")

# Relationship between Categorical Variables and Target

import matplotlib.pyplot as plt
import pandas as pd

# 1. Distribución de categorías
# Categorical columns: One-hot encoded `Case_*` and other categorical variables
categorical_columns = [col for col in dataset.columns if col.startswith("Case_")] + [
    "_PP_Claim_Variation",
    "_Provider_Benefit_Variation",
    "_Customer_Benefit_Variation",
]

# Print value counts for each categorical variable
for col in categorical_columns:
    print(f"Value counts for {col}:")
    print(dataset[col].value_counts())
    print("\n")

# 2. Relación con la variable objetivo
# Analyze relationship between categorical variables and _Success_qual
for col in categorical_columns:
    if col.startswith("Case_"):
        # Since Case_* are binary (True/False), we'll group them
        cross_tab = pd.crosstab(dataset[col], dataset["_Success_qual"], normalize="index")
    else:
        # For other categorical variables with multiple levels
        cross_tab = pd.crosstab(dataset[col], dataset["_Success_qual"], normalize="index")

    print(f"Relationship between {col} and _Success_qual:")
    print(cross_tab)
    print("\n")

    # Plot relationship
    cross_tab.plot(kind="bar", stacked=True, figsize=(8, 4), color=["orange", "skyblue"])
    plt.title(f"Relationship between {col} and _Success_qual")
    plt.ylabel("Proportion")
    plt.xlabel(f"Categories of {col}")
    plt.legend(title="_Success_qual", labels=["0 (Not Successful)", "1 (Successful)"])
    plt.show()

# Consolidate low-frequency categories in Case_Study

# List of columns representing one-hot encoded `_Case_Study`
case_study_columns = [col for col in dataset.columns if col.startswith("Case_")]

# Define the threshold for low-frequency categories
low_freq_threshold = 10

# Sum the occurrences of each `_Case_Study` category
case_study_counts = dataset[case_study_columns].sum()

# Identify low-frequency categories
low_freq_categories = case_study_counts[case_study_counts < low_freq_threshold].index.tolist()

# Combine low-frequency categories into a single column "Case_Other"
dataset["Case_Other"] = dataset[low_freq_categories].sum(axis=1)

# Drop the original low-frequency one-hot encoded columns
dataset.drop(columns=low_freq_categories, inplace=True)

# Verify the result
print("Updated dataset with consolidated categories:")
print(dataset.head())

# Plot the distribution of the consolidated categories

# Verificar la distribución de las categorías consolidadas
print("Distribution of Case_Other:")
print(dataset["Case_Other"].value_counts())

# Relación entre Case_Other y _Success_qual
cross_tab = pd.crosstab(dataset["Case_Other"], dataset["_Success_qual"], normalize="index")
print("Relationship between Case_Other and _Success_qual:")
print(cross_tab)

# Visualizar la relación
cross_tab.plot(kind="bar", stacked=True, figsize=(8, 4), color=["orange", "green"])
plt.title("Relationship between Case_Other and _Success_qual")
plt.xlabel("Case_Other")
plt.ylabel("Proportion")
plt.legend(["Unsuccessful", "Successful"])
plt.show()

# Class Weigh to handle imbalanced data on the target variable

# Validar información de X
print("Información de X (Características):")
print(X.info())  # Tipos de datos y columnas

# Mostrar primeras filas de X para confirmar los datos
print("\nPrimeras filas de X:")
print(X.head())

# Validar distribución de la variable objetivo y
print("\nDistribución de la variable objetivo (y):")
print(y.value_counts())

# Verificar las primeras filas de y
print("\nPrimeras filas de y:")
print(y.head())

# Asegurarnos de que no hay valores faltantes en X ni en y
print("\nValores faltantes en X:")
print(X.isnull().sum())

print("\nValores faltantes en y:")
print(y.isnull().sum())

# Implementation of Logistic Regression Model

# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

# Dividir los datos nuevamente para asegurarnos de la alineación después de los cambios
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Confirmar tamaños de los nuevos conjuntos
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}, {y_train.shape}")
print(f"Tamaño del conjunto de prueba: {X_test.shape}, {y_test.shape}")

# Ajustar el modelo de regresión logística
logreg = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
logreg.fit(X_train, y_train)

# Predicciones
y_pred = logreg.predict(X_test)

# Evaluación del modelo
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(conf_matrix)

classification_rep = classification_report(y_test, y_pred)
print("\nReporte de clasificación:")
print(classification_rep)

# Coeficientes del modelo
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": logreg.coef_[0]}).sort_values(
    by="Coefficient", ascending=False
)

print("\nCoeficientes del modelo (importancia de las características):")
print(coefficients)

# Plot feature importance (coefficients)
plt.figure(figsize=(12, 6))
coefficients["abs_coef"] = abs(coefficients["Coefficient"])
coefficients = coefficients.sort_values("abs_coef", ascending=True)
plt.barh(coefficients["Feature"], coefficients["Coefficient"])
plt.title("Feature Importance in Logistic Regression Model")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()

# Print model accuracy
print("\nModel Performance Analysis:")
accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum()
print(f"Accuracy: {accuracy:.3f}")
print(f"True Positives: {conf_matrix[1, 1]}")
print(f"False Positives: {conf_matrix[0, 1]}")
print(f"True Negatives: {conf_matrix[0, 0]}")
print(f"False Negatives: {conf_matrix[1, 0]}")

# Calculate additional metrics
sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
print(f"\nSensitivity (True Positive Rate): {sensitivity:.3f}")
print(f"Specificity (True Negative Rate): {specificity:.3f}")

# Print top 5 most important features excluding Case_ prefixed features
print("\nTop 5 Most Influential Features:")
filtered_coef = coefficients[~coefficients["Feature"].str.startswith("Case_")]
top_features = filtered_coef.nlargest(5, "abs_coef")
print(top_features[["Feature", "Coefficient"]])
print("\nCoefficient Interpretation:")
print("- Positive coefficients indicate the feature increases the probability of success")
print("- Negative coefficients indicate the feature decreases the probability of success")
print("- The magnitude indicates how strongly the feature influences the prediction")
print(
    "- For each one-unit increase in the feature, the log odds of success change by the coefficient value"
)

# Example interpretation of most influential feature
most_influential = filtered_coef.iloc[-1]
print(f"\nExample interpretation for most influential feature '{most_influential['Feature']}':")
print(f"Coefficient value: {most_influential['Coefficient']:.3f}")
if most_influential["Coefficient"] > 0:
    print(f"A one standard deviation increase in {most_influential['Feature']} multiplies")
    print(f"the odds of success by {np.exp(most_influential['Coefficient']):.3f}")
else:
    print(f"A one standard deviation increase in {most_influential['Feature']} multiplies")
    print(
        f"the odds of success by {np.exp(abs(most_influential['Coefficient'])):.3f} in the opposite direction"
    )

    # Improve model performance with cross-validation and hyperparameter tuning

    # Cross-validation scores with current model
    cv_scores = cross_val_score(logreg, X, y, cv=5, scoring="balanced_accuracy")
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Enhanced grid search with class balancing and regularization parameters
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "class_weight": ["balanced", {0: 2, 1: 1}, None],  # Added custom weights
    }

    # Use balanced accuracy scoring to better handle imbalanced classes
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=2000),  # Increased max_iter
        param_grid,
        cv=5,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    # Print best parameters and score
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

    # Evaluate best model with multiple metrics
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    print("\nBest model classification report:")
    print(classification_report(y_test, y_pred_best))

    # ROC curve and precision-recall curve
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot both ROC and PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    ax1.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax1.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    ax2.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc:.2f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.show()
