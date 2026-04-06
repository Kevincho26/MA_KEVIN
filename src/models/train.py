from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def train_logistic_regression(X_train, y_train, **model_params) -> LogisticRegression:
    """Instantiate and fit a logistic-regression classifier."""
    model = LogisticRegression(**model_params)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, **model_params) -> DecisionTreeClassifier:
    """Instantiate and fit a decision-tree classifier."""
    model = DecisionTreeClassifier(**model_params)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, **model_params) -> RandomForestClassifier:
    """Instantiate and fit a random-forest classifier."""
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, **model_params):
    """Instantiate and fit an XGBoost classifier."""
    from xgboost import XGBClassifier

    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train)
    return model


__all__ = [
    "train_logistic_regression",
    "train_decision_tree",
    "train_random_forest",
    "train_xgboost",
]
