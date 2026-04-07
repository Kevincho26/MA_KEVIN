from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier


def tune_estimator_grid_search(
    estimator,
    X_train,
    y_train,
    param_grid,
    *,
    scoring: str = "roc_auc",
    cv: int = 5,
    n_jobs: int = -1,
    refit: bool = True,
    verbose: int = 0,
    return_train_score: bool = False,
) -> GridSearchCV:
    """Fit a GridSearchCV object and return the fitted search."""
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=refit,
        verbose=verbose,
        return_train_score=return_train_score,
    )
    search.fit(X_train, y_train)
    return search


def tune_estimator_random_search(
    estimator,
    X_train,
    y_train,
    param_distributions,
    *,
    n_iter: int = 25,
    scoring: str = "roc_auc",
    cv: int = 5,
    n_jobs: int = -1,
    refit: bool = True,
    random_state: int = 42,
    verbose: int = 0,
    return_train_score: bool = False,
) -> RandomizedSearchCV:
    """Fit a RandomizedSearchCV object and return the fitted search."""
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=refit,
        random_state=random_state,
        verbose=verbose,
        return_train_score=return_train_score,
    )
    search.fit(X_train, y_train)
    return search


def prepare_cv_results_df(
    search, *, top_n: int | None = None, sort_by: str = "rank_test_score"
) -> pd.DataFrame:
    """Return a tidy DataFrame from a fitted CV search object."""
    cv_results_df = pd.DataFrame(search.cv_results_).sort_values(by=sort_by)
    if top_n is not None:
        cv_results_df = cv_results_df.head(top_n)
    return cv_results_df.reset_index(drop=True)


def tune_logistic_regression_grid_search(
    X_train,
    y_train,
    param_grid,
    *,
    estimator_params: dict[str, Any] | None = None,
    scoring: str = "roc_auc",
    cv: int = 5,
    n_jobs: int = -1,
    refit: bool = True,
    verbose: int = 0,
    return_train_score: bool = False,
) -> GridSearchCV:
    """Run grid search for LogisticRegression."""
    estimator = LogisticRegression(**(estimator_params or {}))
    return tune_estimator_grid_search(
        estimator,
        X_train,
        y_train,
        param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=refit,
        verbose=verbose,
        return_train_score=return_train_score,
    )


def tune_decision_tree_grid_search(
    X_train,
    y_train,
    param_grid,
    *,
    estimator_params: dict[str, Any] | None = None,
    scoring: str = "roc_auc",
    cv: int = 5,
    n_jobs: int = -1,
    refit: bool = True,
    verbose: int = 0,
    return_train_score: bool = False,
) -> GridSearchCV:
    """Run grid search for DecisionTreeClassifier."""
    estimator = DecisionTreeClassifier(**(estimator_params or {}))
    return tune_estimator_grid_search(
        estimator,
        X_train,
        y_train,
        param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=refit,
        verbose=verbose,
        return_train_score=return_train_score,
    )


def tune_random_forest_random_search(
    X_train,
    y_train,
    param_distributions,
    *,
    estimator_params: dict[str, Any] | None = None,
    n_iter: int = 25,
    scoring: str = "roc_auc",
    cv: int = 5,
    n_jobs: int = -1,
    refit: bool = True,
    random_state: int = 42,
    verbose: int = 0,
    return_train_score: bool = False,
) -> RandomizedSearchCV:
    """Run randomized search for RandomForestClassifier."""
    estimator = RandomForestClassifier(**(estimator_params or {}))
    return tune_estimator_random_search(
        estimator,
        X_train,
        y_train,
        param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=refit,
        random_state=random_state,
        verbose=verbose,
        return_train_score=return_train_score,
    )


def tune_xgboost_random_search(
    X_train,
    y_train,
    param_distributions,
    *,
    estimator_params: dict[str, Any] | None = None,
    n_iter: int = 25,
    scoring: str = "roc_auc",
    cv: int = 5,
    n_jobs: int = -1,
    refit: bool = True,
    random_state: int = 42,
    verbose: int = 0,
    return_train_score: bool = False,
):
    """Run randomized search for XGBClassifier."""
    from xgboost import XGBClassifier

    estimator = XGBClassifier(**(estimator_params or {}))
    return tune_estimator_random_search(
        estimator,
        X_train,
        y_train,
        param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=refit,
        random_state=random_state,
        verbose=verbose,
        return_train_score=return_train_score,
    )


__all__ = [
    "prepare_cv_results_df",
    "tune_estimator_grid_search",
    "tune_estimator_random_search",
    "tune_logistic_regression_grid_search",
    "tune_decision_tree_grid_search",
    "tune_random_forest_random_search",
    "tune_xgboost_random_search",
]
