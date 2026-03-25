from __future__ import annotations

from typing import Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_supervised_data(
    X,
    y,
    *,
    test_size: float = 0.3,
    random_state: int = 42,
    stratify: bool = True,
):
    """Split supervised data into train and test sets."""
    stratify_target = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )


def scale_train_test(
    X_train,
    X_test,
    *,
    scaler: StandardScaler | None = None,
) -> tuple[Any, Any, StandardScaler]:
    """Fit a scaler on train data and transform both train and test sets."""
    fitted_scaler = scaler or StandardScaler()
    X_train_scaled = fitted_scaler.fit_transform(X_train)
    X_test_scaled = fitted_scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, fitted_scaler
