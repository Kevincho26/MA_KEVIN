from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def drop_columns_if_present(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    columns_to_drop = [column for column in columns if column in df.columns]
    return df.drop(columns=columns_to_drop).copy()


def split_features_and_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' was not found in the DataFrame.")

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()
    return X, y


def select_numeric_columns(
    df: pd.DataFrame,
    exclude: Iterable[str] | None = None,
) -> list[str]:
    excluded = set(exclude or [])
    return [
        column for column in df.select_dtypes(include=["number"]).columns if column not in excluded
    ]
