from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


def expand_mapped_indicator_columns(
    df: pd.DataFrame,
    source_column: str,
    value_map: Mapping[object, str],
    labels: Sequence[str] | None = None,
    drop_source: bool = True,
) -> pd.DataFrame:
    """Expand a categorical-coded source column into one indicator column per label."""
    if source_column not in df.columns:
        return df.copy()

    result = df.copy()
    mapped = result[source_column].map(value_map)

    indicator_labels = (
        list(labels) if labels is not None else list(dict.fromkeys(value_map.values()))
    )

    for label in indicator_labels:
        result[f"{source_column}_{label}"] = (mapped == label).astype(int)

    if drop_source:
        result = result.drop(columns=[source_column])

    return result


def bin_numeric_column_to_indicators(
    df: pd.DataFrame,
    source_column: str,
    bins: Sequence[float],
    labels: Sequence[str],
    prefix: str | None = None,
    drop_source: bool = True,
    right: bool = True,
    include_lowest: bool = False,
) -> pd.DataFrame:
    """Bin a numeric column and expand the resulting categories into indicator columns."""
    if source_column not in df.columns:
        return df.copy()

    if len(bins) != len(labels) + 1:
        raise ValueError("`bins` must contain exactly one more element than `labels`.")

    result = df.copy()
    numeric_series = pd.to_numeric(result[source_column], errors="coerce")
    categories = pd.cut(
        numeric_series,
        bins=bins,
        labels=labels,
        right=right,
        include_lowest=include_lowest,
    )

    column_prefix = prefix or source_column
    for label in labels:
        result[f"{column_prefix}_{label}"] = (categories == label).astype(int)

    if drop_source:
        result = result.drop(columns=[source_column])

    return result


def one_hot_encode_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    drop_source: bool = True,
) -> pd.DataFrame:
    """One-hot encode the requested columns and optionally drop the originals."""
    result = df.copy()

    for column in columns:
        if column not in result.columns:
            continue

        dummies = pd.get_dummies(result[column].astype(str), prefix=column, dtype=int)

        if drop_source:
            result = result.drop(columns=[column])

        result = pd.concat([result, dummies], axis=1)

    return result


def encode_ordinal_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    value_map: Mapping[object, int],
    coerce_to_string: bool = False,
) -> pd.DataFrame:
    """Encode multiple columns with the same ordinal mapping."""
    result = df.copy()

    for column in columns:
        if column not in result.columns:
            continue

        series = result[column].astype(str) if coerce_to_string else result[column]
        result[column] = series.map(value_map).astype(int)

    return result
