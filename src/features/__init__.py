"""Feature preparation utilities for the project."""

from src.features.engineering import (
    bin_numeric_column_to_indicators,
    expand_mapped_indicator_columns,
    one_hot_encode_columns,
)
from src.features.preprocessing import (
    drop_columns_if_present,
    select_numeric_columns,
    split_features_and_target,
)

__all__ = [
    "bin_numeric_column_to_indicators",
    "drop_columns_if_present",
    "expand_mapped_indicator_columns",
    "one_hot_encode_columns",
    "select_numeric_columns",
    "split_features_and_target",
    "encode_ordinal_columns",
]

from src.features.engineering import (
    encode_ordinal_columns,
)
