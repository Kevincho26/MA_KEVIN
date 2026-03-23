from __future__ import annotations

import pandas as pd

from src.utils.paths import PROCESSED_DIR


def load_base_dataset() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "base_dataset.csv")


def load_supervised_modeling_dataset() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "supervised_modeling_dataset.csv")


def load_unsupervised_base_dataset() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "unsupervised_base_dataset.csv")


def load_unsupervised_modeling_dataset() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "unsupervised_modeling_dataset.csv")


def load_association_rules_dataset() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "association_rules_dataset.csv")
