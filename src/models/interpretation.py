from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def prepare_feature_importance_df(features, importances) -> pd.DataFrame:
    """Build and sort a feature-importance DataFrame."""
    return pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(
        by="Importance", ascending=False
    )


def plot_feature_importance(
    importance_df: pd.DataFrame,
    *,
    title: str,
    top_n: int = 5,
    figsize: tuple[int, int] = (10, 6),
    fmt: str = "%.2f",
) -> None:
    """Plot feature importance values and print the top features."""
    plt.figure(figsize=figsize)
    ax = sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm")
    plt.title(title)

    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, label_type="edge", fontsize=9, padding=3)

    plt.tight_layout()
    plt.show()

    print("\nTop influencing features:")
    print(importance_df.head(top_n))
