from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def evaluate_binary_classifier(
    y_true,
    y_pred,
    y_prob,
    *,
    confusion_matrix_title: str = "Confusion Matrix",
    roc_title: str = "ROC Curve",
    display_labels: list[str] | None = None,
) -> float:
    """Print standard binary-classification metrics and plot confusion matrix + ROC curve."""
    labels = display_labels or ["unsuccessful", "successful"]

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title(confusion_matrix_title)
    plt.show()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(roc_title)
    plt.legend()
    plt.grid(True)
    plt.show()

    return auc_score
