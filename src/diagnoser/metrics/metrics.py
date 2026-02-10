from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def classification_error(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)


def regression_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def classification_metrics(y_true, y_pred, y_score=None, positive_label=1):
    """
    Compute standard classification metrics for imbalanced datasets.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted class labels.
        y_score: Continuous score/probability for the positive class (optional).
        positive_label: Label for the positive class.

    Returns:
        dict of metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "error": classification_error(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    if y_score is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        except ValueError:
            metrics["roc_auc"] = None

        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_score)
        except ValueError:
            metrics["pr_auc"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics