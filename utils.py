from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix


def threshold_sweep(y_true: np.ndarray, proba: np.ndarray, num=50) -> np.ndarray:
    '''Sweep through thresholds and compute metrics'''
    thresholds = np.linspace(0.0, 1.0, num=num)
    rows = []

    for t in thresholds:
        y = (proba >= t).astype(int)
        tp = int(((y_true == 1) & (y == 1)).sum())
        fp = int(((y_true == 0) & (y == 1)).sum())
        fn = int(((y_true == 1) & (y == 0)).sum())

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        rows.append([t, prec, rec, f1])

    return np.array(rows)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    '''Plot confusion matrix'''
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # Add text annotations
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, f"{z}", ha="center", va="center", color="white" if z > cm.max()/2 else "black")

    plt.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray, proba: np.ndarray):
    '''Plot ROC curve'''
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")

    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_pr_curve(y_true: np.ndarray, proba: np.ndarray):
    '''Plot Precision-Recall curve'''
    prec, rec, _ = precision_recall_curve(y_true, proba)
    pr_auc = auc(rec, prec)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"AUC = {pr_auc:.3f}", linewidth=2)

    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_threshold_sweep(table: np.ndarray):
    '''Plot threshold sweep results'''
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(table[:, 0], table[:, 1], label="Precision", linewidth=2, marker='o', markersize=3)
    ax.plot(table[:, 0], table[:, 2], label="Recall", linewidth=2, marker='s', markersize=3)
    ax.plot(table[:, 0], table[:, 3], label="F1-Score", linewidth=2, marker='^', markersize=3)

    ax.set_title("Threshold Sweep Analysis")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig
