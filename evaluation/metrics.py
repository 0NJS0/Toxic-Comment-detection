"""
Evaluation Metrics for Multi-Label Toxicity Classification
============================================================

This module computes all metrics used in the research project.

Metrics computed:
    - Accuracy (subset accuracy: all labels must match exactly)
    - Precision, Recall, F1 (macro and micro averaged)
    - ROC-AUC (per label and macro average)
    - Hamming Loss (fraction of wrong labels)
    - Loss (BCE loss)
    - Confusion Matrix (per-label, with CLI + image output)

Why macro vs micro F1?
    - Macro F1: average F1 per label (treats rare labels equally)
    - Micro F1: global F1 across all labels (dominated by common labels)
    - Since severe_toxic and threat are very rare (0.3-0.9%),
    - macro F1 will be lower but more informative.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    hamming_loss,
    confusion_matrix,
)
from typing import Dict, List, Optional
from pathlib import Path


def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute all classification metrics.

    Parameters
    ----------
    logits : np.ndarray
        Raw model outputs (before sigmoid), shape (N, num_labels)
    labels : np.ndarray
        Ground truth binary labels, shape (N, num_labels)

    Returns
    -------
    Dict[str, float]
        Dictionary of computed metrics
    """
    # Apply sigmoid to convert logits to probabilities
    probs = 1.0 / (1.0 + np.exp(-logits))

    # Threshold at 0.5 for binary predictions
    predictions = (probs >= 0.5).astype(np.float32)

    # --- Basic metrics ---
    accuracy = accuracy_score(labels, predictions)  # subset accuracy: all labels exact match

    # --- Per-label metrics (macro averaged: unweighted mean across labels) ---
    precision_macro = precision_score(labels, predictions, average="macro", zero_division=0)
    recall_macro = recall_score(labels, predictions, average="macro", zero_division=0)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)

    # --- Micro averaged (global: aggregate all TP/FP/FN across all labels) ---
    precision_micro = precision_score(labels, predictions, average="micro", zero_division=0)
    recall_micro = recall_score(labels, predictions, average="micro", zero_division=0)
    f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)

    # --- ROC-AUC (needs probabilities, not hard predictions) ---
    # ROC-AUC requires at least one positive and one negative in each label
    roc_auc_values = []
    for i in range(labels.shape[1]):
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
            roc_auc_values.append(auc)
        except ValueError:
            # Handle case where all labels are the same (e.g., all 0)
            roc_auc_values.append(0.5)
    roc_auc_macro = np.mean(roc_auc_values)

    # --- Hamming Loss (fraction of incorrectly predicted labels) ---
    ham_loss = hamming_loss(labels, predictions)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "roc_auc": float(roc_auc_macro),
        "hamming_loss": float(ham_loss),
        "subset_accuracy": float(accuracy),
    }


def compute_per_label_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    label_names: list,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each label individually.

    This is useful for understanding which labels the model
    handles well vs poorly.

    Parameters
    ----------
    logits : np.ndarray
        Raw logits, shape (N, num_labels)
    labels : np.ndarray
        Ground truth, shape (N, num_labels)
    label_names : list
        Names of the label columns

    Returns
    -------
    Dict[str, Dict[str, float]]
        {label_name: {precision, recall, f1, roc_auc}}
    """
    probs = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probs >= 0.5).astype(np.float32)

    per_label = {}
    for i, name in enumerate(label_names):
        f1 = f1_score(labels[:, i], predictions[:, i], zero_division=0)
        precision = precision_score(labels[:, i], predictions[:, i], zero_division=0)
        recall = recall_score(labels[:, i], predictions[:, i], zero_division=0)

        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            auc = 0.5

        per_label[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(auc),
            "support": int(labels[:, i].sum()),
        }

    return per_label


def generate_confusion_matrices(
    logits: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    save_path: Optional[Path] = None,
) -> None:
    """
    Generate per-label binary confusion matrices.

    For multi-label classification, each label gets its own 2x2
    confusion matrix (TN, FP, FN, TP). Results are printed to CLI
    and optionally saved as a PNG figure.

    Parameters
    ----------
    logits : np.ndarray
        Raw logits from the model, shape (N, num_labels)
    labels : np.ndarray
        Ground truth binary labels, shape (N, num_labels)
    label_names : List[str]
        Names of each label column
    save_path : Path, optional
        If provided, saves the confusion matrix figure to this path
    """
    probs = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probs >= 0.5).astype(np.int32)
    labels_int = labels.astype(np.int32)

    # Compute per-label confusion matrices
    cm_dict = {}
    for i, name in enumerate(label_names):
        cm = confusion_matrix(labels_int[:, i], predictions[:, i], labels=[0, 1])
        cm_dict[name] = cm

    # --- CLI display ---
    print("\n" + "=" * 70)
    print("CONFUSION MATRICES (Per-Label Binary)")
    print("=" * 70)
    print(f"{'Label':<20} {'TN':>6} {'FP':>6} {'FN':>6} {'TP':>6}  {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    for name, cm in cm_dict.items():
        tn, fp, fn, tp = cm.ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"{name:<20} {tn:>6} {fp:>6} {fn:>6} {tp:>6}  {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

    # --- Row-wise per-label matrices ---
    print("\nPer-Label Confusion Matrices:\n")
    for name, cm in cm_dict.items():
        tn, fp, fn, tp = cm.ravel()
        print(f"  [{name}]")
        print(f"                Predicted")
        print(f"                Neg    Pos")
        print(f"  Actual Neg   {tn:>5}  {fp:>5}")
        print(f"         Pos   {fn:>5}  {tp:>5}")
        print()

    # --- Image save ---
    if save_path is not None:
        _save_confusion_matrix_figure(cm_dict, label_names, save_path)


def _save_confusion_matrix_figure(
    cm_dict: dict,
    label_names: List[str],
    save_path: Path,
) -> None:
    """Save a multi-panel confusion matrix figure to disk."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n_labels = len(label_names)
    n_cols = 3
    n_rows = int(np.ceil(n_labels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    axes = axes.flatten()

    for i, name in enumerate(label_names):
        cm = cm_dict[name]
        tn, fp, fn, tp = cm.ravel()

        ax = axes[i]
        # Plot as a heatmap
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=cm.max() * 1.1 if cm.max() > 0 else 1)
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")

        # Add text annotations
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for row in range(2):
            for col in range(2):
                ax.text(col, row, f"{cm[row, col]:,}", ha="center", va="center",
                        color="white" if cm[row, col] > thresh else "black", fontsize=14, fontweight="bold")

        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_xticks([0, 1], ["Neg", "Pos"], fontsize=9)
        ax.set_yticks([0, 1], ["Neg", "Pos"], fontsize=9)

        # Add metrics as subtitle
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        ax.text(0.5, -0.15, f"P:{prec:.3f} R:{rec:.3f} F1:{f1:.3f}",
                transform=ax.transAxes, ha="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Per-Label Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix figure saved: {save_path}")
