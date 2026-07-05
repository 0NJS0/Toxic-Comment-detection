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

Why macro vs micro F1?
    - Macro F1: average F1 per label (treats rare labels equally)
    - Micro F1: global F1 across all labels (dominated by common labels)
    Since severe_toxic and threat are very rare (0.3-0.9%),
    macro F1 will be lower but more informative.
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
from typing import Dict


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
