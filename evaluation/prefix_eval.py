"""
Prefix Evaluation: Detection Delay Metrics
============================================

Metrics for *while-typing* early toxicity detection:

1. **Detection Delay** — how many additional tokens/characters are needed
   before the prefix prediction matches the full-text prediction.
2. **Prefix Accuracy** — accuracy at each prefix length.
3. **Agreement Rate** — how often the prefix prediction agrees with the
   full-text prediction at each threshold.
4. **Character Equivalent** — token-based prefix lengths converted to
   approximate characters (for UX reporting).

These metrics tell us: *how early can we reliably detect toxicity?*
"""

import numpy as np
from typing import Dict, List, Optional


def agreement_rate(
    full_preds: np.ndarray,
    prefix_preds: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Fraction of samples where the prefix prediction matches the
    full-text prediction (element-wise, per label).
    """
    full_bin = (full_preds >= threshold).astype(np.int32)
    prefix_bin = (prefix_preds >= threshold).astype(np.int32)
    return float(np.mean(full_bin == prefix_bin))


def detection_delay(
    prefix_results: Dict[str, Dict],
    threshold: float = 0.5,
    reference_key: str = "full",
) -> Dict[str, float]:
    """
    For each prefix length, compute the *gap* in F1-macro vs. the full-text
    reference. The "detection delay" is expressed as the extra tokens needed
    to close this gap.

    Returns a dict of delay metrics per prefix length.
    """
    ref = prefix_results[reference_key]
    ref_f1 = ref.get("f1_macro", 0.0)

    delays = {}
    for key, metrics in prefix_results.items():
        if key == reference_key:
            continue
        f1_gap = ref_f1 - metrics.get("f1_macro", 0.0)
        delays[key] = {
            "prefix_length_tokens": metrics.get("prefix_length_tokens"),
            "f1_macro": metrics.get("f1_macro", 0.0),
            "f1_gap": float(f1_gap),
            "agreement_rate": metrics.get("agreement_rate", 0.0),
            "accuracy": metrics.get("accuracy", 0.0),
        }

    return delays


def token_to_char_equivalent(
    prefix_length_tokens: Optional[int],
    avg_chars_per_token: float = 4.5,
) -> Optional[float]:
    """
    Convert token-based prefix length to estimated character count.

    For English text, tokens average ~4.5 characters (including spaces).
    This is used for UX reporting to say "we can detect toxicity after
    approximately N characters typed."
    """
    if prefix_length_tokens is None:
        return None
    return round(prefix_length_tokens * avg_chars_per_token)


def compute_prefix_metrics(
    full_logits: np.ndarray,
    prefix_logits_by_length: Dict[str, np.ndarray],
    labels: np.ndarray,
    label_cols: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict:
    """
    Comprehensive prefix evaluation.

    Parameters
    ----------
    full_logits : np.ndarray (n_samples, n_labels)
        Model predictions on full sequences.
    prefix_logits_by_length : dict of str -> np.ndarray
        Predictions at each prefix length, keyed by prefix_N.
    labels : np.ndarray (n_samples, n_labels)
        Ground truth.
    label_cols : list of str or None
        Label names for per-label reporting.
    threshold : float
        Classification threshold.

    Returns
    -------
    dict with agreement rates, delays, and per-length metrics.
    """
    # Apply sigmoid to get probabilities
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    full_probs = sigmoid(full_logits)
    results = {}

    for key, logits in prefix_logits_by_length.items():
        probs = sigmoid(logits)

        # Per-sample agreement with full-text prediction
        agr = agreement_rate(full_probs, probs, threshold)

        # Accuracy and F1
        from evaluation.metrics import compute_metrics
        metrics = compute_metrics(probs, labels)

        results[key] = {
            **metrics,
            "agreement_rate": agr,
            "char_equivalent": token_to_char_equivalent(
                int(key.split("_")[1]) if "_" in key else None
            ),
        }

    # Detection delays
    delays = detection_delay(results, threshold=threshold)

    return {
        "per_length": results,
        "delays": delays,
    }
