"""
Helper Utilities
=================

Purpose:
    Shared utility functions used across multiple modules.

    Keeping these in one place prevents code duplication and makes
    the project more maintainable.

Current functions:
    - ensure_dir(path): Creates a directory if it doesn't exist
    - set_random_seed(seed): Sets seeds for reproducibility
    - get_timestamp(): Returns a timestamp string for file naming
"""

import os
import random
import numpy as np
from pathlib import Path
from datetime import datetime


def ensure_dir(path: str) -> Path:
    """
    Create a directory if it does not already exist.

    This is used extensively throughout the project to ensure
    output directories exist before saving files.

    Parameters
    ----------
    path : str
        Path to the directory to create.

    Returns
    -------
    Path
        The path object for the created/existing directory.

    Example
    -------
    >>> save_dir = ensure_dir("results/plots")
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Why is this necessary?
        Random number generators produce different numbers each run
        unless we fix the seed. Without this, you cannot reproduce
        results exactly.

    Parameters
    ----------
    seed : int
        The random seed value (e.g., 42).
    """
    random.seed(seed)
    np.random.seed(seed)

    # We'll add torch seed later when we import torch in Module 3
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Ensures deterministic behavior on GPU (slower but reproducible)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed yet


def get_timestamp() -> str:
    """
    Return a timestamp string for use in filenames.

    This ensures that logs, checkpoints, and reports have
    unique, sortable names.

    Returns
    -------
    str
        Timestamp in format: YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_label_columns(config) -> list:
    """
    Extract the list of label column names from the config.

    This helper exists so we don't have to hardcode label names
    in every module.

    Parameters
    ----------
    config : dict
        The project configuration dictionary.

    Returns
    -------
    list
        List of label column names, e.g.,
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    """
    return config["data"]["labels"]
