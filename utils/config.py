"""
Configuration Loader
====================

Purpose:
    Load settings from the YAML configuration file into a Python dictionary.

Why a config file?
    Hardcoding values (batch size, learning rate, paths, etc.) makes it
    difficult to reproduce experiments. By keeping all settings in one
    YAML file, we can:
    - Run experiments with different settings without changing code
    - Track which settings produced which results
    - Share exact configurations with other researchers

Usage:
    from utils.config import load_config
    config = load_config("configs/config.yaml")
    batch_size = config["training"]["batch_size"]
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load the YAML configuration file and return it as a nested dictionary.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
        Defaults to "configs/config.yaml".

    Returns
    -------
    Dict[str, Any]
        Nested dictionary containing all configuration settings.

    Example
    -------
    >>> config = load_config()
    >>> print(config["training"]["learning_rate"])
    2e-05
    """
    config_file = Path(config_path)

    # Check if the config file exists before trying to load it
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found at: {config_file.absolute()}\n"
            f"Please ensure configs/config.yaml exists."
        )

    # Open and parse the YAML file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config
