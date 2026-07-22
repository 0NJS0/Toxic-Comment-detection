#!/usr/bin/env python3
"""
================================================================================
MODULE 1: Data Preparation
================================================================================

WHAT THIS MODULE DOES
----------------------
This is the first module in the research pipeline. It:

1. Loads the raw Jigsaw Toxic Comment Classification dataset
2. Cleans the text by:
   - Removing URLs (they carry no toxicity signal)
   - Removing @mentions (usernames are not content)
   - Converting to lowercase (standard for DistilBERT)
   - Normalizing emojis (preserves emotional meaning as text)
   - Normalizing whitespace (consistent tokenization)
3. Removes duplicate comments
4. Saves the cleaned dataset to disk

WHY THIS MATTERS
----------------
Raw text data is noisy. Transformer models learn patterns from text,
so noise like URLs and inconsistent formatting forces them to waste
capacity on irrelevant patterns. Cleaning removes this noise so the
model can focus on actual toxic language.

HOW TO RUN
----------
    uv run python module_01_data_preparation.py

OUTPUT
------
- data/processed/train_cleaned.csv
- data/processed/test_cleaned.csv

Both files contain all original columns plus a new "clean_text" column.
================================================================================
"""

import sys
from pathlib import Path

# Add the project root to Python's path so we can import our modules
# This is needed because we're running from the project root
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from utils.config import load_config
from utils.helpers import set_random_seed, ensure_dir
from preprocessing.clean import run_data_preparation


def main():
    """
    Main entry point for Module 1: Data Preparation.

    Steps:
        1. Load configuration from configs/config.yaml
        2. Set random seed for reproducibility
        3. Ensure required directories exist
        4. Run the data preparation pipeline
    """
    print("=" * 60)
    print("Federated Edge-Based Early Toxicity Detection")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Step 1: Load configuration
    # ------------------------------------------------------------------
    # The configuration file contains ALL settings for this project.
    # We never hardcode values in Python files.
    # ------------------------------------------------------------------
    print("[1/4] Loading configuration...")
    config = load_config("configs/config.yaml")
    print(f"  Project: {config['project']['name']}")
    print(f"  Seed:    {config['project']['seed']}")
    print()

    # ------------------------------------------------------------------
    # Step 2: Set random seed
    # ------------------------------------------------------------------
    # This ensures that any randomness in our preprocessing is
    # reproducible. For example, if we later sample data randomly,
    # setting the seed means we'll get the same sample every time.
    # ------------------------------------------------------------------
    print("[2/4] Setting random seed...")
    set_random_seed(config["project"]["seed"])
    print()

    # ------------------------------------------------------------------
    # Step 3: Ensure output directories exist
    # ------------------------------------------------------------------
    # Before running any computation, make sure the directories
    # where we'll save results actually exist.
    # ------------------------------------------------------------------
    print("[3/4] Ensuring output directories exist...")
    ensure_dir(config["data"]["processed"]["path"])
    ensure_dir(config["results"]["plots"])
    ensure_dir(config["results"]["logs"])
    ensure_dir(config["results"]["reports"])
    print()

    # ------------------------------------------------------------------
    # Step 4: Run the data preparation pipeline
    # ------------------------------------------------------------------
    # This loads, cleans, and saves the dataset.
    # All the details are in preprocessing/clean.py.
    # ------------------------------------------------------------------
    print("[4/4] Running data preparation pipeline...")
    print()
    run_data_preparation(config)


if __name__ == "__main__":
    main()
