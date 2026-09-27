#!/usr/bin/env python3
"""
================================================================================
MODULE 2: Tokenization
================================================================================

WHAT THIS MODULE DOES
---------------------
1. Loads the cleaned dataset from Module 1 (data/processed/train_cleaned.csv)
2. Splits training data into 80% train / 10% validation / 10% test (stratified)
3. Loads the DistilBERT WordPiece tokenizer
4. Converts text → input_ids + attention_mask
5. Creates HuggingFace Dataset objects
6. Saves the tokenized dataset to data/processed/tokenized/

WHY THIS MATTERS
----------------
Transformers can't read text directly. Tokenization converts
each word and subword into an integer ID that the model's
embedding layer can process. This module produces the actual
numerical input that the model will train on.

HOW TO RUN
----------
    uv run python module_02_tokenization.py

OUTPUT
------
    data/processed/tokenized/
    ├── train/          # Tokenized training data (80%)
    ├── validation/     # Tokenized validation data (10%)
    ├── test/           # Tokenized test data (10%)
    ├── tokenizer/      # Saved tokenizer (for inference)
    ├── dataset_info.json
    └── stats.pkl       # Dataset statistics

INDEPENDENT EXECUTION
---------------------
This module loads the cleaned CSV from Module 1.
It does NOT need to be run after Module 1 in the same session.
================================================================================
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from utils.config import load_config
from utils.helpers import set_random_seed, ensure_dir
from preprocessing.tokenize import run_tokenization


def main():
    print("=" * 60)
    print("Federated Edge-Based Early Toxicity Detection")
    print("Module 2: Tokenization")
    print("=" * 60)
    print()

    # Step 1: Load configuration
    print("[1/3] Loading configuration...")
    config = load_config("configs/config.yaml")
    print(f"  Model:       {config['model']['name']}")
    print(f"  Max length:  {config['data']['max_seq_length']}")
    print()

    # Step 2: Set random seed
    print("[2/3] Setting random seed...")
    set_random_seed(config["project"]["seed"])
    print()

    # Step 3: Run tokenization pipeline
    print("[3/3] Running tokenization pipeline...")
    print()
    run_tokenization(config)


if __name__ == "__main__":
    main()
