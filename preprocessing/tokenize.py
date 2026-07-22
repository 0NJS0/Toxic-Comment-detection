"""
Tokenization Module
====================

Module 2: Cleaned Text → Tokenized Dataset

What this module does:
    1. Loads the cleaned dataset from Module 1
    2. Splits training data into train (80%), validation (10%), and test (10%)
    3. Loads a HuggingFace tokenizer (DistilBERT WordPiece)
    4. Tokenizes all comments (text → input_ids + attention_mask)
    5. Saves the tokenized dataset to disk

Why tokenization?
    Transformer models don't read text characters directly.
    They read integers called "token IDs." Tokenization converts
    words and subwords into these IDs.

    Example:
        "you are stupid" → [2017, 2024, 13193]

    The model looks up each ID in its embedding matrix to get
    a vector representation, then processes all vectors through
    its transformer layers.

Why split train/validation here?
    We split BEFORE training so that:
    - The validation set is never seen during training
    - Both train and val go through identical tokenization
    - Metrics on val set reflect true generalization

Vocabulary: WordPiece (DistilBERT)
    DistilBERT uses WordPiece tokenization with ~30,000 tokens.
    Common words are single tokens: "the" → [1996]
    Rare words are split into subwords: "toxicity" → ["tox", "##icity"]
    Unknown characters become [UNK] = [100]
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from transformers import AutoTokenizer
from datasets import Dataset as HFDataset, DatasetDict


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
# These match the DistilBERT tokenizer defaults.
# We define them here so they're easy to find and modify.

PAD_TOKEN_ID = 0      # DistilBERT uses token ID 0 for [PAD]
CLS_TOKEN_ID = 101    # [CLS] token marks the start of a sequence
SEP_TOKEN_ID = 102    # [SEP] token marks the end of a sequence


# ---------------------------------------------------------------------------
# Step 1: Load and split the cleaned data
# ---------------------------------------------------------------------------

def load_cleaned_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the cleaned dataset and split into train, validation, and test.

    We use a stratified split to preserve the label distribution
    across all splits. This is important because the dataset is
    highly imbalanced (~10% toxic).

    Split ratio: 80% train, 10% validation, 10% test.

    Parameters
    ----------
    config : dict
        Project configuration with data paths.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_dataframe, validation_dataframe, test_dataframe)
    """
    from sklearn.model_selection import train_test_split

    proc_config = config["data"]["processed"]
    data_path = Path(proc_config["path"]) / proc_config["train_file"]
    label_cols = config["data"]["labels"]

    print(f"Loading cleaned data from: {data_path}")
    df = pd.read_csv(data_path)

    print(f"Total samples: {len(df):,}")

    # Stratify on binary toxic/non-toxic to preserve class balance
    stratify_col = (df[label_cols].sum(axis=1) > 0).astype(int)

    # First split: 80% train, 20% temp (val + test)
    df_train, df_temp = train_test_split(
        df,
        test_size=0.2,
        random_state=config["project"]["seed"],
        stratify=stratify_col,
    )

    # Second split: split the 20% into 10% val + 10% test
    stratify_temp = (df_temp[label_cols].sum(axis=1) > 0).astype(int)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.5,
        random_state=config["project"]["seed"],
        stratify=stratify_temp,
    )

    print(f"\nSplit sizes:")
    print(f"  Training:     {len(df_train):,} ({100*len(df_train)/len(df):.1f}%)")
    print(f"  Validation:   {len(df_val):,} ({100*len(df_val)/len(df):.1f}%)")
    print(f"  Test:         {len(df_test):,} ({100*len(df_test)/len(df):.1f}%)")

    # Show that stratification preserved label balance
    print(f"\nStratification check (% toxic in each split):")
    for col in label_cols:
        train_pct = df_train[col].mean() * 100
        val_pct = df_val[col].mean() * 100
        test_pct = df_test[col].mean() * 100
        print(f"  {col:20s}: train={train_pct:.2f}%  val={val_pct:.2f}%  test={test_pct:.2f}%")

    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# Step 2: Load tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(config: dict) -> AutoTokenizer:
    """
    Load the HuggingFace tokenizer specified in the config.

    For DistilBERT, this loads the WordPiece tokenizer with
    ~30,000 tokens. The tokenizer handles:
    - Splitting text into tokens (words + subwords)
    - Mapping tokens to IDs
    - Adding special tokens ([CLS], [SEP], [PAD])
    - Padding and truncation

    Parameters
    ----------
    config : dict
        Project configuration with model name.

    Returns
    -------
    AutoTokenizer
        The loaded tokenizer.
    """
    model_name = config["model"]["name"]
    cache_dir = config["model"].get("cache_dir", "models/cache")

    print(f"\nLoading tokenizer: {model_name}")
    print("  (This downloads from HuggingFace on first run)")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    print(f"  Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"  Max length supported: {tokenizer.model_max_length}")

    return tokenizer


# ---------------------------------------------------------------------------
# Step 3: Tokenize function
# ---------------------------------------------------------------------------

def tokenize_texts(
    texts: list,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dict[str, np.ndarray]:
    """
    Tokenize a list of text strings.

    The tokenizer returns a dictionary with:
    - input_ids:      list of token IDs (integers)
    - attention_mask: list of 1/0 (1 = real token, 0 = padding)

    We use padding="max_length" so all sequences have the same
    length. This is required for batch processing on GPU.

    Parameters
    ----------
    texts : list
        List of text strings to tokenize.
    tokenizer : AutoTokenizer
        The HuggingFace tokenizer.
    max_length : int
        Maximum number of tokens per sequence.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with "input_ids" and "attention_mask" arrays.
    """
    print(f"  Tokenizing {len(texts):,} texts (max_length={max_length})...")

    # The tokenizer does everything in one call:
    # - Tokenizes each text into words/subwords
    # - Maps each token to its integer ID
    # - Adds [CLS] at the start and [SEP] at the end
    # - Pads shorter sequences to max_length
    # - Truncates longer sequences to max_length
    # - Creates attention_mask (1 for real tokens, 0 for padding)
    encoded = tokenizer(
        texts,
        padding="max_length",   # pad to max_length for batch processing
        truncation=True,        # cut sequences longer than max_length
        max_length=max_length,
        return_tensors="np",    # return numpy arrays (not PyTorch yet)
    )

    print(f"  input_ids shape:      {encoded['input_ids'].shape}")
    print(f"  attention_mask shape: {encoded['attention_mask'].shape}")

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


# ---------------------------------------------------------------------------
# Step 4: Create HuggingFace Dataset
# ---------------------------------------------------------------------------

def create_dataset(
    df: pd.DataFrame,
    encoded: Dict[str, np.ndarray],
    label_cols: list,
    is_test: bool = False,
) -> HFDataset:
    """
    Create a HuggingFace Dataset from tokenized data and labels.

    HuggingFace Datasets provide:
    - Memory-mapped storage (handles large datasets)
    - Easy integration with Trainer API
    - Lazy loading (doesn't load all data into RAM at once)

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame (for labels and IDs).
    encoded : Dict[str, np.ndarray]
        Tokenized input_ids and attention_mask.
    label_cols : list
        Names of the label columns.
    is_test : bool
        Whether this is test data (no labels).

    Returns
    -------
    HFDataset
        A HuggingFace Dataset ready for training.
    """
    data_dict = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "id": df["id"].values,
    }

    # Add labels if available (not for test set)
    if not is_test:
        labels = df[label_cols].values.astype(np.float32)
        data_dict["labels"] = labels
        print(f"  Labels shape: {labels.shape}")

    dataset = HFDataset.from_dict(data_dict)
    return dataset


# ---------------------------------------------------------------------------
# Step 5: Save tokenized dataset
# ---------------------------------------------------------------------------

def save_tokenized_dataset(
    dataset_dict: DatasetDict,
    config: dict,
) -> None:
    """
    Save the tokenized dataset to disk.

    We use HuggingFace Datasets' save_to_disk which saves:
    - The tokenized data (input_ids, attention_mask, labels)
    - Metadata (column names, dtypes)
    - Format (arrow format for fast loading)

    Later modules load with: datasets.load_from_disk(path)

    Parameters
    ----------
    dataset_dict : DatasetDict
        The tokenized dataset (train + validation + test).
    config : dict
        Project configuration with output paths.
    """
    proc_config = config["data"]["processed"]
    output_dir = Path(proc_config["path"]) / "tokenized"

    print(f"\nSaving tokenized dataset to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # HuggingFace's save_to_disk saves all data + metadata
    dataset_dict.save_to_disk(str(output_dir))

    print("  [OK] Tokenized dataset saved!")

    # Also save a small stats file for reference
    stats = {
        "train_size": len(dataset_dict["train"]),
        "validation_size": len(dataset_dict["validation"]),
        "test_size": len(dataset_dict["test"]),
        "num_labels": len(config["data"]["labels"]),
        "max_seq_length": config["data"]["max_seq_length"],
        "model_name": config["model"]["name"],
        "label_columns": config["data"]["labels"],
    }

    stats_path = output_dir / "stats.pkl"
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    print(f"  Stats saved to: {stats_path}")


# ---------------------------------------------------------------------------
# Step 6: Show sample tokenization
# ---------------------------------------------------------------------------

def show_sample(
    dataset: HFDataset,
    tokenizer: AutoTokenizer,
    idx: int = 0,
) -> None:
    """
    Decode and display a single tokenized sample.

    This helps verify that tokenization worked correctly by
    converting token IDs back to text and showing the labels.

    Parameters
    ----------
    dataset : HFDataset
        The tokenized dataset.
    tokenizer : AutoTokenizer
        The tokenizer (for decoding).
    idx : int
        Index of the sample to display.
    """
    sample = dataset[idx]

    # Decode token IDs back to readable text
    decoded = tokenizer.decode(
        sample["input_ids"],
        skip_special_tokens=True,  # don't show [CLS], [SEP], [PAD]
    )

    print(f"\nSample #{idx}:")
    print(f"  Decoded text: {decoded[:150]}...")
    print(f"  input_ids (first 20): {sample['input_ids'][:20]}")
    print(f"  attention_mask (first 20): {sample['attention_mask'][:20]}")

    if "labels" in sample:
        # Get label column names from the dataset
        print(f"  Labels: {sample['labels']}")


# ---------------------------------------------------------------------------
# Main Module 2 pipeline
# ---------------------------------------------------------------------------

def run_tokenization(config: dict) -> None:
    """
    Execute the full tokenization pipeline.

    Steps:
     1. Load cleaned data and split into train/validation/test
     2. Load the HuggingFace tokenizer
     3. Tokenize training texts
     4. Tokenize validation texts
     5. Tokenize test texts
     6. Create HuggingFace Datasets
     7. Show sample verification
     8. Save tokenized dataset to disk

    Parameters
    ----------
    config : dict
        The project configuration dictionary.
    """
    print("=" * 60)
    print("MODULE 2: TOKENIZATION")
    print("=" * 60)

    label_cols = config["data"]["labels"]
    max_length = config["data"]["max_seq_length"]

    # Step 1: Load and split data
    print("\n[1/7] Loading and splitting data...")
    df_train, df_val, df_test = load_cleaned_data(config)

    # Step 2: Load tokenizer
    print("\n[2/7] Loading tokenizer...")
    tokenizer = load_tokenizer(config)

    # Save tokenizer to tokenized folder for later use
    proc_path = Path(config["data"]["processed"]["path"])
    tokenizer_path = proc_path / "tokenized" / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_path))
    print(f"  Tokenizer saved to: {tokenizer_path}")

    # Step 3: Tokenize training data
    print("\n[3/7] Tokenizing training data...")
    train_texts = df_train["clean_text"].tolist()
    train_encoded = tokenize_texts(train_texts, tokenizer, max_length)

    # Step 4: Tokenize validation data
    print("\n[4/7] Tokenizing validation data...")
    val_texts = df_val["clean_text"].tolist()
    val_encoded = tokenize_texts(val_texts, tokenizer, max_length)

    # Step 5: Tokenize test data
    print("\n[5/7] Tokenizing test data...")
    test_texts = df_test["clean_text"].tolist()
    test_encoded = tokenize_texts(test_texts, tokenizer, max_length)

    # Step 6: Create datasets
    print("\n[6/7] Creating datasets...")
    train_dataset = create_dataset(df_train, train_encoded, label_cols, is_test=False)
    val_dataset = create_dataset(df_val, val_encoded, label_cols, is_test=False)
    test_dataset = create_dataset(df_test, test_encoded, label_cols, is_test=False)

    # Combine into a DatasetDict for easy access
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    })

    # Step 7: Show sample verification
    print("\n[7/7] Verification and saving...")
    print("\n" + "-" * 40)
    print("VERIFICATION: Sample tokenized outputs")
    print("-" * 40)
    show_sample(train_dataset, tokenizer, idx=0)
    show_sample(train_dataset, tokenizer, idx=100)

    # Check for any sequences that were truncated.
    # If the last token position (index -1) has a non-pad token,
    # the sequence was longer than max_length and got truncated.
    input_ids_array = np.array(train_dataset["input_ids"])
    truncated = (input_ids_array[:, -1] != PAD_TOKEN_ID).sum()
    if truncated > 0:
        print(f"\n  [WARN] {truncated:,} training sequences ({100*truncated/len(train_dataset):.1f}%) "
              f"were TRUNCATED (exceeded {max_length} tokens)")
    else:
        print(f"\n  [OK] No sequences exceeded {max_length} tokens")

    # Save
    save_tokenized_dataset(dataset_dict, config)

    print(f"\n" + "=" * 60)
    print("MODULE 2 COMPLETE")
    print("=" * 60)
    print(f"\nTokenized dataset ready for Module 3 (Training).")
