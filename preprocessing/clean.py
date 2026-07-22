"""
Data Cleaning Module
=====================

Module 1: Raw Data → Cleaned Data

What this module does:
    1. Loads the raw Jigsaw CSV files
    2. Removes duplicate comments
    3. Removes URLs (they carry no toxicity signal) [case-insensitive]
    4. Removes @mentions (usernames are not content) [case-insensitive]
    5. Converts emojis to text descriptions (preserves emotional meaning)
    6. Converts text to lowercase (standard for DistilBERT, also catches
       capitalized text from emoji demojization like :Statue_of_Liberty:)
    7. Normalizes whitespace (consistent tokenization)
    8. Saves the cleaned dataset

Why these cleaning steps?
    Transformer models like DistilBERT learn patterns from text.
    Noise like URLs, mentions, and inconsistent casing forces the
    model to waste capacity on irrelevant patterns. Cleaning
    removes this noise so the model focuses on actual toxic language.

Multi-label classification note:
    The Jigsaw dataset has 6 binary labels per comment. A comment
    can be both "toxic" AND "obscene" simultaneously. This is
    different from multi-class where each sample has ONE label.
"""

import re
import pandas as pd
import emoji
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# Individual cleaning functions
# ---------------------------------------------------------------------------
# Each function does ONE thing. This makes the code easy to understand,
# test, and modify. You can add/remove cleaning steps without
# affecting the rest of the pipeline.

def remove_urls(text: str) -> str:
    """
    Remove URLs (http://..., https://..., www.xxx...) from text.

    Why remove URLs?
        URLs are references to web pages, not natural language.
        They don't help the model detect toxicity. A URL like
        "http://example.com" could appear in both toxic and
        non-toxic comments equally.

    The regex pattern matches:
        - http:// or https:// followed by non-whitespace characters
        - www. followed by non-whitespace characters
    """
    if not isinstance(text, str):
        return ""

    # Pattern explanation:
    #   r'https?://\S+'  - matches http:// or https:// followed by non-spaces
    #   |                - OR
    #   r'www\.\S+'      - matches www. followed by non-spaces
    #
    # We use re.IGNORECASE because URLs can have mixed case:
    #   "Www.Example.Com" should be treated the same as "www.example.com"
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', text, flags=re.IGNORECASE)


def remove_mentions(text: str) -> str:
    """
    Remove @username mentions from text.

    Why remove mentions?
        @mentions are user identifiers (e.g., @john_doe).
        They don't contribute to the toxicity of the message.
        "I hate @user123" has the same toxicity as "I hate you".
    """
    if not isinstance(text, str):
        return ""

    # Pattern: @ symbol followed by non-whitespace characters
    # Case-insensitive to catch @User and @user uniformly
    mention_pattern = r'@\S+'
    return re.sub(mention_pattern, '', text, flags=re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple whitespace characters into a single space
    and strip leading/trailing whitespace.

    Why normalize whitespace?
        After removing URLs and mentions, we may have leftover
        spaces. "hello   world  " should become "hello world".
        Consistent whitespace ensures the tokenizer works correctly.
    """
    if not isinstance(text, str):
        return ""

    # Replace one or more whitespace chars (\s+) with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_emoji(text: str) -> str:
    """
    Convert emojis to their text descriptions using the `emoji` library.

    Example:
        "I am so happy 😊" → "I am so happy :smiling_face_with_smiling_eyes:"

    Why normalize emojis?
        Emojis carry emotional meaning that is relevant for toxicity
        detection. However, Transformer tokenizers don't handle
        emojis well (they get split into strange subwords).
        Converting to text like ":smiling_face:" preserves the
        emotional signal in a format the model understands.
    """
    if not isinstance(text, str):
        return ""

    # demojize converts emoji characters to :text: format
    return emoji.demojize(text)


def clean_text(
    text: str,
    remove_urls_flag: bool = True,
    remove_mentions_flag: bool = True,
    lowercase_flag: bool = True,
    normalize_emoji_flag: bool = True,
    normalize_whitespace_flag: bool = True,
) -> str:
    """
    Apply all cleaning operations to a single text string.

    ORDER MATTERS:
    1. Remove URLs first (they may contain @ symbols)
    2. Remove mentions second
    3. Normalize emojis third (convert emojis to text BEFORE lowercasing,
       because emoji text like ":Statue_of_Liberty:" may contain capitals)
    4. Lowercase fourth (catches any capitals from emoji demojization)
    5. Normalize whitespace last (cleans up artifacts from previous steps)

    Parameters
    ----------
    text : str
        The raw comment text to clean.
    remove_urls_flag : bool
        Whether to remove URLs.
    remove_mentions_flag : bool
        Whether to remove @mentions.
    lowercase_flag : bool
        Whether to convert to lowercase.
    normalize_emoji_flag : bool
        Whether to convert emojis to text.
    normalize_whitespace_flag : bool
        Whether to normalize whitespace.

    Returns
    -------
    str
        The cleaned text. Returns original text as fallback if
        result would otherwise be empty (e.g., URL-only comments).
    """
    if not isinstance(text, str):
        return ""

    if remove_urls_flag:
        text = remove_urls(text)

    if remove_mentions_flag:
        text = remove_mentions(text)

    # Emoji normalization BEFORE lowercasing:
    # The emoji library's demojize() preserves original capitalization
    # (e.g., 🗽 becomes ":Statue_of_Liberty:" not ":statue_of_liberty:")
    # If we lowercase first, the capitalized emoji text survives.
    # By demojizing first, then lowercasing, we catch everything.
    if normalize_emoji_flag:
        text = normalize_emoji(text)

    if lowercase_flag:
        text = text.lower()

    if normalize_whitespace_flag:
        text = normalize_whitespace(text)

    return text


# ---------------------------------------------------------------------------
# DataFrame-level operations
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame, subset: str = "comment_text") -> pd.DataFrame:
    """
    Remove duplicate comments from the DataFrame.

    Why remove duplicates?
        The Jigsaw dataset contains duplicate comments. If we keep
        them, the model will see the exact same comment multiple times,
        which:
        - Wastes training time
        - Biases the model toward repeated patterns
        - Inflates accuracy metrics artificially

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    subset : str
        Column name to check for duplicates.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed.
    """
    before = len(df)
    df = df.drop_duplicates(subset=[subset], keep="first")
    after = len(df)
    print(f"  Removed {before - after:,} duplicate comments ({after:,} remaining)")
    return df


# ---------------------------------------------------------------------------
# Main data loading and processing pipeline
# ---------------------------------------------------------------------------

def load_raw_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw training and test CSV files from disk.

    The training set has columns:
        id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

    The test set has columns:
        id, comment_text
    (No labels - the test set is for Kaggle submission)

    Since the test set has no labels, we will later split the
    training set into train/validation for our experiments.

    Parameters
    ----------
    config : dict
        The project configuration dictionary.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (training_dataframe, test_dataframe)
    """
    raw_config = config["data"]["raw"]
    raw_path = Path(raw_config["path"])
    train_file = raw_config["train_file"]
    test_file = raw_config["test_file"]

    train_path = raw_path / train_file
    test_path = raw_path / test_file

    print("=" * 60)
    print("MODULE 1: DATA PREPARATION")
    print("=" * 60)
    print(f"\nLoading training data from: {train_path}")
    df_train = pd.read_csv(train_path)

    print(f"Loading test data from: {test_path}")
    df_test = pd.read_csv(test_path)

    print(f"\nDataset sizes:")
    print(f"  Training samples: {len(df_train):,}")
    print(f"  Test samples:     {len(df_test):,}")

    # Display label distribution to understand class balance
    label_cols = config["data"]["labels"]
    print(f"\nLabel columns: {label_cols}")
    print(f"\nTraining label distribution (positive samples):")
    for col in label_cols:
        positive = df_train[col].sum()
        percentage = (positive / len(df_train)) * 100
        print(f"  {col:20s}: {int(positive):6,} ({percentage:.2f}%)")

    return df_train, df_test


def clean_dataframe(
    df: pd.DataFrame,
    config: dict,
    is_test: bool = False,
) -> pd.DataFrame:
    """
    Clean all comments in a DataFrame using the cleaning pipeline.

    This function applies `clean_text()` to every row in the
    `comment_text` column. For the training set, we also
    remove duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to clean.
    config : dict
        Configuration containing preprocessing flags.
    is_test : bool
        Whether this is the test set (don't remove duplicates).

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame.
    """
    preproc_config = config["preprocessing"]
    label_cols = config["data"]["labels"]

    print(f"\nCleaning text data...")

    # Apply the cleaning pipeline to every comment
    # We use .apply() with axis=1 to process each row
    # lambda x: clean_text(x["comment_text"], ...) applies the
    # cleaning function to the comment_text column of each row
    df["clean_text"] = df.apply(
        lambda row: clean_text(
            text=row["comment_text"],
            remove_urls_flag=preproc_config["remove_urls"],
            remove_mentions_flag=preproc_config["remove_mentions"],
            lowercase_flag=preproc_config["lowercase"],
            normalize_emoji_flag=preproc_config["normalize_emoji"],
            normalize_whitespace_flag=preproc_config["normalize_whitespace"],
        ),
        axis=1,  # apply the function to each row (axis=1 = rows)
    )

    # Handle edge case: comments that become empty after cleaning
    # (e.g., comments that were ONLY a URL). We fill with a
    # placeholder so pandas doesn't treat them as NaN in CSV.
    empty_count = (df["clean_text"].str.strip() == "").sum()
    if empty_count > 0:
        print(f"  Note: {empty_count} comments became empty after cleaning (filling with '[no_text]')")
        df["clean_text"] = df["clean_text"].replace(
            r"^\s*$", "[no_text]", regex=True
        )

    print(f"  Cleaned {len(df):,} comments")

    # Remove duplicates (only for training set)
    if not is_test and preproc_config["remove_duplicates"]:
        df = remove_duplicates(df, subset="clean_text")

    return df


def compute_label_stats(df: pd.DataFrame, label_cols: list) -> dict:
    """
    Compute basic statistics about label distribution.

    This helps us understand:
    - How many comments have each label
    - How many comments have multiple labels
    - The average number of labels per comment

    Parameters
    ----------
    df : pd.DataFrame
        The dataset with label columns.
    label_cols : list
        Names of the label columns.

    Returns
    -------
    dict
        Dictionary of statistics.
    """
    stats = {}

    # Number of positive samples for each label
    stats["label_counts"] = {col: int(df[col].sum()) for col in label_cols}

    # Number of labels per comment (sum across all 6 binary labels)
    label_sum = df[label_cols].sum(axis=1)
    stats["avg_labels_per_comment"] = float(label_sum.mean())
    stats["max_labels_per_comment"] = int(label_sum.max())

    # How many comments have at least one toxic label?
    stats["toxic_comments"] = int((label_sum > 0).sum())
    stats["clean_comments"] = int((label_sum == 0).sum())

    return stats


def save_processed_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    config: dict,
) -> None:
    """
    Save the cleaned datasets to disk.

    We save both the original fields and the new clean_text column.
    This allows later modules (tokenization, training) to load
    the data without re-cleaning.

    Parameters
    ----------
    df_train : pd.DataFrame
        Cleaned training data.
    df_test : pd.DataFrame
        Cleaned test data.
    config : dict
        Configuration containing output paths.
    """
    proc_config = config["data"]["processed"]
    output_path = Path(proc_config["path"])
    train_file = proc_config["train_file"]
    test_file = proc_config["test_file"]

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    train_out = output_path / train_file
    test_out = output_path / test_file

    print(f"\nSaving cleaned data:")
    print(f"  Training: {train_out}")
    print(f"  Test:     {test_out}")

    # Safety check: fill any remaining NaN in clean_text before saving
    # (shouldn't happen with our fix above, but just in case)
    df_train["clean_text"] = df_train["clean_text"].fillna("[no_text]")
    df_test["clean_text"] = df_test["clean_text"].fillna("[no_text]")

    # Save without index (pandas adds an index column by default)
    df_train.to_csv(train_out, index=False)
    df_test.to_csv(test_out, index=False)

    print("  [OK] Data saved successfully!")


# ---------------------------------------------------------------------------
# Main Module 1 pipeline
# ---------------------------------------------------------------------------

def run_data_preparation(config: dict) -> None:
    """
    Execute the full data preparation pipeline.

    This is the main entry point for Module 1. It:
    1. Loads raw data
    2. Cleans all text
    3. Removes duplicates
    4. Computes statistics
    5. Saves cleaned data
    6. Shows sample outputs

    Parameters
    ----------
    config : dict
        The project configuration dictionary.
    """
    # Step 1: Load raw data
    df_train, df_test = load_raw_data(config)

    # Step 2: Clean training data
    df_train_clean = clean_dataframe(df_train, config, is_test=False)

    # Step 3: Clean test data (no label columns, no duplicate removal)
    df_test_clean = clean_dataframe(df_test, config, is_test=True)

    # Step 4: Compute and display statistics
    label_cols = config["data"]["labels"]
    stats = compute_label_stats(df_train_clean, label_cols)

    print(f"\n" + "=" * 60)
    print("CLEANING RESULTS")
    print("=" * 60)
    print(f"\nLabel distribution after cleaning:")
    for label, count in stats["label_counts"].items():
        percentage = (count / len(df_train_clean)) * 100
        print(f"  {label:20s}: {count:6,} ({percentage:.2f}%)")
    print(f"\nMulti-label statistics:")
    print(f"  Avg labels per comment:    {stats['avg_labels_per_comment']:.2f}")
    print(f"  Max labels per comment:    {stats['max_labels_per_comment']}")
    print(f"  Toxic comments (any label): {stats['toxic_comments']:,}")
    print(f"  Clean comments (no label):  {stats['clean_comments']:,}")

    # Step 5: Show before/after examples
    print(f"\n" + "=" * 60)
    print("BEFORE vs AFTER CLEANING (Sample 3 comments)")
    print("=" * 60)
    sample_indices = [0, 100, 500]  # Pick some interesting examples
    for idx in sample_indices:
        if idx < len(df_train):
            print(f"\nSample #{idx}:")
            print(f"  BEFORE: {df_train.iloc[idx]['comment_text'][:100]}...")
            print(f"  AFTER:  {df_train_clean.iloc[idx]['clean_text'][:100]}...")

    # Step 6: Save processed data
    save_processed_data(df_train_clean, df_test_clean, config)

    print(f"\n" + "=" * 60)
    print("MODULE 1 COMPLETE")
    print("=" * 60)
    print(f"\nCleaned data is ready for Module 2 (Tokenization).")
