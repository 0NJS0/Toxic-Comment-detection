"""
Data Partitioning for Federated Learning
=========================================

Two strategies for simulating heterogeneous clients:

1. **Dirichlet partitioning** (standard FL benchmark)
   - Samples client assignment from Dirichlet(alpha) per label
   - alpha → 0: each client gets 1-2 label types (extreme non-IID)
   - alpha → ∞: uniform IID distribution

2. **Topic-based partitioning** (realistic keyboard scenario)
   - Clusters comments by vocabulary (TF-IDF + K-means)
   - Each client corresponds to a topic cluster (gaming, politics, etc.)
   - Tests personalization on real language variation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict


def create_iid_partitions(
    df: pd.DataFrame,
    num_clients: int = 10,
    seed: int = 42,
) -> List[pd.DataFrame]:
    """Randomly shuffle data and split evenly among clients (IID)."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(df))
    client_size = len(df) // num_clients
    partitions = []
    for cid in range(num_clients):
        start = cid * client_size
        end = start + client_size if cid < num_clients - 1 else len(df)
        partitions.append(df.iloc[indices[start:end]].copy())
    return partitions


def create_dirichlet_partitions(
    df: pd.DataFrame,
    label_cols: List[str],
    num_clients: int = 10,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[pd.DataFrame]:
    """
    Create non-IID partitions using Dirichlet label distribution.

    Each sample's label vector is converted to a distribution over labels.
    Dirichlet(alpha) assigns each sample to a client, with alpha controlling
    how concentrated each client's labels are.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset with label columns.
    label_cols : list of str
        Label column names (e.g., ["toxic", "obscene", ...]).
    num_clients : int
        Number of clients.
    alpha : float
        Dirichlet concentration parameter.
        Smaller = more non-IID. Typical values: 0.1 (extreme), 0.5 (moderate), 1.0 (mild).
    seed : int
        Random seed.

    Returns
    -------
    list of pd.DataFrame
        One DataFrame per client.
    """
    rng = np.random.RandomState(seed)
    num_samples = len(df)
    label_matrix = df[label_cols].values  # (N, 6)

    # For non-toxic samples (all labels 0), give a small uniform distribution
    label_sum = label_matrix.sum(axis=1)
    dist = np.where(
        label_sum[:, None] > 0,
        label_matrix / label_sum[:, None],
        np.ones(len(label_cols))[None, :] / len(label_cols),
    )

    # Dirichlet: for each sample, sample client assignment probabilities
    client_probs = rng.dirichlet([alpha] * num_clients, size=num_samples)
    # Weight by label distribution
    assignment_weights = client_probs * dist[:, None, :].sum(axis=-1)
    client_assignments = assignment_weights.argmax(axis=1)

    partitions = [df.iloc[np.where(client_assignments == cid)[0]].copy()
                  for cid in range(num_clients)]

    _print_partition_stats(partitions, label_cols)
    return partitions


def create_topic_partitions(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    num_clients: int = 10,
    seed: int = 42,
    max_features: int = 5000,
) -> List[pd.DataFrame]:
    """
    Create non-IID partitions by topic clustering (TF-IDF + K-means).

    This simulates realistic user groups who share vocabulary domains
    (e.g., gaming slang, political arguments, general chat).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with cleaned text.
    text_col : str
        Column name for text.
    num_clients : int
        Number of topic clusters.
    seed : int
        Random seed.
    max_features : int
        TF-IDF vocabulary size.

    Returns
    -------
    list of pd.DataFrame
        One DataFrame per topic cluster.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("scikit-learn required for topic partitioning")

    print(f"  Vectorizing {len(df):,} texts (TF-IDF, {max_features} features)...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = vectorizer.fit_transform(df[text_col].fillna(""))

    print(f"  Clustering into {num_clients} topics (K-means)...")
    kmeans = KMeans(n_clusters=num_clients, random_state=seed, n_init=5)
    clusters = kmeans.fit_predict(X)

    # Show top words per cluster
    print(f"\n  Topic clusters:")
    feature_names = vectorizer.get_feature_names_out()
    for cid in range(num_clients):
        centroid = kmeans.cluster_centers_[cid]
        top_idx = centroid.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_idx]
        cluster_size = (clusters == cid).sum()
        print(f"    Client {cid} ({cluster_size:,} samples): {', '.join(top_words)}")

    partitions = [df.iloc[np.where(clusters == cid)[0]].copy()
                  for cid in range(num_clients)]
    return partitions


def _print_partition_stats(partitions: List[pd.DataFrame], label_cols: List[str]) -> None:
    """Print label distribution summary across partitions."""
    print(f"\n  Partition label distributions:")
    print(f"  {'Client':<8} {'Size':<8}", end="")
    for col in label_cols:
        print(f"{col:>14}", end="")
    print()

    for cid, part in enumerate(partitions):
        print(f"  {cid:<8} {len(part):<8}", end="")
        for col in label_cols:
            pct = part[col].mean() * 100
            print(f"{pct:>13.2f}%", end="")
        print()


def train_val_split_partitions(
    partitions: List[pd.DataFrame],
    val_frac: float = 0.1,
    label_cols: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Split each partition into train/val sets (stratified)."""
    from sklearn.model_selection import train_test_split

    train_parts = []
    val_parts = []
    for part in partitions:
        if label_cols and len(part) > 1:
            stratify = (part[label_cols].sum(axis=1) > 0).astype(int)
            # Handle edge case: only one class
            if stratify.nunique() == 1:
                stratify = None
            train, val = train_test_split(
                part, test_size=val_frac, random_state=seed, stratify=stratify,
            )
        else:
            train, val = train_test_split(
                part, test_size=val_frac, random_state=seed,
            )
        train_parts.append(train)
        val_parts.append(val)
    return train_parts, val_parts


def save_partitions(
    partitions: List[pd.DataFrame],
    output_dir: str,
    prefix: str = "",
) -> None:
    """Save partitions to CSV files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for cid, part in enumerate(partitions):
        path = out / f"{prefix}client_{cid}.csv"
        part.to_csv(path, index=False)
    print(f"  Saved {len(partitions)} partitions to {out}")


def load_partitions(
    input_dir: str,
    num_clients: int = 10,
    prefix: str = "",
) -> List[pd.DataFrame]:
    """Load partitions from CSV files."""
    inp = Path(input_dir)
    partitions = []
    for cid in range(num_clients):
        path = inp / f"{prefix}client_{cid}.csv"
        partitions.append(pd.read_csv(path))
    return partitions
