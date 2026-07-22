"""
FedPref Federated Learning Orchestrator
=========================================

Full pipeline:
  1. Load cleaned dataset
  2. Partition data (IID / Dirichlet / Topic)
  3. Split each partition into train/val
  4. Tokenize each partition
  5. Create Flower-compatible clients
  6. Run federated learning (manual FedAvg loop)
  7. Evaluate global model + per-client personalization
  8. Save all results
"""

import json
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Optional

from datasets import Dataset as HFDataset

from utils.config import load_config
from utils.helpers import set_random_seed, ensure_dir

from federated.partition import (
    create_iid_partitions,
    create_dirichlet_partitions,
    create_topic_partitions,
    train_val_split_partitions,
)
from federated.server import run_federated
from evaluation.metrics import compute_metrics


def tokenize_partition(
    texts: List[str],
    labels: np.ndarray,
    ids: List[str],
    tokenizer,
    max_length: int,
) -> HFDataset:
    """Tokenize a single client's partition into HF Dataset."""
    encoded = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    data_dict = {
        "input_ids": encoded["input_ids"].tolist(),
        "attention_mask": encoded["attention_mask"].tolist(),
        "labels": labels.astype(np.float32).tolist(),
        "id": ids.tolist(),
    }
    return HFDataset.from_dict(data_dict)


def prepare_partitions(
    config: dict,
    tokenizer,
) -> tuple:
    """
    Load and partition the dataset for federated learning.

    Returns
    -------
    tuple: (train_datasets, val_datasets, global_val_dataset)
    """
    data_cfg = config["data"]
    label_cols = data_cfg["labels"]
    max_length = data_cfg["max_seq_length"]
    fed_cfg = config["federated"]
    num_clients = fed_cfg["num_clients"]

    data_path = Path(data_cfg["processed"]["path"]) / data_cfg["processed"]["train_file"]
    print(f"\nLoading cleaned data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Total samples: {len(df):,}")

    from sklearn.model_selection import train_test_split
    stratify = (df[label_cols].sum(axis=1) > 0).astype(int)
    df_train, df_global_val = train_test_split(
        df, test_size=0.1, random_state=config["project"]["seed"],
        stratify=stratify,
    )
    print(f"Global split: {len(df_train):,} train, {len(df_global_val):,} val")

    partition_type = fed_cfg.get("partition_type", "iid")
    print(f"\nCreating {partition_type} partitions ({num_clients} clients)...")

    if partition_type == "iid" or fed_cfg.get("iid", True):
        partitions = create_iid_partitions(
            df_train, num_clients=num_clients, seed=config["project"]["seed"],
        )
    elif partition_type == "dirichlet":
        partitions = create_dirichlet_partitions(
            df_train, label_cols, num_clients=num_clients,
            alpha=fed_cfg.get("alpha", 0.5), seed=config["project"]["seed"],
        )
    elif partition_type == "topic":
        partitions = create_topic_partitions(
            df_train, text_col="clean_text", num_clients=num_clients,
            seed=config["project"]["seed"],
        )
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")

    train_dfs, val_dfs = train_val_split_partitions(
        partitions, val_frac=0.1, label_cols=label_cols,
        seed=config["project"]["seed"],
    )

    print(f"\nTokenizing partitions...")
    train_datasets = []
    val_datasets = []

    for cid in range(num_clients):
        train_ds = tokenize_partition(
            texts=train_dfs[cid]["clean_text"].tolist(),
            labels=train_dfs[cid][label_cols].values,
            ids=train_dfs[cid]["id"].values,
            tokenizer=tokenizer, max_length=max_length,
        )
        val_ds = tokenize_partition(
            texts=val_dfs[cid]["clean_text"].tolist(),
            labels=val_dfs[cid][label_cols].values,
            ids=val_dfs[cid]["id"].values,
            tokenizer=tokenizer, max_length=max_length,
        )
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        print(f"  Client {cid}: {len(train_ds)} train, {len(val_ds)} val")

    global_val = tokenize_partition(
        texts=df_global_val["clean_text"].tolist(),
        labels=df_global_val[label_cols].values,
        ids=df_global_val["id"].values,
        tokenizer=tokenizer, max_length=max_length,
    )
    print(f"  Global val: {len(global_val)} samples")

    return train_datasets, val_datasets, global_val


def run_experiment(config: dict) -> Dict:
    """Run a single federated learning experiment."""
    from transformers import AutoTokenizer
    from training.train import MultiLabelDataCollator
    from federated.client import FedPrefClient

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    tokenizer_path = Path(config["data"]["processed"]["path"]) / "tokenized" / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    train_datasets, val_datasets, global_val_dataset = prepare_partitions(
        config, tokenizer,
    )

    # Create clients
    data_collator = MultiLabelDataCollator(tokenizer)
    clients = []
    for cid in range(len(train_datasets)):
        client = FedPrefClient(
            cid=cid,
            train_dataset=train_datasets[cid],
            val_dataset=val_datasets[cid],
            config=config,
            data_collator=data_collator,
            device=device,
        )
        clients.append(client)

    # Run FL
    results = run_federated(
        config=config,
        clients=clients,
        global_val_dataset=global_val_dataset,
        tokenizer=tokenizer,
        device=device,
        output_dir="results/federated",
    )

    return results


def run_all_experiments(config: dict) -> Dict:
    """
    Run all FL partition experiments.
    """
    all_results = {}

    experiments = [
        ("iid", None, True),
        ("dirichlet", 0.5, False),
        ("dirichlet", 0.1, False),
        ("topic", None, False),
    ]

    for partition_type, alpha, is_iid in experiments:
        print(f"\n\n{'#'*60}")
        print(f"# Experiment: {partition_type.upper()} (alpha={alpha})")
        print(f"{'#'*60}")

        config["federated"]["iid"] = is_iid
        config["federated"]["partition_type"] = partition_type
        if alpha is not None:
            config["federated"]["alpha"] = alpha

        exp_name = f"{partition_type}{'_'+str(alpha) if alpha else ''}"
        results = run_experiment(config)
        all_results[exp_name] = results

        exp_path = Path("results/federated") / f"experiment_{exp_name}.json"
        with open(exp_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    summary_path = Path("results/federated") / "all_experiments_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"All experiments complete!")
    print(f"Summary saved: {summary_path}")

    return all_results
