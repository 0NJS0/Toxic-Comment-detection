"""
Training Module for Centralized DistilBERT Training
=====================================================

What this module does:
    1. Loads tokenized dataset from Module 2 (train/validation/test)
    2. Creates PyTorch DataLoaders
    3. Sets up optimizer (AdamW), scheduler, and loss (BCEWithLogitsLoss)
    4. Trains the model for N epochs
    5. Evaluates on validation set after each epoch
    6. Evaluates on held-out test set after training
    7. Saves checkpoints, logs, and plots

Training loop details:
    - Each epoch: iterate over all batches, compute loss, backprop, step optimizer
    - After each epoch: run validation, compute metrics, save checkpoint
    - After training: final evaluation on unseen test set
    - Metrics tracked: loss, accuracy, precision, recall, F1 (macro & micro), ROC-AUC
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

from datasets import load_from_disk
from transformers import get_linear_schedule_with_warmup

from models.distilbert import create_model, DistilBERTForMultiLabelClassification
from evaluation.metrics import compute_metrics, generate_confusion_matrices


# ---------------------------------------------------------------------------
# Class Weighting
# ---------------------------------------------------------------------------

def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss to handle class imbalance.

    pos_weight[i] = (num_negatives[i] / num_positives[i])
    Gives higher weight to rare positive labels so the model
    doesn't just predict all-negative.

    Parameters
    ----------
    labels : np.ndarray
        Training labels, shape (N, num_labels)

    Returns
    -------
    torch.Tensor
        pos_weight tensor of shape (num_labels,)
    """
    num_pos = labels.sum(axis=0)
    num_neg = labels.shape[0] - num_pos
    pos_weight = num_neg / np.maximum(num_pos, 1)
    return torch.tensor(pos_weight, dtype=torch.float32)


def compute_truncation_stats(dataset) -> None:
    """
    Compute and print how many sequences were truncated.
    """
    import numpy as np
    input_ids = np.array(dataset["train"]["input_ids"])
    pad_token_id = 0
    truncated = (input_ids[:, -1] != pad_token_id).sum()
    total = len(dataset["train"])
    if truncated > 0:
        print(f"  Truncation warning: {truncated:,} ({100*truncated/total:.1f}%) training sequences "
              f"exceeded the max length ({input_ids.shape[1]} tokens)")
    else:
        print(f"  No sequences truncated at {input_ids.shape[1]} tokens")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_tokenized_dataset(config: dict):
    """
    Load the tokenized dataset saved by Module 2.

    Returns a HuggingFace DatasetDict with 'train', 'validation', and 'test' splits.
    """
    data_path = Path(config["data"]["processed"]["path"]) / "tokenized"
    print(f"\nLoading tokenized dataset from: {data_path}")
    dataset = load_from_disk(str(data_path))
    print(f"  Train: {len(dataset['train']):,} samples")
    print(f"  Val:   {len(dataset['validation']):,} samples")
    print(f"  Test:  {len(dataset['test']):,} samples")
    return dataset


class MultiLabelDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        import torch
        input_ids = torch.tensor([s["input_ids"] for s in batch], dtype=torch.long)
        attention_mask = torch.tensor([s["attention_mask"] for s in batch], dtype=torch.long)
        labels = torch.tensor([s["labels"] for s in batch], dtype=torch.float32)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.

    HuggingFace Datasets store data as lists. This function
    converts batches of lists into PyTorch tensors.

    Parameters
    ----------
    batch : list
        List of samples, each containing input_ids, attention_mask, labels

    Returns
    -------
    Dict[str, torch.Tensor]
        Batched tensors ready for model input
    """
    input_ids = torch.tensor([s["input_ids"] for s in batch], dtype=torch.long)
    attention_mask = torch.tensor([s["attention_mask"] for s in batch], dtype=torch.long)
    labels = torch.tensor([s["labels"] for s in batch], dtype=torch.float32)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: nn.Module,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The DistilBERT model
    dataloader : DataLoader
        Training data loader
    optimizer : torch.optim.Optimizer
        AdamW optimizer
    scheduler : _LRScheduler
        Learning rate scheduler
    loss_fn : nn.Module
        BCEWithLogitsLoss
    device : torch.device
        Device to train on (cuda/cpu)
    gradient_clip : float
        Max gradient norm for clipping

    Returns
    -------
    Dict[str, float]
        Average loss and metrics for this epoch
    """
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass: raw logits (no sigmoid)
        logits = model(input_ids, attention_mask)

        # Compute loss (BCEWithLogitsLoss applies sigmoid internally)
        loss = loss_fn(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()
        scheduler.step()

        # Track
        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())

        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Compute epoch metrics
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Average loss
    avg_loss = total_loss / len(dataloader)

    # Compute metrics (accuracy, F1, ROC-AUC, etc.)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = avg_loss

    return metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    return_logits: bool = False,
) -> Dict[str, float]:
    """
    Evaluate the model on validation set.

    No gradient computation — uses torch.no_grad() for speed and memory.

    Parameters
    ----------
    model : nn.Module
        The DistilBERT model
    dataloader : DataLoader
        Validation data loader
    loss_fn : nn.Module
        BCEWithLogitsLoss
    device : torch.device
        Device to evaluate on
    return_logits : bool
        If True, also returns all_logits and all_labels

    Returns
    -------
    Dict[str, float] or tuple
        Validation metrics, or (metrics, all_logits, all_labels) if return_logits=True
    """
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    avg_loss = total_loss / len(dataloader)

    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = avg_loss

    if return_logits:
        return metrics, all_logits, all_labels
    return metrics


# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------

def run_training(config: dict) -> None:
    """
    Execute the full centralized training pipeline.

    Steps:
     1. Setup device, seed, and output directories
     2. Load tokenized dataset
     3. Create DataLoaders
     4. Initialize model, optimizer, scheduler, loss
     5. Train for N epochs with validation after each
     6. Evaluate on held-out test set
     7. Save final model, checkpoints, logs, visualizations
    """
    print("=" * 60)
    print("MODULE 3: CENTRALIZED TRAINING (DistilBERT)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    train_config = config["training"]
    model_config = config["model"]

    # Setup output directories
    checkpoint_dir = Path(config["results"]["checkpoints"])
    log_dir = Path(config["results"]["logs"])
    plot_dir = Path(config["results"]["plots"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading tokenized dataset...")
    dataset = load_tokenized_dataset(config)

    # Create DataLoaders
    # num_workers=0 for CPU, >0 for GPU (but can cause issues on some systems)
    num_workers = 2 if device.type == "cuda" else 0
    train_loader = DataLoader(
        dataset["train"],
        batch_size=train_config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        dataset["validation"],
        batch_size=train_config["batch_size"] * 2,  # Larger batches for eval
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # Show truncation stats
    compute_truncation_stats(dataset)

    # ------------------------------------------------------------------
    # Initialize model
    # ------------------------------------------------------------------
    print("\n[2/6] Initializing model...")
    model = create_model(config)
    total_params = model.get_num_parameters()
    print(f"  Model: {model_config['name']}")
    print(f"  Parameters: {total_params:,}")

    # Count trainable vs frozen params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,}")

    model = model.to(device)

    # ------------------------------------------------------------------
    # Setup optimizer, scheduler, loss
    # ------------------------------------------------------------------
    print("\n[3/6] Setting up optimizer and scheduler...")

    # AdamW with weight decay (only applied to non-bias, non-LayerNorm params)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": train_config["weight_decay"],
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=train_config["learning_rate"],
    )

    # Linear schedule with warmup
    # Warmup: gradually increase LR from 0 → target LR
    # Decay: gradually decrease LR from target → 0
    total_steps = len(train_loader) * train_config["epochs"]
    warmup_steps = min(train_config["warmup_steps"], total_steps // 10)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Compute class weights for imbalanced labels
    all_labels = np.array(dataset["train"]["labels"])
    pos_weight = compute_pos_weight(all_labels)
    label_names = config["data"]["labels"]
    print(f"\n  Class weights (neg/pos ratio):")
    for name, w in zip(label_names, pos_weight.tolist()):
        print(f"    {name:<16s} {w:.2f}")

    # BCEWithLogitsLoss with per-label pos_weight for rare classes
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    print(f"  Optimizer: AdamW (lr={train_config['learning_rate']})")
    print(f"  Scheduler: Linear warmup ({warmup_steps} steps) + linear decay")
    print(f"  Loss:      BCEWithLogitsLoss (weighted)")
    print(f"  Total steps: {total_steps}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("\n[4/6] Starting training...")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1_macro": [],
        "val_f1_macro": [],
        "learning_rates": [],
    }

    best_val_f1 = 0.0
    epochs_no_improve = 0
    early_stop_patience = train_config.get("early_stopping_patience", 0)
    start_time = time.time()

    for epoch in range(train_config["epochs"]):
        epoch_start = time.time()
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{train_config['epochs']}")
        print(f"{'='*50}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn,
            device, train_config["gradient_clip"],
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        epoch_time = time.time() - epoch_start

        # Log
        print(f"\n  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1-macro: {train_metrics['f1_macro']:.4f}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1-macro: {val_metrics['f1_macro']:.4f}")
        print(f"  Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Store history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["train_f1_macro"].append(train_metrics["f1_macro"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["learning_rates"].append(scheduler.get_last_lr()[0])

        # Save best model (by validation F1-macro)
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            epochs_no_improve = 0
            best_path = checkpoint_dir / "best_model"
            best_path.mkdir(exist_ok=True)
            model.save_pretrained(str(best_path))
            print(f"  [OK] New best model saved! (F1-macro: {best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")

        # Early stopping check
        if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
            print(f"\n  Early stopping triggered after {epoch + 1} epochs "
                  f"(no val F1 improvement for {early_stop_patience} epochs)")
            break

        # Save checkpoint (every epoch)
        ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }, str(ckpt_path))
        print(f"  Checkpoint saved: {ckpt_path}")

    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Best val F1-macro: {best_val_f1:.4f}")

    # ------------------------------------------------------------------
    # Final evaluation on validation set
    # ------------------------------------------------------------------
    print("\n[5/7] Running final evaluation on validation set...")

    final_metrics = evaluate(model, val_loader, loss_fn, device)
    _print_metrics("Final validation results:", final_metrics)

    # ------------------------------------------------------------------
    # Test set evaluation (using best model from training)
    # ------------------------------------------------------------------
    print("\n[6/7] Evaluating on held-out test set...")

    best_model_path = checkpoint_dir / "best_model"
    if best_model_path.exists():
        print(f"  Loading best model from {best_model_path}...")
        test_model = DistilBERTForMultiLabelClassification.from_pretrained(
            str(best_model_path),
            model_name=config["model"]["name"],
        )
        test_model = test_model.to(device)
    else:
        print("  No best model found, using last-epoch model...")
        test_model = model

    test_loader = DataLoader(
        dataset["test"],
        batch_size=train_config["batch_size"] * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    test_metrics, test_logits, test_labels = evaluate(test_model, test_loader, loss_fn, device, return_logits=True)
    _print_metrics("Test set results:", test_metrics)

    # Generate and display confusion matrices for test set
    label_names = config["data"]["labels"]
    cm_save_path = plot_dir / "confusion_matrices.png"
    generate_confusion_matrices(test_logits, test_labels, label_names, save_path=cm_save_path)

    # ------------------------------------------------------------------
    # Save results and plots
    # ------------------------------------------------------------------
    print("\n[7/7] Saving results, logs, and plots...")

    # Save training history as JSON
    history_path = log_dir / "training_history.json"
    with open(history_path, "w") as f:
        # Convert numpy values to Python floats
        clean_history = {}
        for k, v in history.items():
            clean_history[k] = [float(x) if hasattr(x, 'item') else x for x in v]
        json.dump(clean_history, f, indent=2)
    print(f"  History saved: {history_path}")

    # Save validation metrics
    val_path = log_dir / "final_metrics.json"
    with open(val_path, "w") as f:
        clean_metrics = {
            k: float(v) if hasattr(v, 'item') else v
            for k, v in final_metrics.items()
        }
        json.dump(clean_metrics, f, indent=2)
    print(f"  Validation metrics saved: {val_path}")

    # Save test metrics
    test_path = log_dir / "test_metrics.json"
    with open(test_path, "w") as f:
        clean_metrics = {
            k: float(v) if hasattr(v, 'item') else v
            for k, v in test_metrics.items()
        }
        json.dump(clean_metrics, f, indent=2)
    print(f"  Test metrics saved: {test_path}")

    # Plot training curves
    _plot_training_curves(history, plot_dir)

    print(f"\n{'='*50}")
    print("MODULE 3 COMPLETE")
    print(f"{'='*50}")
    print(f"\nTrained model saved to: {checkpoint_dir}")
    print(f"Training logs saved to: {log_dir}")
    print(f"Test metrics saved to:  {test_path}")
    print(f"Plots saved to:         {plot_dir}")
    print(f"\nModel is ready for evaluation / federated learning.")


def _print_metrics(heading: str, metrics: dict) -> None:
    print(f"\n  {heading}")
    print(f"  Loss:        {metrics['loss']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1-macro:    {metrics['f1_macro']:.4f}")
    print(f"  F1-micro:    {metrics['f1_micro']:.4f}")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")


def _plot_training_curves(history: dict, plot_dir: Path) -> None:
    """
    Generate and save training/validation curves.

    This creates loss and F1 curves to visualize training progress.
    """
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1-macro curve
    axes[1].plot(epochs, history["train_f1_macro"], "b-o", label="Train F1-macro")
    axes[1].plot(epochs, history["val_f1_macro"], "r-o", label="Val F1-macro")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1-macro")
    axes[1].set_title("F1-macro Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = plot_dir / "training_curves.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved: {plot_path}")
