"""
Training Module for Centralized DistilBERT Training
=====================================================

What this module does:
    1. Loads tokenized dataset from Module 2
    2. Creates PyTorch DataLoaders
    3. Sets up optimizer (AdamW), scheduler, and loss (BCEWithLogitsLoss)
    4. Trains the model for N epochs
    5. Evaluates on validation set after each epoch
    6. Saves checkpoints, logs, and plots

Training loop details:
    - Each epoch: iterate over all batches, compute loss, backprop, step optimizer
    - After each epoch: run validation, compute metrics, save checkpoint
    - Metrics tracked: loss, accuracy, precision, recall, F1 (macro & micro), ROC-AUC
"""

import os
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
from transformers import (
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from models.distilbert import create_model
from evaluation.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_tokenized_dataset(config: dict):
    """
    Load the tokenized dataset saved by Module 2.

    Returns a HuggingFace DatasetDict with 'train' and 'validation' splits.
    """
    data_path = Path(config["data"]["processed"]["path"]) / "tokenized"
    print(f"\nLoading tokenized dataset from: {data_path}")
    dataset = load_from_disk(str(data_path))
    print(f"  Train: {len(dataset['train']):,} samples")
    print(f"  Val:   {len(dataset['validation']):,} samples")
    return dataset


def load_tokenizer(config: dict) -> AutoTokenizer:
    """Load the saved tokenizer from the tokenized dataset directory."""
    tokenizer_path = Path(config["data"]["processed"]["path"]) / "tokenized" / "tokenizer"
    print(f"  Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    return tokenizer


class MultiLabelDataCollator:
    """
    Collator that dynamically pads variable-length sequences per batch,
    then combines with multi-label labels.

    Uses HuggingFace's DataCollatorWithPadding for proper padding,
    avoiding wasted storage from pre-padded sequences.
    """

    def __init__(self, tokenizer: AutoTokenizer):
        self.pad_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest",
            return_tensors="pt",
        )

    def __call__(self, batch: list) -> Dict[str, torch.Tensor]:
        # Dynamically pad input_ids + attention_mask to longest in batch
        padded = self.pad_collator(
            [{"input_ids": s["input_ids"], "attention_mask": s["attention_mask"]} for s in batch]
        )
        # Labels are fixed-size (6 binary targets), no padding needed
        labels = torch.tensor([s["labels"] for s in batch], dtype=torch.float32)
        return {**padded, "labels": labels}


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

    Returns
    -------
    Dict[str, float]
        Validation metrics
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

    return metrics


# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(checkpoint_dir: Path) -> tuple:
    """Find the latest checkpoint epoch number and path."""
    if not checkpoint_dir.exists():
        return 0, None
    ckpt_files = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if not ckpt_files:
        return 0, None
    latest = ckpt_files[-1]
    epoch = int(latest.stem.split("_")[-1])
    return epoch, latest


def run_training(config: dict) -> None:
    """
    Execute the full centralized training pipeline.

    Steps:
    1. Setup device, seed, and output directories
    2. Load tokenized dataset
    3. Create DataLoaders
    4. Initialize model, optimizer, scheduler, loss
    5. Train for N epochs with validation after each
    6. Save final model, checkpoints, logs, visualizations
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
    # Check for existing checkpoints (resume support)
    # ------------------------------------------------------------------
    resume_epoch, resume_path = _find_latest_checkpoint(checkpoint_dir)
    if resume_epoch > 0:
        print(f"\nFound existing checkpoint: epoch {resume_epoch}")
        print(f"  Path: {resume_path}")
        print(f"  Will resume from epoch {resume_epoch + 1}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading tokenized dataset...")
    dataset = load_tokenized_dataset(config)
    tokenizer = load_tokenizer(config)
    data_collator = MultiLabelDataCollator(tokenizer)

    # Create DataLoaders
    num_workers = 2 if device.type == "cuda" else 0
    train_loader = DataLoader(
        dataset["train"],
        batch_size=train_config["batch_size"],
        shuffle=True,
        collate_fn=data_collator,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        dataset["validation"],
        batch_size=train_config["batch_size"] * 2,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=num_workers,
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # ------------------------------------------------------------------
    # Initialize model
    # ------------------------------------------------------------------
    print("\n[2/6] Initializing model...")
    model = create_model(config)
    total_params = model.get_num_parameters()
    print(f"  Model: {model_config['name']}")
    print(f"  Parameters: {total_params:,}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,}")
    model = model.to(device)

    # ------------------------------------------------------------------
    # Setup optimizer, scheduler, loss
    # ------------------------------------------------------------------
    print("\n[3/6] Setting up optimizer and scheduler...")

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

    total_steps = len(train_loader) * train_config["epochs"]
    warmup_steps = min(train_config["warmup_steps"], total_steps // 10)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_fn = nn.BCEWithLogitsLoss()

    print(f"  Optimizer: AdamW (lr={train_config['learning_rate']})")
    print(f"  Scheduler: Linear warmup ({warmup_steps} steps) + linear decay")
    print(f"  Loss:      BCEWithLogitsLoss")
    print(f"  Total steps: {total_steps}")

    # ------------------------------------------------------------------
    # Resume from checkpoint if available
    # ------------------------------------------------------------------
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1_macro": [],
        "val_f1_macro": [],
        "learning_rates": [],
    }
    start_epoch = 0
    best_val_f1 = 0.0

    if resume_epoch > 0 and resume_path is not None:
        print(f"\nResuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"]
        best_val_f1 = checkpoint["val_metrics"].get("f1_macro", 0.0)

        print(f"  Loaded model/optimizer/scheduler from epoch {start_epoch}")
        print(f"  Previous best F1-macro: {best_val_f1:.4f}")
        print(f"  Resuming from epoch {start_epoch + 1}/{train_config['epochs']}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("\n[4/6] Starting training...")

    start_time = time.time()

    for epoch in range(start_epoch, train_config["epochs"]):
        epoch_start = time.time()
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{train_config['epochs']}")
        print(f"{'='*50}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn,
            device, train_config["gradient_clip"],
        )

        val_metrics = evaluate(model, val_loader, loss_fn, device)

        epoch_time = time.time() - epoch_start

        print(f"\n  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1-macro: {train_metrics['f1_macro']:.4f}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1-macro: {val_metrics['f1_macro']:.4f}")
        print(f"  Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["train_f1_macro"].append(train_metrics["f1_macro"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["learning_rates"].append(scheduler.get_last_lr()[0])

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_path = checkpoint_dir / "best_model"
            best_path.mkdir(exist_ok=True)
            model.save_pretrained(str(best_path))
            print(f"  ✓ New best model saved! (F1-macro: {best_val_f1:.4f})")

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
    # Final evaluation
    # ------------------------------------------------------------------
    print("\n[5/6] Running final evaluation...")

    final_metrics = evaluate(model, val_loader, loss_fn, device)
    print(f"\n  Final validation results:")
    print(f"  Loss:        {final_metrics['loss']:.4f}")
    print(f"  Accuracy:    {final_metrics['accuracy']:.4f}")
    print(f"  Precision:   {final_metrics['precision']:.4f}")
    print(f"  Recall:      {final_metrics['recall']:.4f}")
    print(f"  F1-macro:    {final_metrics['f1_macro']:.4f}")
    print(f"  F1-micro:    {final_metrics['f1_micro']:.4f}")
    print(f"  ROC-AUC:     {final_metrics['roc_auc']:.4f}")
    print(f"  Hamming Loss: {final_metrics['hamming_loss']:.4f}")

    # ------------------------------------------------------------------
    # Save results and plots
    # ------------------------------------------------------------------
    print("\n[6/6] Saving results, logs, and plots...")

    # Save training history as JSON
    history_path = log_dir / "training_history.json"
    with open(history_path, "w") as f:
        # Convert numpy values to Python floats
        clean_history = {}
        for k, v in history.items():
            clean_history[k] = [float(x) if hasattr(x, 'item') else x for x in v]
        json.dump(clean_history, f, indent=2)
    print(f"  History saved: {history_path}")

    # Save final metrics
    final_path = log_dir / "final_metrics.json"
    with open(final_path, "w") as f:
        clean_metrics = {
            k: float(v) if hasattr(v, 'item') else v
            for k, v in final_metrics.items()
        }
        json.dump(clean_metrics, f, indent=2)
    print(f"  Final metrics saved: {final_path}")

    # Plot training curves
    _plot_training_curves(history, plot_dir)

    print(f"\n{'='*50}")
    print(f"MODULE 3 COMPLETE ✓")
    print(f"{'='*50}")
    print(f"\nTrained model saved to: {checkpoint_dir}")
    print(f"Training logs saved to: {log_dir}")
    print(f"Plots saved to:         {plot_dir}")
    print(f"\nModel is ready for Module 4/5 (Federated Learning / Evaluation).")


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
