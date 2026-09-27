import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_from_disk
from utils.config import load_config
from utils.helpers import set_random_seed, ensure_dir
from models.atam import DistilBERTWithATAM, create_atam_model
from evaluation.metrics import compute_metrics


def load_tokenized_dataset(config: dict):
    data_path = Path(config["data"]["processed"]["path"]) / "tokenized"
    print(f"Loading tokenized dataset from: {data_path}")
    dataset = load_from_disk(str(data_path))
    print(f"  Train: {len(dataset['train']):,} samples")
    print(f"  Val:   {len(dataset['validation']):,} samples")
    print(f"  Test:  {len(dataset['test']):,} samples")
    return dataset


def collate_fn(batch):
    import torch
    input_ids = torch.tensor([s["input_ids"] for s in batch], dtype=torch.long)
    attention_mask = torch.tensor([s["attention_mask"] for s in batch], dtype=torch.long)
    labels = torch.tensor([s["labels"] for s in batch], dtype=torch.float32)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics, all_logits, all_labels


def train_atam(
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str = "results/atam",
):
    train_cfg = config["training"]
    config["model"]["use_atam"] = True
    model = create_atam_model(config)
    total_params = model.get_num_parameters()
    print(f"\nModel parameters: {total_params:,}")
    model = model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": train_cfg["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=train_cfg["learning_rate"],
    )

    from transformers import get_linear_schedule_with_warmup
    total_steps = len(train_loader) * train_cfg["epochs"]
    warmup_steps = min(train_cfg["warmup_steps"], total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_f1_macro": [], "val_loss": []}
    best_f1 = 0.0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm
    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}")

        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                val_loss += loss.item()
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        val_metrics = compute_metrics(all_logits, all_labels)
        val_loss_avg = val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}: Train Loss={avg_loss:.4f} | "
              f"Val Loss={val_loss_avg:.4f} | Val F1={val_metrics['f1_macro']:.4f}")

        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss_avg)
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            model.save_pretrained(str(output_path / "best_model"))
            print(f"  New best model saved (F1={best_f1:.4f})")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, str(output_path / f"checkpoint_epoch_{epoch+1}.pt"))
        with open(output_path / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val F1-macro: {best_f1:.4f}")
    return history


def main():
    print("=" * 60)
    print("MODULE 6: ATAM Training (Adaptive Toxic Attention Module)")
    print("=" * 60)

    config = load_config("configs/config.yaml")
    set_random_seed(config["project"]["seed"])
    ensure_dir("results/atam")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = load_tokenized_dataset(config)

    train_loader = DataLoader(
        dataset["train"],
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        dataset["validation"],
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        dataset["test"],
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False,
        collate_fn=collate_fn,
    )

    history = train_atam(config, train_loader, val_loader, device)

    print(f"\n{'='*60}")
    print("TEST SET EVALUATION")
    print("=" * 60)
    best_path = Path("results/atam/best_model")
    if best_path.exists():
        print(f"Loading best model from {best_path}...")
        from models.atam import DistilBERTWithATAM
        test_model = DistilBERTWithATAM.from_pretrained(
            str(best_path),
            model_name=config["model"]["name"],
        )
        test_model = test_model.to(device)
    else:
        print("No best model found, re-creating model...")
        config["model"]["use_atam"] = True
        from models.atam import create_atam_model
        test_model = create_atam_model(config)
        test_model.load_state_dict(torch.load(
            "results/atam/checkpoint_epoch_6.pt", map_location=device, weights_only=True
        )["model_state_dict"])
        test_model.to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    test_metrics, test_logits, test_labels = evaluate(test_model, test_loader, loss_fn, device)

    print(f"  Loss:        {test_metrics['loss']:.4f}")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  Precision:   {test_metrics['precision']:.4f}")
    print(f"  Recall:      {test_metrics['recall']:.4f}")
    print(f"  F1-macro:    {test_metrics['f1_macro']:.4f}")
    print(f"  F1-micro:    {test_metrics['f1_micro']:.4f}")
    print(f"  ROC-AUC:     {test_metrics['roc_auc']:.4f}")
    print(f"  Hamming Loss: {test_metrics['hamming_loss']:.4f}")

    from evaluation.metrics import generate_confusion_matrices
    label_names = config["data"]["labels"]
    generate_confusion_matrices(
        test_logits, test_labels, label_names,
        save_path=Path("results/atam/confusion_matrices.png"),
    )

    with open("results/atam/test_metrics.json", "w") as f:
        clean = {k: float(v) if hasattr(v, "item") else v for k, v in test_metrics.items()}
        json.dump(clean, f, indent=2)

    print(f"\n{'='*60}")
    print("MODULE 6 COMPLETE")
    print(f"Best val F1-macro: {max(history['val_f1_macro']):.4f}")
    print(f"Test F1-macro:     {test_metrics['f1_macro']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
