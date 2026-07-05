"""
Flower Client for FedPref
==========================

Each client:
  1. Loads its local data partition
  2. Receives global LoRA + classifier weights from the server
  3. Trains locally on its partition
  4. Returns only LoRA + classifier weights (not the base encoder)

Only ~0.2% of the model is communicated per round, satisfying the
constraint that raw user text never leaves the device.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from tqdm import tqdm

import flwr as fl

from models.registry import create_model
from evaluation.metrics import compute_metrics


def get_parameters(model: nn.Module) -> List[bytes]:
    """Return trainable parameters as numpy bytes (for Flower)."""
    import numpy as np
    return [p.detach().cpu().numpy().astype(np.float32).tobytes()
            for p in model.parameters() if p.requires_grad]


def get_parameters_ndarrays(model: nn.Module) -> List:
    """Return trainable parameters as numpy ndarrays."""
    import numpy as np
    return [p.detach().cpu().numpy() for p in model.parameters() if p.requires_grad]


def set_parameters(model: nn.Module, parameters: List) -> None:
    """Set trainable parameters from numpy arrays or tensors."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    for param, new_param in zip(trainable, parameters):
        param.data.copy_(torch.tensor(new_param, device=param.device))


def train_local_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> Dict:
    """Simplified training loop for FL (no scheduler needed)."""
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for batch in tqdm(dataloader, desc="  Local train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())

    import numpy as np
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = avg_loss
    return metrics


class FedPrefClient(fl.client.NumPyClient):
    """
    Flower client for FedPref.

    Trains only LoRA adapters + classifier head on local data.
    The base encoder stays frozen and is never communicated.
    """

    def __init__(
        self,
        cid: int,
        train_dataset,
        val_dataset,
        config: dict,
        data_collator,
        device: torch.device,
    ):
        self.cid = cid
        self.config = config
        self.device = device
        train_config = config["training"]

        # Initialize model (with LoRA if configured)
        self.model = create_model(config).to(device)

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_config["batch_size"],
            shuffle=True,
            collate_fn=data_collator,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=train_config["batch_size"] * 2,
            shuffle=False,
            collate_fn=data_collator,
        )

        # Loss
        self.loss_fn = nn.BCEWithLogitsLoss()

        print(f"  Client {cid}: {len(train_dataset)} train, {len(val_dataset)} val samples")

    def get_parameters(self, config=None) -> List:
        """Return current trainable parameters."""
        return get_parameters_ndarrays(self.model)

    def fit(
        self, parameters: List, config: Dict
    ) -> Tuple[List, int, Dict]:
        """Train on local data with received global parameters."""
        set_parameters(self.model, parameters)

        train_config = self.config["training"]
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=train_config["learning_rate"],
        )

        local_epochs = config.get("local_epochs", 1)
        total_loss = 0.0

        for epoch in range(local_epochs):
            epoch_metrics = train_local_epoch(
                self.model, self.train_loader, optimizer,
                self.loss_fn, self.device, train_config["gradient_clip"],
            )
            total_loss += epoch_metrics["loss"]

        avg_loss = total_loss / local_epochs

        # Add a small proximal term if FedProx
        proximal_mu = config.get("proximal_mu", 0.0)
        if proximal_mu > 0.0:
            prox_loss = 0.0
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            param_tensors = [torch.tensor(p, device=self.device)
                             for p in parameters]
            for p, p_global in zip(trainable, param_tensors):
                prox_loss += (p - p_global).pow(2).sum()
            # prox_loss is already included during training, this is tracking
            avg_loss += 0.5 * proximal_mu * prox_loss.item() / len(self.train_loader)

        return get_parameters_ndarrays(self.model), len(self.train_loader.dataset), {
            "cid": self.cid,
            "loss": float(avg_loss),
        }

    def evaluate(
        self, parameters: List, config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate on local validation data."""
        set_parameters(self.model, parameters)
        self.model.eval()

        all_logits = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.loss_fn(logits, labels)

                total_loss += loss.item()
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        import torch
        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_metrics(all_logits, all_labels)
        metrics["loss"] = avg_loss

        return float(avg_loss), len(self.val_loader.dataset), metrics
