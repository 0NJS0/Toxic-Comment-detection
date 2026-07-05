"""
DistilBERT Model for Multi-Label Toxicity Classification
=========================================================

This module defines the model architecture used for toxicity detection.

Architecture:
    Input (batch, 128) → DistilBERT → [CLS] (batch, 768) → Dropout → Linear(768→6) → Logits

Key Design Decisions:
    - We use DistilBERT's [CLS] token as the sentence representation
    - Final layer outputs 6 RAW LOGITS (no sigmoid)
    - BCEWithLogitsLoss applies sigmoid internally
    - Dropout prevents overfitting on this relatively small dataset

Why 6 outputs?
    Jigsaw has 6 toxicity labels: toxic, severe_toxic, obscene, threat, insult, identity_hate
    Each is an independent binary classification.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional


class DistilBERTForMultiLabelClassification(nn.Module):
    """
    DistilBERT with a multi-label classification head.

    This wraps HuggingFace's DistilBERT model and adds a custom
    classification head for our 6 toxicity labels.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 6,
        dropout: float = 0.1,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the model.

        Parameters
        ----------
        model_name : str
            HuggingFace model identifier (e.g., "distilbert-base-uncased",
            "huawei-noah/TinyBERT_General_4L_312D")
        num_labels : int
            Number of output labels (6 for Jigsaw toxicity)
        dropout : float
            Dropout probability for the classification head
        cache_dir : str, optional
            Directory to cache the pretrained model
        """
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels

        # Load pretrained DistilBERT
        # We use AutoModel to get the base model without any classification head
        self.distilbert = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

        # Get the hidden size from the model config
        # DistilBERT-base: 768, TinyBERT: 312, MobileBERT: 512
        hidden_size = self.distilbert.config.hidden_size

        # Classification head: dropout + linear layer
        # We apply dropout to the [CLS] token representation before the final layer
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize the classifier weights
        # (DistilBERT weights are already pretrained)
        self._init_classifier()

    def _init_classifier(self):
        """Initialize the classification head weights."""
        # Standard initialization for linear layers
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs, shape (batch_size, seq_len)
        attention_mask : torch.Tensor
            1 for real tokens, 0 for padding, shape (batch_size, seq_len)

        Returns
        -------
        torch.Tensor
            Raw logits, shape (batch_size, num_labels)
            NO sigmoid applied — BCEWithLogitsLoss handles that
        """
        # Pass through DistilBERT
        # outputs.last_hidden_state: (batch_size, seq_len, hidden_size)
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Extract [CLS] token representation (first token)
        # In DistilBERT, [CLS] is at position 0
        cls_output = outputs.last_hidden_state[:, 0]  # (batch_size, hidden_size)

        # Apply dropout and classification head
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # (batch_size, num_labels)

        return logits

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_pretrained(self, save_path: str):
        """Save model and config to disk."""
        self.distilbert.save_pretrained(save_path)
        # Save classifier separately
        torch.save(
            self.classifier.state_dict(),
            f"{save_path}/classifier.pt"
        )
        # Save dropout config
        torch.save(
            {"dropout": self.dropout.p, "num_labels": self.num_labels},
            f"{save_path}/head_config.pt"
        )

    @classmethod
    def from_pretrained(cls, load_path: str, model_name: str = None):
        """Load model from disk."""
        # Load head config
        head_config = torch.load(f"{load_path}/head_config.pt", map_location="cpu")
        dropout = head_config["dropout"]
        num_labels = head_config["num_labels"]

        # Create model with matching architecture
        model = cls(
            model_name=model_name or "distilbert-base-uncased",
            num_labels=num_labels,
            dropout=dropout,
        )

        # Load base model weights
        model.distilbert = model.distilbert.from_pretrained(load_path)

        # Load classifier weights
        model.classifier.load_state_dict(
            torch.load(f"{load_path}/classifier.pt", map_location="cpu")
        )

        return model


def create_model(config: dict) -> DistilBERTForMultiLabelClassification:
    """
    Factory function to create model from config.

    Parameters
    ----------
    config : dict
        Project configuration dictionary

    Returns
    -------
    DistilBERTForMultiLabelClassification
        Initialized model
    """
    model_config = config["model"]
    return DistilBERTForMultiLabelClassification(
        model_name=model_config["name"],
        num_labels=model_config["num_labels"],
        dropout=model_config["dropout"],
        cache_dir=model_config.get("cache_dir"),
    )


if __name__ == "__main__":
    # Quick test
    model = DistilBERTForMultiLabelClassification()
    print(f"Model: {model.model_name}")
    print(f"Parameters: {model.get_num_parameters():,}")
    print(f"Output labels: {model.num_labels}")

    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    logits = model(input_ids, attention_mask)
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")  # Should be (2, 6)