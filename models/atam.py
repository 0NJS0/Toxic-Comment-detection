import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Optional, Tuple


class AdaptiveToxicAttentionModule(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int = 6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.W = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states projected through W, then tanh, then v to get energy scores.
        Softmax over sequence dimension gives attention weights.
        Weighted sum produces toxicity-aware representation.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size) from encoder
            attention_mask: (batch_size, seq_len) with 1 for real tokens, 0 for padding

        Returns:
            toxicity_aware_repr: (batch_size, hidden_size)
            attention_weights: (batch_size, seq_len) for visualization
        """
        e = self.v(torch.tanh(self.W(hidden_states)))
        e = e.squeeze(-1)

        attention_mask = attention_mask.float()
        e = e.masked_fill(attention_mask == 0, -1e9)

        attention_weights = F.softmax(e, dim=-1)

        toxicity_aware_repr = torch.bmm(
            attention_weights.unsqueeze(1), hidden_states
        ).squeeze(1)

        return toxicity_aware_repr, attention_weights


class DistilBERTWithATAM(nn.Module):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 6,
        dropout: float = 0.1,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        self.distilbert = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        hidden_size = self.distilbert.config.hidden_size

        self.atam = AdaptiveToxicAttentionModule(hidden_size, num_labels)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

        self._init_classifier()

    def _init_classifier(self):
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state

        cls_output = hidden_states[:, 0]

        toxicity_repr, attention_weights = self.atam(hidden_states, attention_mask)

        combined = torch.cat([cls_output, toxicity_repr], dim=-1)

        combined = self.dropout(combined)
        logits = self.classifier(combined)

        return logits

    def get_attention_weights(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = outputs.last_hidden_state
            _, attention_weights = self.atam(hidden_states, attention_mask)
        return attention_weights

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_pretrained(self, save_path: str):
        self.distilbert.save_pretrained(save_path)
        torch.save(self.classifier.state_dict(), f"{save_path}/classifier.pt")
        torch.save(self.atam.state_dict(), f"{save_path}/atam.pt")
        torch.save(
            {
                "dropout": self.dropout.p,
                "num_labels": self.num_labels,
                "hidden_size": self.distilbert.config.hidden_size,
            },
            f"{save_path}/head_config.pt",
        )

    @classmethod
    def from_pretrained(cls, load_path: str, model_name: str = None):
        import os
        head_config = torch.load(
            os.path.join(load_path, "head_config.pt"), map_location="cpu"
        )
        model = cls(
            model_name=model_name or "distilbert-base-uncased",
            num_labels=head_config["num_labels"],
            dropout=head_config["dropout"],
        )
        base_path = os.path.join(load_path, "distilbert")
        if os.path.exists(base_path):
            model.distilbert = model.distilbert.from_pretrained(base_path)
        else:
            model.distilbert = model.distilbert.from_pretrained(load_path)
        model.classifier.load_state_dict(
            torch.load(os.path.join(load_path, "classifier.pt"), map_location="cpu")
        )
        atam_path = os.path.join(load_path, "atam.pt")
        if os.path.exists(atam_path):
            model.atam.load_state_dict(
                torch.load(atam_path, map_location="cpu")
            )
        return model


def create_atam_model(
    config: dict,
    model_name: Optional[str] = None,
    num_labels: Optional[int] = None,
    dropout: Optional[float] = None,
) -> DistilBERTWithATAM:
    model_cfg = config["model"]
    return DistilBERTWithATAM(
        model_name=model_name or model_cfg["name"],
        num_labels=num_labels or model_cfg["num_labels"],
        dropout=dropout or model_cfg.get("dropout", 0.1),
        cache_dir=model_cfg.get("cache_dir"),
    )
