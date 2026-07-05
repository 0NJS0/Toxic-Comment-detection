"""
Live Toxicity Predictor with Prefix Support
=============================================

Wraps a trained model and provides:
  - Full-text toxicity prediction
  - Prefix-based prediction (while-typing simulation)
  - Per-label probability breakdown
  - Agreement between prefix and full-text predictions
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class ToxicityPredictor:
    """
    Wraps a trained model for live (while‑typing) toxicity detection.

    Usage:
        predictor = ToxicityPredictor(model, tokenizer, device)
        result = predictor.predict("some text")
        result_by_char = predictor.predict_at_prefix("some text", prefix_pct=0.5)
    """

    LABEL_NAMES = [
        "toxic", "severe_toxic", "obscene",
        "threat", "insult", "identity_hate",
    ]

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: torch.device = torch.device("cpu"),
        threshold: float = 0.5,
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.threshold = threshold

    def predict(self, text: str) -> Dict:
        """
        Predict toxicity for the full text.

        Returns
        -------
        dict: {
            "text": str,
            "probabilities": dict of label -> float,
            "binary": dict of label -> bool,
            "is_toxic": bool,
            "num_toxic_labels": int,
            "max_probability": float,
        }
        """
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=192,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        return self._format_result(text, probs)

    def predict_prefix(
        self, text: str, prefix_length_tokens: Optional[int]
    ) -> Dict:
        """
        Predict toxicity from a prefix of the text.

        For while‑typing simulation:
          - ``prefix_length_tokens``: number of tokens to use
          - If ``None``: uses full text

        Returns same dict as ``predict()``, but also includes the prefix info.
        """
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=192,
        )
        input_ids = encoded["input_ids"][0]   # 1D
        attn_mask = encoded["attention_mask"][0]

        full_len = len(input_ids)

        if prefix_length_tokens is None or prefix_length_tokens >= full_len:
            chosen_len = full_len
        else:
            chosen_len = prefix_length_tokens

        prefix_ids = input_ids[:chosen_len].unsqueeze(0).to(self.device)
        prefix_mask = attn_mask[:chosen_len].unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(prefix_ids, attention_mask=prefix_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        result = self._format_result(text, probs)
        result["prefix_length_tokens"] = chosen_len
        result["full_length_tokens"] = full_len
        return result

    def predict_at_char(
        self, text: str, prefix_pct: float = 0.5
    ) -> Dict:
        """
        Predict toxicity at a given character prefix percentage.
        Useful for the interactive slider in the UI.
        """
        char_len = max(1, int(len(text) * prefix_pct))
        prefix_text = text[:char_len]
        return self.predict(prefix_text)

    def compare_prefixes(self, text: str) -> List[Dict]:
        """
        Predict at multiple prefix lengths and return comparison.
        Used for detection-delay analysis in the UI.
        """
        results = []
        for plen in [8, 16, 32, 64, 128, None]:
            r = self.predict_prefix(text, plen)
            results.append(r)
        return results

    def _format_result(self, text: str, probs: np.ndarray) -> Dict:
        binary = {name: bool(probs[i] >= self.threshold)
                  for i, name in enumerate(self.LABEL_NAMES)}
        probs_dict = {name: float(probs[i])
                      for i, name in enumerate(self.LABEL_NAMES)}

        return {
            "text": text,
            "probabilities": probs_dict,
            "binary": binary,
            "is_toxic": any(binary.values()),
            "num_toxic_labels": sum(binary.values()),
            "max_probability": float(probs.max()),
        }


def load_predictor(
    model_path: str,
    config_path: str = "configs/config.yaml",
    device: Optional[torch.device] = None,
) -> ToxicityPredictor:
    """
    Convenience: load a saved model and return a ToxicityPredictor.

    Parameters
    ----------
    model_path : str
        Path to the .pt checkpoint.
    config_path : str
        Path to config YAML.
    device : torch.device or None
        Defaults to CUDA if available else CPU.

    Returns
    -------
    ToxicityPredictor
    """
    from utils.config import load_config
    from models.registry import create_model

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(config_path)
    label_cols = config["data"]["labels"]

    model = create_model(
        config,
        model_name=config["model"]["name"],
        num_labels=len(label_cols),
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer_path = Path(config["data"]["processed"]["path"]) / "tokenized" / "tokenizer"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    return ToxicityPredictor(model, tokenizer, device=device)
