"""
FedPref Simulation — evaluation module for Streamlit UI.
"""

from .predictor import ToxicityPredictor, load_predictor
__all__ = ["ToxicityPredictor", "load_predictor"]
