"""
FedPref Interactive Simulation (Streamlit UI)
===============================================

A web-based demo for while-typing toxicity detection.

Features:
  - Type text and see per-label toxicity scores update in real time
  - Slider to control prefix length (early detection simulation)
  - Compare predictions at different prefix lengths
  - Curated test examples

Usage:
    uv run streamlit run simulation/app.py
"""

import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from simulation.predictor import ToxicityPredictor, load_predictor
import pandas as pd


# ---- Page config ----
st.set_page_config(
    page_title="FedPref — While-Typing Toxicity Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ FedPref: While-Typing Toxicity Detection")
st.markdown(
    "Type a comment below. The model predicts toxicity in **real time** "
    "as you type, from partial prefixes and from the full text."
)


# ---- Load model ----
@st.cache_resource
def load_model():
    checkpoint_path = "results/training/best_model.pt"
    if not Path(checkpoint_path).exists():
        st.error(
            f"Checkpoint not found at {checkpoint_path}. "
            "Run `uv run python module_03_training.py` first."
        )
        st.stop()
    return load_predictor(checkpoint_path)


predictor = load_model()


# ---- Label display config ----
LABEL_COLORS = {
    "toxic": "#ff4b4b",
    "severe_toxic": "#d62728",
    "obscene": "#ff7f0e",
    "threat": "#8c564b",
    "insult": "#e377c2",
    "identity_hate": "#9467bd",
}


# ---- Sidebar: model info & settings ----
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Detection threshold",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    )
    predictor.threshold = threshold

    max_seq_length = st.slider(
        "Max token length", min_value=32, max_value=256, value=192, step=32,
    )

    st.header("📊 Model Info")
    total_params = sum(p.numel() for p in predictor.model.parameters())
    trainable_params = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
    st.metric("Total parameters", f"{total_params:,}")
    st.metric("Trainable parameters", f"{trainable_params:,}")
    st.metric("Device", str(predictor.device).upper())

    st.header("📝 Example Comments")
    examples = [
        "I love this! Great work!",
        "You are an idiot and a moron.",
        "This is the worst thing I've ever seen.",
        "I respectfully disagree with your point.",
        "Go kill yourself, you piece of trash.",
        "Your argument is flawed because...",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["input_text"] = ex


# ---- Main area ----
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("✏️ Enter text")
    default_text = st.session_state.get("input_text", "")
    text = st.text_area(
        "Comment",
        value=default_text,
        height=150,
        placeholder="Type a comment here...",
        label_visibility="collapsed",
    )

    if text.strip():
        # Full-text prediction
        result = predictor.predict(text)

        # Prefix-based prediction
        prefix_len = st.slider(
            "Prefix length (tokens)",
            min_value=8, max_value=max_seq_length, value=64, step=8,
        )
        prefix_result = predictor.predict_prefix(text, prefix_len)

    else:
        result = None
        prefix_result = None

    # ---- Prefix comparison section ----
    if text.strip():
        st.subheader("📈 Detection Delay Analysis")
        st.markdown("How early can the model detect toxicity?")

        comparisons = predictor.compare_prefixes(text)
        comp_data = []
        for r in comparisons:
            prefix_label = f"Full" if r["prefix_length_tokens"] == r["full_length_tokens"] else f"{r['prefix_length_tokens']} tok"
            comp_data.append({
                "Prefix": prefix_label,
                "Toxic?": "⚠️ Yes" if r["is_toxic"] else "✅ No",
                "Max Prob": f"{r['max_probability']:.3f}",
                "Num Labels": r["num_toxic_labels"],
            })

        st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

        char_equiv_8 = 8 * 4.5
        stub = "  *(enough for keyboard rejection)*" if result and result.get("is_toxic") else ""
        st.caption(f"8 tokens ≈ {char_equiv_8:.0f} characters typed {stub}")

with col2:
    st.subheader("🔍 Toxicity Scores")

    if text.strip() and result:
        # Full-text display
        st.markdown("**Full Text**")
        for label in ToxicityPredictor.LABEL_NAMES:
            prob = result["probabilities"][label]
            is_binary = result["binary"][label]
            color = LABEL_COLORS.get(label, "#888")
            bar_pct = prob * 100

            st.markdown(
                f"<div style='margin-bottom: 8px;'>"
                f"  <div style='display: flex; justify-content: space-between;'>"
                f"    <span style='color: {color}; font-weight: bold;'>{label}</span>"
                f"    <span>{'⚠️' if is_binary else '✓'} {prob:.3f}</span>"
                f"  </div>"
                f"  <div style='background: #eee; border-radius: 4px; height: 12px; width: 100%;'>"
                f"    <div style='background: {color}; width: {bar_pct}%; height: 12px; border-radius: 4px;'></div>"
                f"  </div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if result["is_toxic"]:
            st.error("🚨 **Toxic content detected!**")
        else:
            st.success("✅ **No toxic content detected**")

        # Prefix display
        st.markdown("---")
        st.markdown(f"**Prefix ({prefix_result['prefix_length_tokens']} tokens)**")
        for label in ToxicityPredictor.LABEL_NAMES:
            prob = prefix_result["probabilities"][label]
            is_binary = prefix_result["binary"][label]
            color = LABEL_COLORS.get(label, "#888")
            bar_pct = prob * 100

            st.markdown(
                f"<div style='margin-bottom: 8px;'>"
                f"  <div style='display: flex; justify-content: space-between;'>"
                f"    <span style='color: {color};'>{label}</span>"
                f"    <span>{'⚠️' if is_binary else '✓'} {prob:.3f}</span>"
                f"  </div>"
                f"  <div style='background: #eee; border-radius: 4px; height: 12px; width: 100%;'>"
                f"    <div style='background: {color}; width: {bar_pct}%; height: 12px; border-radius: 4px;'></div>"
                f"  </div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    else:
        st.info("Type a comment to see predictions.")


# ---- Footer ----
st.markdown("---")
st.markdown(
    "**FedPref** — Federated Prefix-Based Early Toxicity Detection | "
    "Model: DistilBERT + LoRA | "
    "[GitHub](https://github.com/anomalyco/Toxic-Comment-detection)"
)
