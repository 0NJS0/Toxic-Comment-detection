"""
Offline teacher pass: LLM -> soft targets + evidence spans + prefix bank.
===========================================================================

Pipeline (runs once on Kaggle; see Module 0 / ``module_00_teacher.py``):
    1. Stratified subsample of the cleaned training set
       (every toxic comment + a balanced sample of clean ones).
    2. For each comment not already in the cache, run the teacher LLM
       (Llama-3.2-3B in 4-bit) with a span-justification prompt.
    3. Parse soft targets + evidence spans, compute the anchor character.
    4. Build the *typing-faithful* prefix bank: re-tokenize the string cut
       at several character fractions and at the anchor char. This is what
       the student will be trained and evaluated on - never a cropped
       pre-tokenized sequence.
    5. Append results to resume-safe Parquet caches.

Everything is guard-railed by a wall-clock budget (``llm.max_minutes``) so an
interrupted Kaggle session stops cleanly and resumes from the cache.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from utils.config import load_config
from utils.helpers import set_random_seed
from llm.cache import TeacherCache
from llm.prompts import build_comment_prompt
from llm.spans import (
    parse_teacher_response,
    span_quality_stats,
)
from preprocessing.anchors import compute_anchor


# ---------------------------------------------------------------------------
# Teacher loading
# ---------------------------------------------------------------------------

def load_teacher(config: dict):
    """
    Load the teacher LLM for generation.

    - "mock": return None (deterministic fake spans, for local smoke tests).
    - GPU: 4/8-bit quantized via bitsandbytes (falls back to fp16 if the
      quantization library is missing).
    - CPU: unquantized fp32 (intentionally slow - only for tiny smoke runs).
    """
    llm_cfg = config["llm"]
    if llm_cfg.get("mock", False):
        return None

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import BitsAndBytesConfig

    model_id = llm_cfg["teacher"]
    cache_dir = llm_cfg.get("cache_dir", "models/cache")
    quant = llm_cfg.get("quant", "4bit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  Loading teacher: {model_id} (quant={quant}, device={device})")
    if device.type == "cuda" and quant in ("4bit", "8bit"):
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            print("  [hint] bitsandbytes not installed - using fp16 instead.")
            quant = "none"
        if quant != "none":
            load_kwargs = (
                {"load_in_4bit": True} if quant == "4bit"
                else {"load_in_8bit": True}
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=BitsAndBytesConfig(**load_kwargs),
                device_map="auto",
                cache_dir=cache_dir,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16,
                device_map="auto", cache_dir=cache_dir,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return {"model": model, "tokenizer": tokenizer, "device": device}


def load_student_tokenizer(config):
    """DistilBERT tokenizer: prefer the Module-2 saved one, else HF."""
    from transformers import AutoTokenizer
    saved = Path(config["data"]["processed"]["path"]) / "tokenized" / "tokenizer"
    if saved.exists():
        return AutoTokenizer.from_pretrained(str(saved))
    return AutoTokenizer.from_pretrained(config["model"]["name"])


# ---------------------------------------------------------------------------
# Subsample
# ---------------------------------------------------------------------------

def build_subsample(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Stratified subsample: all toxic + balanced clean up to target size."""
    llm_cfg = config["llm"]
    labels = config["data"]["labels"]
    seed = config["project"]["seed"]
    size = int(llm_cfg.get("subsample_size", 25000))

    is_toxic = df[labels].sum(axis=1) > 0
    toxic = df[is_toxic]
    clean = df[~is_toxic]

    if llm_cfg.get("stratify_toxic", True):
        take_toxic = min(len(toxic), size)
        take_clean = min(len(clean), max(0, size - take_toxic))
    else:
        take_toxic = min(len(toxic), size)
        take_clean = 0

    rng = np.random.RandomState(seed)
    toxic_sel = toxic if take_toxic >= len(toxic) else toxic.iloc[
        rng.choice(len(toxic), take_toxic, replace=False)]
    clean_sel = clean if take_clean >= len(clean) else clean.iloc[
        rng.choice(len(clean), take_clean, replace=False)]
    sub = pd.concat([toxic_sel, clean_sel]).reset_index(drop=True)

    n_toxic = int(sub[labels].sum(axis=1).gt(0).sum())
    print(f"  Subsample: {len(sub):,} comments ({n_toxic:,} toxic, "
          f"{len(sub) - n_toxic:,} clean)")
    return sub


# ---------------------------------------------------------------------------
# Prefix bank
# ---------------------------------------------------------------------------

def build_prefix_rows(
    comment_id: str,
    text: str,
    tokenizer,
    config: dict,
    anchor_char: Optional[int],
) -> List[Dict]:
    """
    Re-tokenize typing-faithful char prefixes of ``text``.

    A char prefix is NOT a crop of a pre-tokenized sequence: the tokenizer
    sees only the characters typed so far, matching runtime inference.
    Returns one row per unique char_cut.
    """
    prefix_cfg = config["prefix"]
    fractions = [float(f) for f in prefix_cfg.get("fractions", [0.25, 0.5, 0.75, 1.0])]
    include_anchor = prefix_cfg.get("include_anchor", True)
    max_length = int(config["data"]["max_seq_length"])

    n = max(len(text), 1)
    cuts = {int(n * f) for f in fractions if 0 < int(n * f) <= n}
    if include_anchor and anchor_char is not None and 0 < anchor_char <= n:
        cuts.add(int(anchor_char))

    rows = []
    for char_cut in sorted(cuts):
        prefix_text = text[:char_cut]
        enc = tokenizer(
            prefix_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        rows.append({
            "id": str(comment_id),
            "char_cut": int(char_cut),
            "is_anchor": bool(anchor_char is not None and int(char_cut) == int(anchor_char)),
            "prefix_text": prefix_text,
            "input_ids": enc["input_ids"][0].tolist(),
            "attention_mask": enc["attention_mask"][0].tolist(),
        })
    return rows


# ---------------------------------------------------------------------------
# Mock teacher (deterministic, for local smoke tests)
# ---------------------------------------------------------------------------

_TOXIC_WORDS = ["idiot", "stupid", "ugly", "hate", "trash", "kill", "moron"]


def _mock_teacher_response(comment: str, labels: List[str]) -> str:
    """Deterministic fake span output so the pipeline is testable on CPU."""
    simple = comment.lower()
    out = {}
    for label in labels:
        matches = [(simple.find(w), w) for w in _TOXIC_WORDS if simple.find(w) >= 0]
        if label == "toxic" and matches:
            start, word = min(matches)
            span = [start, start + len(word)]
            out[label] = {"present": True, "conf": 0.95, "spans": [span]}
        else:
            out[label] = {"present": False, "conf": 0.05, "spans": []}
    return json.dumps(out)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_teacher_pass(config: dict) -> Dict:
    """
    Execute the offline teacher pass with resume + budget guards.

    Returns a summary dict (counts, budget-hit flag, quality stats).
    """
    t_start = time.time()
    max_minutes = float(config["llm"].get("max_minutes", 300))
    labels = config["data"]["labels"]
    batch_size = int(config["llm"].get("batch_size", 16))
    flush_batches = int(config["llm"].get("flush_batches", 10))

    set_random_seed(config["project"]["seed"])

    # --- 1. subsample --------------------------------------------------
    df = pd.read_csv(Path(config["data"]["processed"]["path"])
                     / config["data"]["processed"]["train_file"])
    subsample = build_subsample(df, config)

    # --- 2. caches -----------------------------------------------------
    llm_cache = TeacherCache(config["llm"]["output_dir"])
    prefix_cache = TeacherCache(config["prefix"]["output_dir"])
    print(llm_cache.summary())
    print(prefix_cache.summary())

    # --- 3. teacher + student tokenizer --------------------------------
    teacher = load_teacher(config)
    student_tok = load_student_tokenizer(config)

    # --- 4. loop over undone comments -----------------------------------
    todo = subsample[~subsample["id"].astype(str).isin(llm_cache.done_ids)]
    n_todo = len(todo)
    print(f"\n  To do: {n_todo:,} comments (of {len(subsample):,})")

    processed = 0
    budget_hit = False
    llm_rows: List[Dict] = []
    prefix_rows: List[Dict] = []
    mock = config["llm"].get("mock", False)

    batches_since_flush = 0
    for idx in range(0, n_todo, batch_size):
        batch = todo.iloc[idx:idx + batch_size]
        texts = batch["clean_text"].astype(str).tolist()
        ids = batch["id"].astype(str).tolist()

        if teacher is not None:
            model, tokenizer, device = teacher["model"], teacher["tokenizer"], teacher["device"]
            messages_batch = [build_comment_prompt(t, labels) for t in texts]
            rendered = tokenizer.apply_chat_template(
                messages_batch, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(rendered, return_tensors="pt",
                               padding=True, truncation=True).to(device)
            with torch.inference_mode():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=int(config["llm"].get("generation_max_new_tokens", 220)),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                raw_responses = tokenizer.batch_decode(
                    gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                )
        else:
            raw_responses = [_mock_teacher_response(t, labels) for t in texts]

        for cid, text, raw in zip(ids, texts, raw_responses):
            rec = parse_teacher_response(raw, labels, len(text))
            anchor_char, anchor_token = compute_anchor(rec["spans"], student_tok, text)

            llm_rows.append({
                "id": cid,
                "text": text,
                "soft_targets": json.dumps(rec["soft_targets"]),
                "spans": json.dumps(rec["spans"]),
                "quality": rec["quality"],
                "anchor_char": anchor_char,
                "anchor_token": anchor_token,
            })
            prefix_rows.extend(build_prefix_rows(cid, text, student_tok, config, anchor_char))

        processed += len(batch)
        batches_since_flush += 1
        if batches_since_flush >= flush_batches:
            llm_cache.append(llm_rows)
            prefix_cache.append(prefix_rows)
            llm_rows, prefix_rows = [], []
            batches_since_flush = 0

        spent = (time.time() - t_start) / 60.0
        print(f"  [{processed:,}/{n_todo:,}] done, {spent:.1f} min ... "
              f"(cache: {llm_cache.size:,})")
        if spent >= max_minutes:
            budget_hit = True
            print(f"  [budget] {max_minutes:.0f} min reached - stopping cleanly. "
                  f"Re-run this cell to resume from the cache.")
            break

    # final flush
    llm_cache.append(llm_rows)
    prefix_cache.append(prefix_rows)

    # --- 5. summary ----------------------------------------------------
    cached = llm_cache.load()
    qstats = {"ok": 0, "weak": 0, "no_evidence": 0}
    if len(cached):
        qstats = span_quality_stats(cached.to_dict("records"), labels)
    print("\n" + "=" * 60)
    print("TEACHER PASS SUMMARY")
    print("=" * 60)
    print(llm_cache.summary())
    print(prefix_cache.summary())
    print(f"  quality: {qstats}")
    print(f"  budget_hit: {budget_hit}")

    return {
        "comments_done": llm_cache.size,
        "prefix_ids": prefix_cache.size,
        "budget_hit": budget_hit,
        "quality": qstats,
        "elapsed_min": (time.time() - t_start) / 60.0,
    }


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    run_teacher_pass(load_config(config_path))