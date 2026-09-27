"""
Tolerant parsing + normalization of teacher responses.
========================================================

The teacher is asked for one JSON object per comment; in practice the raw
generation may include prose, markdown fences, or partial JSON. This module
extracts the JSON defensively, normalizes spans against the actual comment
length, and assigns a per-comment quality flag that the training stage /
attention analysis can use to decide which items are anchorable.

Normalized record per comment:
    {
        "soft_targets": [conf, ...] per label (in config label order),
        "spans":        {label: [[start, end), ...]},
        "quality":      "ok" | "weak" | "no_evidence",
    }
"""

import json
import re
from typing import Dict, List, Optional, Tuple


def extract_json_object(text: str) -> Optional[Dict]:
    """Pull the first top-level JSON object out of noisy model output."""
    if not text:
        return None
    # Strip markdown fences if the model wrapped the answer.
    stripped = re.sub(r"```(?:json)?", "", text)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    cand = stripped[start:end + 1]
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        return None


def _normalize_span(raw, text_len: int) -> Optional[List[int]]:
    """Verify + clamp a span to [0, text_len). Returns None if unusable."""
    try:
        s, e = int(raw[0]), int(raw[1])
    except (TypeError, ValueError, IndexError):
        return None
    s = max(0, min(s, text_len))
    e = max(0, min(e, text_len))
    if e <= s:
        return None
    return [s, e]


def parse_teacher_response(
    raw_text: str,
    labels: List[str],
    comment_len: int,
) -> Dict:
    """Parse a raw teacher answer into a normalized record."""
    soft_targets = [0.0] * len(labels)
    spans: Dict[str, List[List[int]]] = {}
    all_empty = True

    obj = extract_json_object(raw_text)
    if obj is None:
        return _record(soft_targets, spans, "no_evidence")

    for i, label in enumerate(labels):
        entry = obj.get(label)
        if not isinstance(entry, dict):
            continue

        present = bool(entry.get("present"))
        conf = entry.get("conf")
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.5 if present else 0.0
        conf = min(1.0, max(0.0, conf))
        soft_targets[i] = conf if present else min(conf, 1 - soft_targets[i])

        raw_spans = entry.get("spans") or []
        cleaned = []
        for raw_span in raw_spans:
            span = _normalize_span(raw_span, comment_len)
            if span:
                cleaned.append(span)
        if present and cleaned:
            all_empty = False
            spans[label] = cleaned
        else:
            spans[label] = []

    quality = "ok" if not all_empty else "no_evidence"
    return _record(soft_targets, spans, quality)


def _record(
    soft_targets: List[float],
    spans: Dict[str, List[List[int]]],
    quality: str,
) -> Dict:
    return {"soft_targets": soft_targets, "spans": spans, "quality": quality}


def earliest_span_char(spans: Dict[str, List[List[int]]]) -> Optional[Tuple[int, str]]:
    """
    Return (earliest_char, label) across all labels' spans, or None.

    This is the anchored "first moment a reader could already know".
    """
    best: Optional[Tuple[int, str]] = None
    for label, label_spans in spans.items():
        for s, e in label_spans:
            if best is None or s < best[0]:
                best = (s, label)
    return best


def span_quality_stats(
    rows: List[Dict],
    labels: List[str],
) -> Dict[str, int]:
    """Aggregate quality flags over cached rows for the run summary."""
    counts = {"ok": 0, "weak": 0, "no_evidence": 0}
    for r in rows:
        counts[r.get("quality", "no_evidence")] += 1
    return counts