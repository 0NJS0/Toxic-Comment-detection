"""
Anchor computation for evidence-anchored early detection.
==========================================================

An *anchor* is the earliest character position at which a perfect reader -
knowing the teacher's evidence spans - could already decide the comment is
toxic. Given per-label evidence spans (character offsets), we:

    1. take the earliest start position across all labels,
    2. map that character offset to the first token that covers it.

This single scalar replaces arbitrary prefix thresholds (8/16/32 tokens) as
the principled target for "when should the student be confident?", and is
used both for training supervision (anchor-weighted BCE) and evaluation
(F1@anchor, detection delay vs the anchor).
"""

from typing import Dict, List, Optional, Tuple

from llm.spans import earliest_span_char


def compute_anchor(
    spans: Dict[str, List[List[int]]],
    tokenizer,
    text: str,
    max_length: int = 256,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Return (anchor_char, anchor_token) from evidence spans.

    ``anchor_char`` is None for non-anchorable comments (no usable spans).
    ``anchor_token`` is the index of the first token (ignoring special
    tokens, in the un-prefixed full-text tokenization) that covers the
    anchor character - informational; the prefix bank is re-tokenized at
    the anchor directly and does not depend on it.
    """
    earliest = earliest_span_char(spans)
    if earliest is None:
        return None, None
    anchor_char = earliest[0]

    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    offsets = enc["offset_mapping"]
    anchor_token: Optional[int] = None
    for i, (s, e) in enumerate(offsets):
        if e > anchor_char:
            anchor_token = i
            break
    if anchor_token is None:
        anchor_token = len(offsets) if offsets else 0

    return anchor_char, anchor_token