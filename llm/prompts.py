"""
Prompt construction for the offline teacher pass.
===================================================

The teacher (a small open-weight LLM such as Llama-3.2-3B) reads the *full*
comment text once and is asked, for each of the Jigsaw labels, to:
    1. decide whether the label is present,
    2. quote the exact character span(s) of text that justify that decision,
    3. give a 0.0-1.0 confidence/soft target.

The quoted spans are what later define the *anchor*: the earliest character
position at which the final decision is already knowable. Nothing in this
module depends on the downstream student.

`span_only=True` (the default) requests a compact JSON-only answer, which
keeps generation short and therefore cheap on an intermittent GPU.
"""

from typing import List, Optional

SYSTEM_PROMPT = (
    "You are a toxicity annotation engine. Read the comment and, for every "
    "label, decide if it is present. Quote the exact characters (substring) "
    "that support your decision - nothing more. Respond only with valid JSON."
)

LABEL_INSTRUCTIONS = """
For each label output an object with:
  - "present": true or false
  - "conf": a float 0.0-1.0 confidence that the label is present
  - "spans": list of [start, end) character offsets into the comment
            that justify the decision; empty list if not present
            (or if you cannot justify reliably)
Return a single JSON object, e.g.:
{"toxic": {"present": true, "conf": 0.95, "spans": [[14, 19]]},
 "severe_toxic": {"present": false, "conf": 0.05, "spans": []},
 ...
}
"""


def build_comment_prompt(
    comment: str,
    labels: List[str],
    span_only: bool = True,
    max_spans: int = 2,
) -> List[dict]:
    """
    Build chat messages for ``tokenizer.apply_chat_template``.

    Returns a list of ``{"role": ..., "content": ...}`` messages appropriate
    for an instruction-tuned model.
    """
    body = (
        f"Comment: {comment}\n\n"
        f"Labels to evaluate: {', '.join(labels)}\n"
    )
    if span_only:
        body += (
            "\nFor each label give present/conf/spans. Keep the JSON compact "
            f"(at most {max_spans} spans per label). No prose before or after the JSON."
        )
    else:
        body += LABEL_INSTRUCTIONS
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": body},
    ]