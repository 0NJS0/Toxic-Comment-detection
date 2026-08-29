"""
Resume-safe Parquet cache for the offline teacher pass.
========================================================

Stores per-comment teacher outputs (soft targets, evidence spans, anchor
position) as one row per comment id, and the re-tokenized prefix bank as
one row per (id, char_cut).

Why not a single monolithic file?
    The teacher pass runs on an intermittently available Kaggle GPU. If a
    session is interrupted, we must resume without re-running finished
    comments. Writing one Parquet shard per batch and scanning existing
    shards for already-done ids gives exactly that property.

On Kaggle the cache lives under ``results/llm_cache`` / ``results/prefix_cache``
inside ``/kaggle/working``. `/kaggle/working` is wiped between sessions, so
the notebook's Module 0 cells also snapshot these directories as a Kaggle
Dataset for cross-session resume (see ``utils/kaggle_utils.py``).
"""

import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


class TeacherCache:
    """Append-only, id-keyed Parquet cache with resume support."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._done_ids: Optional[set] = None

    # ------------------------------------------------------------------
    # Indexing (resume support)
    # ------------------------------------------------------------------
    @property
    def done_ids(self) -> set:
        if self._done_ids is None:
            self._done_ids = set()
            for shard in self.path.glob("part-*.parquet"):
                try:
                    ids = pd.read_parquet(shard, columns=["id"])["id"].astype(str)
                    self._done_ids.update(ids.tolist())
                except Exception:  # ignore corrupt/incomplete shards
                    continue
        return self._done_ids

    def is_done(self, comment_id: str) -> bool:
        return str(comment_id) in self.done_ids

    @property
    def size(self) -> int:
        return len(self.done_ids)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def append(self, rows: List[Dict]) -> int:
        """Append a batch of rows as a new shard. Returns rows written."""
        if not rows:
            return 0
        # Normalise ids to str up-front so the done-set stays consistent.
        for r in rows:
            r["id"] = str(r["id"])
        df = pd.DataFrame(rows)
        shard = self.path / f"part-{uuid.uuid4().hex}.parquet"
        df.to_parquet(shard, index=False)
        self.done_ids.update(df["id"].astype(str).tolist())
        return len(df)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Load every shard into one DataFrame (row order is shard order)."""
        shards = sorted(self.path.glob("part-*.parquet"))
        if not shards:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)

    def summary(self) -> str:
        return (f"  {self.path}: {self.size:,} comment ids cached "
                f"({len(list(self.path.glob('part-*.parquet'))):,} shards)")
