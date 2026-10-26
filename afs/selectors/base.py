"""Selector interface.

A selector sees only cached features and a budget, never pixels and never the
reader. That isolation is what makes the accuracy differences in our results
attributable to selection alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from afs.data.base import QASample


@dataclass
class SelectionContext:
    sample: QASample
    frame_feats: np.ndarray            # [N, d] unit-norm
    times: np.ndarray                  # [N] seconds
    query: Optional[np.ndarray] = None  # [d] unit-norm, or None (task-agnostic)
    budget: int = 8
    # Only the oracle uses this: maps a candidate index set -> a score to maximise.
    score_fn: Optional[Callable[[np.ndarray], float]] = None

    @property
    def n_frames(self) -> int:
        return int(self.frame_feats.shape[0])

    def relevance(self, normalize: bool = True) -> np.ndarray:
        """Cosine similarity of every frame to the query, optionally rescaled
        to [0, 1] within this video (cosine dynamic range is small and varies a
        lot per video, so raw values make the diversity weight unstable)."""
        if self.query is None:
            return np.zeros(self.n_frames, dtype=np.float32)
        r = self.frame_feats @ self.query.astype(np.float32)
        if normalize and len(r) > 1:
            lo, hi = float(r.min()), float(r.max())
            if hi - lo > 1e-6:
                r = (r - lo) / (hi - lo)
            else:
                r = np.zeros_like(r)
        return r.astype(np.float32)


class Selector:
    """Base class. ``select`` returns sorted frame indices, at most ``budget``."""

    needs_query: bool = False
    registry_name: str = "base"

    def select(self, ctx: SelectionContext) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, ctx: SelectionContext) -> np.ndarray:
        k = min(int(ctx.budget), ctx.n_frames)
        if k <= 0:
            return np.zeros((0,), dtype=int)
        idx = np.asarray(self.select(ctx), dtype=int)
        idx = np.unique(idx)
        return np.sort(idx[:k])
