"""Readers: the frozen downstream model that answers from the selected frames.

Frozen is the whole point - we never fine-tune it, so every accuracy difference
between runs is caused by frame selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from afs.data.base import QASample


@dataclass
class ReaderOutput:
    pred_idx: int
    scores: Optional[List[float]] = None   # per-option score, if available
    latency_s: float = 0.0

    @property
    def confidence(self) -> float:
        if not self.scores:
            return float("nan")
        s = np.asarray(self.scores, dtype=np.float64)
        e = np.exp(s - s.max())
        p = e / e.sum()
        return float(p.max())


class Reader:
    """Answers a multiple-choice question given a subset of frames."""

    registry_name: str = "base"
    #: approximate cost of one frame, for the compute axis of our Pareto plots
    gflops_per_frame: float = 0.0

    def answer(self, sample: QASample, frames: np.ndarray, frame_idx: np.ndarray) -> ReaderOutput:
        raise NotImplementedError

    def score_correct(
        self, sample: QASample, frames: np.ndarray, frame_idx: np.ndarray
    ) -> float:
        """Score used by the greedy oracle: how much this frame set supports the
        *ground-truth* answer. Defaults to 1.0/0.0 from a hard prediction."""
        out = self.answer(sample, frames, frame_idx)
        if out.scores:
            s = np.asarray(out.scores, dtype=np.float64)
            e = np.exp(s - s.max())
            return float((e / e.sum())[sample.answer_idx])
        return float(out.pred_idx == sample.answer_idx)

    def cost_gflops(self, n_frames: int) -> float:
        return self.gflops_per_frame * n_frames
