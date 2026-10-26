"""The baselines every result must be compared against."""

from __future__ import annotations

import numpy as np

from afs.registry import SELECTORS
from afs.selectors.base import SelectionContext, Selector
from afs.utils import stable_seed


@SELECTORS.register("uniform")
class UniformSelector(Selector):
    """Evenly spaced frames - what essentially every video model does today."""

    def __init__(self, **_: object) -> None:
        pass

    def select(self, ctx: SelectionContext) -> np.ndarray:
        n, k = ctx.n_frames, min(ctx.budget, ctx.n_frames)
        if k == 1:
            return np.array([n // 2])
        # centres of k equal segments: avoids the free win of always taking frame 0
        edges = np.linspace(0, n, k + 1)
        return ((edges[:-1] + edges[1:]) / 2).astype(int).clip(0, n - 1)


@SELECTORS.register("random")
class RandomSelector(Selector):
    def __init__(self, seed: int = 0, **_: object) -> None:
        self.seed = seed

    def select(self, ctx: SelectionContext) -> np.ndarray:
        rng = np.random.default_rng(stable_seed(self.seed, ctx.sample.qid))
        n, k = ctx.n_frames, min(ctx.budget, ctx.n_frames)
        return rng.choice(n, size=k, replace=False)


@SELECTORS.register("stride")
class StrideSelector(Selector):
    """Fixed stride from the start - the naive 'first K seconds' failure mode."""

    def __init__(self, stride: int = 1, **_: object) -> None:
        self.stride = max(1, int(stride))

    def select(self, ctx: SelectionContext) -> np.ndarray:
        n, k = ctx.n_frames, min(ctx.budget, ctx.n_frames)
        return np.arange(0, n, self.stride)[:k]


@SELECTORS.register("topk")
class TopKSelector(Selector):
    """Pure query relevance, no diversity term.

    This is the ablation that shows why the project is a subset-selection
    problem: the top-K frames are usually near-duplicates from one window.
    """

    needs_query = True

    def __init__(self, **_: object) -> None:
        pass

    def select(self, ctx: SelectionContext) -> np.ndarray:
        r = ctx.relevance()
        k = min(ctx.budget, ctx.n_frames)
        return np.argsort(-r)[:k]
