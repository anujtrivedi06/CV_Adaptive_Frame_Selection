"""Oracle selectors - the upper bound on what any selector could achieve.

The oracle is the most important baseline in the project: it says how much
headroom exists at all. If the oracle at K=4 is barely above uniform at K=4,
then no amount of clever selection will help on that dataset, and *that* is a
finding worth reporting.

Two flavours:

``needle``  exact, free, toy dataset only - we know the evidence frames.
``greedy``  generic: forward search driven by a caller-supplied score function
            (normally the reader's likelihood of the correct answer). Greedy
            rather than exhaustive, so it is a *lower bound on the oracle* -
            always describe it that way in the report.
"""

from __future__ import annotations

import numpy as np

from afs.registry import SELECTORS
from afs.selectors.base import SelectionContext, Selector
from afs.selectors.baselines import UniformSelector


@SELECTORS.register("oracle_needle")
class NeedleOracleSelector(Selector):
    """Cover every evidence window first, then pad uniformly.

    Round-robin across the groups rather than draining the first one, so that
    with a budget of 2 and two required moments the oracle takes one from each
    - which is precisely the behaviour we want the real selector to learn.
    """

    def __init__(self, **_: object) -> None:
        self._uniform = UniformSelector()

    def select(self, ctx: SelectionContext) -> np.ndarray:
        from afs.data.base import evidence_groups

        groups = [
            [i for i in g if 0 <= i < ctx.n_frames]
            for g in evidence_groups(ctx.sample)
        ]
        groups = [g for g in groups if g]
        k = min(ctx.budget, ctx.n_frames)
        if not groups:
            return self._uniform.select(ctx)

        picked: list[int] = []
        depth = 0
        while len(picked) < k and any(len(g) > depth for g in groups):
            for g in groups:
                if len(picked) >= k:
                    break
                if len(g) > depth:
                    picked.append(int(g[depth]))
            depth += 1

        if len(picked) < k:
            for i in self._uniform.select(ctx):
                if len(picked) >= k:
                    break
                if int(i) not in picked:
                    picked.append(int(i))
        return np.array(sorted(picked[:k]), dtype=int)


@SELECTORS.register("oracle_greedy")
class GreedyOracleSelector(Selector):
    """Forward-greedy search using ``ctx.score_fn``.

    ``candidate_pool`` keeps the reader-call count sane: we only consider a
    uniformly spaced pool of P candidates, so the cost is K * P reader calls
    per question rather than K * N.
    """

    def __init__(self, candidate_pool: int = 32, **_: object) -> None:
        self.candidate_pool = int(candidate_pool)

    def select(self, ctx: SelectionContext) -> np.ndarray:
        if ctx.score_fn is None:
            raise ValueError(
                "oracle_greedy needs ctx.score_fn (the evaluator supplies it)"
            )
        n = ctx.n_frames
        k = min(ctx.budget, n)
        pool = np.unique(
            np.linspace(0, n - 1, min(self.candidate_pool, n)).round().astype(int)
        )

        chosen: list[int] = []
        best_score = -np.inf
        for _ in range(k):
            best_i, best_s = None, best_score
            for i in pool:
                if i in chosen:
                    continue
                s = ctx.score_fn(np.array(sorted(chosen + [int(i)]), dtype=int))
                if s > best_s:
                    best_s, best_i = s, int(i)
            if best_i is None:  # no candidate improves the score
                break
            chosen.append(best_i)
            best_score = best_s
        if not chosen:
            chosen = [int(pool[len(pool) // 2])]
        return np.array(sorted(chosen), dtype=int)
