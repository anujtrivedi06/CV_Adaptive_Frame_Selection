"""Our method: query-conditioned submodular frame selection.

We maximise, over subsets S with |S| <= K,

    F(S) = alpha * sum_{i in S} r_i                         (relevance, modular)
         + beta  * (1/N) sum_{j=1..N} max_{i in S} s_ij     (semantic coverage)
         + gamma * (1/M) sum_{b=1..M} max_{i in S} k(t_i,b) (temporal coverage)

Each term is monotone and submodular (a modular sum, and two facility-location
functions), so F is monotone submodular. Greedy forward selection therefore
attains at least (1 - 1/e) ~ 0.632 of the optimum under the cardinality
constraint - a guarantee pure top-K ranking does not have, and one worth a
paragraph in the report.

Cost is O(K*N^2) with N <= ~256 candidate frames, i.e. microseconds on CPU.
"""

from __future__ import annotations

import numpy as np

from afs.registry import SELECTORS
from afs.selectors.base import SelectionContext, Selector


def temporal_kernel(times: np.ndarray, n_bins: int, sigma_frac: float) -> np.ndarray:
    """``[N, n_bins]`` coverage of each frame over an even grid of time bins."""
    t = np.asarray(times, dtype=np.float64)
    if len(t) == 0:
        return np.zeros((0, n_bins), dtype=np.float32)
    lo, hi = float(t.min()), float(t.max())
    span = max(hi - lo, 1e-6)
    centres = np.linspace(lo, hi, n_bins)
    sigma = max(span * sigma_frac, 1e-6)
    d = (t[:, None] - centres[None, :]) / sigma
    return np.exp(-0.5 * d * d).astype(np.float32)


def greedy_submodular(
    relevance: np.ndarray,
    sim: np.ndarray | None,
    temporal: np.ndarray | None,
    budget: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Greedy maximisation of the objective above. Returns chosen indices.

    Each of the three terms is rescaled to [0, 1] before weighting: relevance
    by the budget K, and each facility-location term by the coverage the full
    frame set would achieve. Without this the terms are on wildly different
    scales (relevance grows with |S|, coverage saturates at 1) and alpha/beta/
    gamma stop being interpretable trade-off weights. Rescaling by positive
    constants preserves monotonicity and submodularity, so the (1 - 1/e)
    guarantee is unaffected.
    """
    n = len(relevance)
    k = int(min(budget, n))
    if k <= 0:
        return np.zeros((0,), dtype=int)

    use_sim = sim is not None and beta > 0
    use_tmp = temporal is not None and gamma > 0

    # Facility location is defined with F(empty) = 0 and non-negative
    # similarities, so we clip cosine at zero and start the running maxima at
    # zero. (Starting them at -inf would make every first gain infinite.)
    if use_sim:
        sim = np.maximum(sim, 0.0).astype(np.float32)
    cur_sim = np.zeros(n, dtype=np.float32) if use_sim else None
    cur_tmp = np.zeros(temporal.shape[1], dtype=np.float32) if use_tmp else None

    # normalisers: the coverage the whole candidate set would achieve
    sim_norm = float(sim.max(axis=0).sum()) if use_sim else 1.0
    tmp_norm = float(temporal.max(axis=0).sum()) if use_tmp else 1.0
    sim_norm = max(sim_norm, 1e-8)
    tmp_norm = max(tmp_norm, 1e-8)
    rel_scale = alpha / float(k)

    chosen: list[int] = []
    remaining = np.ones(n, dtype=bool)

    for _ in range(k):
        gains = rel_scale * relevance.astype(np.float32)

        if use_sim:
            delta = np.maximum(sim - cur_sim[None, :], 0.0)
            gains = gains + beta * delta.sum(axis=1) / sim_norm

        if use_tmp:
            delta_t = np.maximum(temporal - cur_tmp[None, :], 0.0)
            gains = gains + gamma * delta_t.sum(axis=1) / tmp_norm

        if not remaining.any():
            break
        gains[~remaining] = -np.inf
        best = int(np.argmax(gains))

        chosen.append(best)
        remaining[best] = False
        if use_sim:
            cur_sim = np.maximum(cur_sim, sim[best])
        if use_tmp:
            cur_tmp = np.maximum(cur_tmp, temporal[best])

    return np.array(sorted(chosen), dtype=int)


@SELECTORS.register("submodular")
class SubmodularSelector(Selector):
    needs_query = True

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        n_time_bins: int = 32,
        time_sigma_frac: float = 0.05,
        **_: object,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.n_time_bins = int(n_time_bins)
        self.time_sigma_frac = float(time_sigma_frac)

    def select(self, ctx: SelectionContext) -> np.ndarray:
        feats = ctx.frame_feats
        r = ctx.relevance()
        sim = (feats @ feats.T).astype(np.float32) if self.beta > 0 else None
        tmp = (
            temporal_kernel(ctx.times, self.n_time_bins, self.time_sigma_frac)
            if self.gamma > 0
            else None
        )
        return greedy_submodular(
            r, sim, tmp, ctx.budget, self.alpha, self.beta, self.gamma
        )


@SELECTORS.register("mmr")
class MMRSelector(Selector):
    """Maximal Marginal Relevance - the classic diversity heuristic.

    Included as the obvious competitor to our facility-location objective; it
    is greedy too but has no approximation guarantee, because
    ``max relevance - max similarity`` is not submodular.
    """

    needs_query = True

    def __init__(self, lam: float = 0.5, **_: object) -> None:
        self.lam = float(lam)

    def select(self, ctx: SelectionContext) -> np.ndarray:
        feats, r = ctx.frame_feats, ctx.relevance()
        n, k = ctx.n_frames, min(ctx.budget, ctx.n_frames)
        sim = (feats @ feats.T).astype(np.float32)

        chosen = [int(np.argmax(r))]
        while len(chosen) < k:
            worst = sim[:, chosen].max(axis=1)
            score = self.lam * r - (1 - self.lam) * worst
            score[chosen] = -np.inf
            chosen.append(int(np.argmax(score)))
        return np.array(sorted(chosen), dtype=int)
