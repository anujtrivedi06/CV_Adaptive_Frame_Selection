"""Per-video adaptive budget - the differentiator.

Every published selector fixes K for the whole dataset. But questions are not
equally hard: some are answered by a single frame, others need a scan of the
whole video. We therefore predict a difficulty score per question from the
*relevance distribution alone* (free - no extra model, no reader call) and
spend 4 frames where 4 suffice and 32 where they do not, subject to

    mean_i K_i  ==  K_target

so total compute is unchanged and the comparison against a fixed-K baseline is
exactly fair. Any win is a pure reallocation win.

Difficulty signals, all computed from the cached features:
  * entropy  - a flat relevance profile means the evidence is diffuse
  * margin   - a weak top-1 peak means nothing clearly stands out
  * spread   - evidence scattered far apart in time needs wider coverage
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from afs.selectors.base import SelectionContext


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = (x - x.max()) / max(temperature, 1e-6)
    e = np.exp(z)
    return e / (e.sum() + 1e-12)


def difficulty_score(
    ctx: SelectionContext,
    temperature: float = 0.1,
    w_entropy: float = 0.5,
    w_margin: float = 0.3,
    w_spread: float = 0.2,
) -> float:
    """-> roughly [0, 1]; higher means 'give this question more frames'."""
    r = ctx.relevance()
    n = len(r)
    if n <= 1:
        return 0.0

    p = _softmax(r, temperature)
    entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(n))  # 0=peaked 1=flat

    order = np.sort(r)[::-1]
    top = float(order[0])
    rest = float(order[1:].mean()) if n > 1 else 0.0
    margin = np.clip(top - rest, 0.0, 1.0)

    # temporal spread of the probability mass, in units of the video length
    t = np.asarray(ctx.times, dtype=np.float64)
    span = max(float(t.max() - t.min()), 1e-6)
    mean_t = float((p * t).sum())
    spread = float(np.sqrt((p * (t - mean_t) ** 2).sum())) / span
    spread = float(np.clip(spread / 0.3, 0.0, 1.0))

    d = w_entropy * entropy + w_margin * (1.0 - margin) + w_spread * spread
    return float(np.clip(d / (w_entropy + w_margin + w_spread), 0.0, 1.0))


def allocate_budgets(
    difficulties: Sequence[float],
    k_target: float,
    k_min: int = 2,
    k_max: int = 32,
    n_frames: Sequence[int] | None = None,
) -> np.ndarray:
    """Map difficulties to integer budgets whose mean is ``k_target``.

    We use the monotone map ``K_i = clip(k_min + d_i * (k_max - k_min) + c)``
    and binary-search the scalar offset ``c`` so the realised mean matches the
    target. Monotone in difficulty, exactly budget-neutral, nothing learned.
    """
    d = np.asarray(difficulties, dtype=np.float64)
    if d.size == 0:
        return np.zeros((0,), dtype=int)
    k_min = int(k_min)
    k_max = int(k_max)
    base = k_min + d * (k_max - k_min)

    def realised(c: float) -> np.ndarray:
        return np.clip(np.round(base + c), k_min, k_max).astype(int)

    lo, hi = -float(k_max), float(k_max)
    budgets = realised(0.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        budgets = realised(mid)
        if budgets.mean() < k_target:
            lo = mid
        else:
            hi = mid

    if n_frames is not None:
        budgets = np.minimum(budgets, np.asarray(n_frames, dtype=int))
    return budgets
