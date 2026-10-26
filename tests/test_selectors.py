import numpy as np
import pytest

from afs.data.base import QASample
from afs.selectors import SELECTORS, SelectionContext, build_selector
from afs.selectors.adaptive import allocate_budgets, difficulty_score
from afs.selectors.submodular import greedy_submodular, temporal_kernel


def make_ctx(n=40, d=8, budget=4, seed=0, query=None, feats=None):
    rng = np.random.default_rng(seed)
    if feats is None:
        feats = rng.normal(size=(n, d)).astype(np.float32)
        feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    if query is None:
        query = feats[0].copy()
    sample = QASample(
        qid="q", video_id="v", video_path="", question="?",
        options=["a", "b"], answer_idx=0,
    )
    return SelectionContext(sample, feats, np.arange(len(feats), dtype=float), query, budget)


@pytest.mark.parametrize("name", ["uniform", "random", "stride", "topk", "mmr", "submodular"])
def test_respects_budget_and_bounds(name):
    ctx = make_ctx(budget=5)
    idx = build_selector({"name": name})(ctx)
    assert len(idx) == 5
    assert len(set(idx.tolist())) == 5
    assert idx.min() >= 0 and idx.max() < ctx.n_frames
    assert list(idx) == sorted(idx)


@pytest.mark.parametrize("name", ["uniform", "topk", "submodular", "mmr"])
def test_budget_larger_than_video(name):
    ctx = make_ctx(n=3, budget=10)
    idx = build_selector({"name": name})(ctx)
    assert len(idx) == 3


def test_uniform_covers_the_whole_video():
    ctx = make_ctx(n=100, budget=4)
    idx = build_selector({"name": "uniform"})(ctx)
    # spread across the timeline, and never the degenerate "first K frames"
    assert idx[0] < 25 and idx[-1] > 75


def test_submodular_beats_topk_on_redundant_frames():
    """The core claim: when the most relevant frames are near-duplicates,
    pure top-K wastes its budget and a diversity term does not."""
    rng = np.random.default_rng(0)
    d = 16
    query = rng.normal(size=d).astype(np.float32)
    query /= np.linalg.norm(query)

    feats = rng.normal(size=(60, d)).astype(np.float32) * 0.2
    # a tight cluster of 10 highly relevant, nearly identical frames
    feats[10:20] = query + rng.normal(size=(10, d)).astype(np.float32) * 0.01
    # one isolated, moderately relevant frame elsewhere in the video
    feats[45] = 0.6 * query + 0.4 * rng.normal(size=d).astype(np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)

    ctx = make_ctx(budget=4, query=query, feats=feats)

    top = build_selector({"name": "topk"})(ctx)
    sub = build_selector({"name": "submodular", "beta": 1.0, "gamma": 0.5})(ctx)

    # top-K collapses into the duplicate cluster; ours does not
    assert all(10 <= i < 20 for i in top)
    assert not all(10 <= i < 20 for i in sub)
    assert float(np.std(sub)) > float(np.std(top))


def test_greedy_coverage_is_nested_and_monotone():
    """With alpha=0 the objective is budget-independent, so greedy must produce
    nested sets whose coverage never decreases as K grows - the property the
    (1 - 1/e) guarantee rests on.

    (With alpha>0 the relevance term is rescaled by K, so the objective itself
    changes with the budget and nestedness is not expected.)
    """
    rng = np.random.default_rng(1)
    n = 30
    feats = rng.normal(size=(n, 8)).astype(np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    sim = np.maximum(feats @ feats.T, 0.0)  # same clipping greedy applies
    rel = rng.random(n).astype(np.float32)
    tmp = temporal_kernel(np.arange(n, dtype=float), 16, 0.05)

    def coverage(idx):
        if len(idx) == 0:
            return 0.0
        return float(sim[idx].max(axis=0).sum()) + float(tmp[idx].max(axis=0).sum())

    prev_set, prev_val = set(), 0.0
    for k in range(1, 10):
        idx = greedy_submodular(rel, sim, tmp, k, alpha=0.0, beta=1.0, gamma=1.0)
        assert len(idx) == k
        cur_set = set(idx.tolist())
        assert prev_set.issubset(cur_set)
        cur_val = coverage(idx)
        assert cur_val >= prev_val - 1e-6
        prev_set, prev_val = cur_set, cur_val


def test_relevance_only_reduces_to_topk():
    """beta = gamma = 0 collapses our objective onto the top-K baseline."""
    ctx = make_ctx(n=40, budget=5, seed=3)
    sub = build_selector({"name": "submodular", "beta": 0.0, "gamma": 0.0})(ctx)
    top = build_selector({"name": "topk"})(ctx)
    assert list(sub) == list(top)


def test_temporal_kernel_shape_and_range():
    t = np.linspace(0, 10, 25)
    k = temporal_kernel(t, 8, 0.05)
    assert k.shape == (25, 8)
    assert k.min() >= 0.0 and k.max() <= 1.0 + 1e-6


def test_oracle_needle_retrieves_the_evidence():
    ctx = make_ctx(n=50, budget=3)
    ctx.sample.meta["needle_frames"] = [31, 32]
    idx = build_selector({"name": "oracle_needle"})(ctx)
    assert {31, 32}.issubset(set(idx.tolist()))
    assert len(idx) == 3


def test_allocate_budgets_is_budget_neutral_and_monotone():
    rng = np.random.default_rng(0)
    d = rng.random(500)
    b = allocate_budgets(d, k_target=8, k_min=2, k_max=32)
    assert abs(b.mean() - 8) < 0.6           # hits the target average
    assert b.min() >= 2 and b.max() <= 32    # respects the bounds
    # harder questions never get fewer frames than easier ones
    order = np.argsort(d)
    assert np.all(np.diff(b[order]) >= 0)


def test_allocate_budgets_degenerate_cases():
    assert len(allocate_budgets([], 8)) == 0
    b = allocate_budgets([0.5] * 10, k_target=4, k_min=4, k_max=4)
    assert set(b.tolist()) == {4}


def test_difficulty_is_higher_for_flat_relevance():
    n, d = 40, 8
    query = np.zeros(d, dtype=np.float32)
    query[0] = 1.0

    peaked = np.full((n, d), 0.0, dtype=np.float32)
    peaked[:, 1] = 1.0
    peaked[20] = query
    flat = np.tile(query, (n, 1)).astype(np.float32)
    flat += np.linspace(0, 0.01, n)[:, None]
    flat /= np.linalg.norm(flat, axis=1, keepdims=True)

    d_peaked = difficulty_score(make_ctx(query=query, feats=peaked))
    d_flat = difficulty_score(make_ctx(query=query, feats=flat))
    assert d_flat > d_peaked


def test_all_selectors_registered():
    for name in ["uniform", "random", "stride", "topk", "mmr", "submodular",
                 "oracle_needle", "oracle_greedy"]:
        assert name in SELECTORS
