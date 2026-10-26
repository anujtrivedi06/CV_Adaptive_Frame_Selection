"""End-to-end smoke test: toy videos -> features -> selection -> mock reader.

Runs on CPU in a few seconds, so CI (and a groupmate's laptop) can verify the
whole pipeline without a dataset download or a GPU.
"""

import numpy as np
import pytest

from afs.config import Config, apply_override, load_config
from afs.data.toy import build_toy_dataset
from afs.decode import decode_video
from afs.eval import build_features, run_eval


def toy_cfg(tmp_path, **over) -> Config:
    cfg = load_config("configs/toy.yaml")
    apply_override(cfg, f"paths.cache_dir={tmp_path / 'cache'}")
    apply_override(cfg, f"paths.runs_dir={tmp_path / 'runs'}")
    apply_override(cfg, f"dataset.root={tmp_path / 'toy'}")
    apply_override(cfg, "dataset.n_videos=8")
    apply_override(cfg, "dataset.n_frames=60")
    for k, v in over.items():
        apply_override(cfg, f"{k}={v}")
    return cfg


def test_config_overrides_and_types():
    cfg = load_config("configs/toy.yaml", ["budget.K=16", "budget.adaptive=true"])
    assert cfg.budget.K == 16 and cfg.budget.adaptive is True
    assert isinstance(cfg.frames["size"], list)


def test_toy_videos_decode_with_the_needle_present(tmp_path):
    root = tmp_path / "toy"
    build_toy_dataset(root, n_videos=2, n_frames=40, needle_len=3, size=(64, 64))
    frames, times = decode_video(root / "videos" / "toy000.mp4", fps=1.0, size=(64, 64))
    assert len(frames) >= 35
    assert frames.dtype == np.uint8 and frames.shape[-1] == 3
    assert len(times) == len(frames)


def test_feature_cache_is_written_once_and_reused(tmp_path):
    cfg = toy_cfg(tmp_path)
    n = build_features(cfg)
    assert n == 8
    assert build_features(cfg) == 0  # second call is a no-op


@pytest.mark.parametrize("selector", ["uniform", "topk", "submodular"])
def test_run_eval_produces_a_summary(tmp_path, selector):
    cfg = toy_cfg(tmp_path, **{"selector.name": selector})
    out = run_eval(cfg, budget=4, progress=False)
    s = out["summary"]
    assert s.n == 8
    assert 0.0 <= s.accuracy <= 1.0
    assert s.mean_frames == 4
    assert s.mean_total_gflops > 0
    assert (out["out_dir"] / "summary.json").exists()
    assert (out["out_dir"] / "per_sample.jsonl").exists()


def test_oracle_is_an_upper_bound_on_the_toy_task(tmp_path):
    """The whole point of the oracle: nothing should beat it."""
    accs = {}
    for name in ["uniform", "random", "topk", "submodular", "oracle_needle"]:
        cfg = toy_cfg(tmp_path, **{"selector.name": name})
        accs[name] = run_eval(cfg, budget=4, progress=False)["summary"].accuracy
    assert accs["oracle_needle"] == 1.0
    for name, acc in accs.items():
        assert acc <= accs["oracle_needle"] + 1e-9, name


def test_adaptive_budget_keeps_average_cost(tmp_path):
    cfg = toy_cfg(tmp_path, **{"budget.adaptive": "true", "budget.K_min": 2,
                               "budget.K_max": 16})
    out = run_eval(cfg, budget=8, progress=False)
    frames = np.array([r.n_selected for r in out["results"]])
    assert abs(frames.mean() - 8) <= 1.0   # same average compute as fixed K=8
    assert frames.std() > 0                # but genuinely spent unevenly
