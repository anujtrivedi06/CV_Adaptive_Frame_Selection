"""The evaluation runner: dataset -> features -> selector -> frozen reader.

One run = one (selector, budget) pair over a fixed question subset. Everything
else is held constant, which is what makes the Pareto curves comparable.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from afs.config import Config
from afs.data import build_dataset
from afs.data.base import QASample, VideoQADataset, covers_evidence
from afs.decode import decode_video
from afs.encoders import build_encoder
from afs.features import FeatureStore
from afs.metrics import SampleResult, append_to_leaderboard, save_results, summarize
from afs.readers import build_reader
from afs.readers.base import Reader
from afs.selectors import SelectionContext, build_selector
from afs.selectors.adaptive import allocate_budgets, difficulty_score


def encoder_tag(cfg: Config) -> str:
    name = cfg.encoder["name"]
    model = str(cfg.encoder.get("model_name", "")).split("/")[-1]
    return f"{name}-{model}" if model else name


def dataset_tag(cfg: Config, dataset: VideoQADataset) -> str:
    """Cache key for a dataset.

    The bare name is not enough: two configs can both be called "toy" while
    pointing at different videos (different root, different mode), and reusing
    one's cached features for the other silently produces wrong results. So we
    fold the whole dataset config - minus `limit`, which only subsets questions
    and does not change any video's features - into the key.
    """
    import hashlib
    import json

    payload = {k: v for k, v in cfg.dataset.items() if k != "limit"}
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]
    base = dataset.name.split("-sub")[0]  # subsetting must not split the cache
    return f"{base}-{digest}"


def build_store(cfg: Config, dataset: VideoQADataset) -> FeatureStore:
    encoder = build_encoder(cfg.encoder)
    return FeatureStore(
        cache_dir=cfg.paths["cache_dir"],
        dataset_name=dataset_tag(cfg, dataset),
        encoder=encoder,
        encoder_tag=encoder_tag(cfg),
        fps=cfg.frames["fps"],
        max_frames=cfg.frames["max_candidates"],
        frame_size=tuple(cfg.frames["size"]),
    )


def build_everything(cfg: Config):
    dataset = build_dataset(cfg.dataset, seed=cfg.get("seed", 0))
    store = build_store(cfg, dataset)
    selector = build_selector(cfg.selector)
    reader = build_reader(cfg.reader, store=store)
    return dataset, store, selector, reader


def _load_pixels(cfg: Config, sample: QASample, idx: np.ndarray) -> np.ndarray:
    frames, _ = decode_video(
        sample.video_path,
        fps=cfg.frames["fps"],
        max_frames=cfg.frames["max_candidates"],
        size=tuple(cfg.frames["size"]),
    )
    idx = np.clip(idx, 0, max(len(frames) - 1, 0))
    return frames[idx]


def build_features(cfg: Config, overwrite: bool = False) -> int:
    """Phase 1: decode + encode every video once. Nothing else runs until this
    has been done."""
    dataset = build_dataset(cfg.dataset, seed=cfg.get("seed", 0))
    store = build_store(cfg, dataset)
    seen: set[str] = set()
    built = 0
    for sample in tqdm(list(dataset), desc="caching features"):
        if sample.video_id in seen:
            continue
        seen.add(sample.video_id)
        if store.has_video(sample.video_id) and not overwrite:
            continue
        store.build_video(sample, overwrite=overwrite)
        built += 1
    return built


def run_eval(
    cfg: Config,
    budget: Optional[int] = None,
    run_name: Optional[str] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    dataset, store, selector, reader = build_everything(cfg)

    k_target = float(budget if budget is not None else cfg.budget["K"])
    adaptive = bool(cfg.budget.get("adaptive", False))
    enc_gflops = float(cfg.cost.get("encoder_gflops_per_frame", 0.0))
    if "gflops_per_frame" in cfg.reader:
        reader.gflops_per_frame = float(cfg.reader["gflops_per_frame"])

    samples = list(dataset)
    contexts: List[SelectionContext] = []
    difficulties: List[float] = []

    # ---- pass 1: features, query vectors, (optionally) difficulty ----
    needs_query = getattr(selector, "needs_query", False) or adaptive
    for sample in tqdm(samples, desc="features", disable=not progress):
        feats, times = store.get_video(sample)
        query = (
            store.query_vector(sample, cfg.selector.get("use_options", True))
            if needs_query
            else None
        )
        ctx = SelectionContext(sample, feats, times, query, int(round(k_target)))
        contexts.append(ctx)
        difficulties.append(difficulty_score(ctx) if adaptive else float("nan"))

    # ---- budget allocation ----
    if adaptive:
        budgets = allocate_budgets(
            difficulties,
            k_target=k_target,
            k_min=int(cfg.budget.get("K_min", 2)),
            k_max=int(cfg.budget.get("K_max", 32)),
            n_frames=[c.n_frames for c in contexts],
        )
    else:
        budgets = np.full(len(contexts), int(round(k_target)), dtype=int)

    # ---- pass 2: select + read ----
    results: List[SampleResult] = []
    needs_pixels = getattr(reader, "needs_pixels", False)

    for ctx, k, diff in tqdm(
        list(zip(contexts, budgets, difficulties)),
        desc="eval",
        disable=not progress,
    ):
        sample = ctx.sample
        ctx.budget = int(k)
        if ctx.score_fn is None and selector.registry_name == "oracle_greedy":
            ctx.score_fn = _make_score_fn(cfg, reader, sample, needs_pixels)

        t0 = time.perf_counter()
        idx = selector(ctx)
        select_s = time.perf_counter() - t0

        pixels = _load_pixels(cfg, sample, idx) if needs_pixels else np.zeros(0)
        out = reader.answer(sample, pixels, idx)

        results.append(
            SampleResult(
                qid=sample.qid,
                video_id=sample.video_id,
                qtype=sample.qtype,
                correct=bool(out.pred_idx == sample.answer_idx),
                pred_idx=int(out.pred_idx),
                answer_idx=int(sample.answer_idx),
                budget=int(k),
                n_selected=int(len(idx)),
                n_candidates=int(ctx.n_frames),
                selected=[int(i) for i in idx],
                confidence=out.confidence,
                difficulty=float(diff),
                select_s=select_s,
                read_s=out.latency_s,
                reader_gflops=reader.cost_gflops(len(idx)),
                selector_gflops=enc_gflops * ctx.n_frames,
                hit_evidence=covers_evidence(sample, idx),
            )
        )

    run_name = run_name or f"{cfg.selector['name']}_K{int(round(k_target))}"
    summary = summarize(
        results,
        run_name=run_name,
        selector=cfg.selector["name"],
        reader=cfg.reader["name"],
        encoder=encoder_tag(cfg),
        dataset=dataset.name,
        budget=k_target,
        extra={"adaptive": adaptive},
    )

    out_dir = Path(cfg.paths["runs_dir"]) / run_name
    save_results(out_dir, summary, results, cfg.to_plain())
    append_to_leaderboard(Path(cfg.paths["runs_dir"]) / "leaderboard.jsonl", summary)
    return {"summary": summary, "results": results, "out_dir": out_dir}


def _make_score_fn(cfg: Config, reader: Reader, sample: QASample, needs_pixels: bool):
    def score(idx: np.ndarray) -> float:
        pixels = _load_pixels(cfg, sample, idx) if needs_pixels else np.zeros(0)
        return reader.score_correct(sample, pixels, idx)

    return score
