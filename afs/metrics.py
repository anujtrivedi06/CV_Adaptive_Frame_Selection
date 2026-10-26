"""Metrics and result aggregation.

We never report a single accuracy number on its own. Every run records the
frames actually used and the compute actually spent, because the deliverable is
a Pareto curve of accuracy against cost, not a leaderboard row.

Honest accounting: a selector that encodes all N frames with a cheap encoder in
order to pick K for an expensive reader must pay for those N cheap encodes. We
charge it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SampleResult:
    qid: str
    video_id: str
    qtype: str
    correct: bool
    pred_idx: int
    answer_idx: int
    budget: int
    n_selected: int
    n_candidates: int
    selected: List[int]
    confidence: float = float("nan")
    difficulty: float = float("nan")
    select_s: float = 0.0
    read_s: float = 0.0
    reader_gflops: float = 0.0
    selector_gflops: float = 0.0
    hit_evidence: Optional[bool] = None  # toy only: did we retrieve the needle?

    @property
    def total_gflops(self) -> float:
        return self.reader_gflops + self.selector_gflops


@dataclass
class RunSummary:
    run_name: str
    selector: str
    reader: str
    encoder: str
    dataset: str
    budget: float
    accuracy: float
    n: int
    mean_frames: float
    mean_reader_gflops: float
    mean_selector_gflops: float
    mean_total_gflops: float
    mean_select_s: float
    mean_read_s: float
    evidence_recall: Optional[float] = None
    per_type: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        per_type = d.pop("per_type")
        d.update({f"acc[{k}]": v for k, v in per_type.items()})
        d.pop("extra")
        return d


def summarize(
    results: List[SampleResult],
    run_name: str,
    selector: str,
    reader: str,
    encoder: str,
    dataset: str,
    budget: float,
    extra: Optional[Dict[str, Any]] = None,
) -> RunSummary:
    n = len(results)
    if n == 0:
        raise ValueError("no results to summarize")

    correct = np.array([r.correct for r in results], dtype=bool)
    per_type: Dict[str, float] = {}
    for qt in sorted({r.qtype for r in results}):
        mask = np.array([r.qtype == qt for r in results], dtype=bool)
        per_type[qt] = float(correct[mask].mean())

    hits = [r.hit_evidence for r in results if r.hit_evidence is not None]
    return RunSummary(
        run_name=run_name,
        selector=selector,
        reader=reader,
        encoder=encoder,
        dataset=dataset,
        budget=budget,
        accuracy=float(correct.mean()),
        n=n,
        mean_frames=float(np.mean([r.n_selected for r in results])),
        mean_reader_gflops=float(np.mean([r.reader_gflops for r in results])),
        mean_selector_gflops=float(np.mean([r.selector_gflops for r in results])),
        mean_total_gflops=float(np.mean([r.total_gflops for r in results])),
        mean_select_s=float(np.mean([r.select_s for r in results])),
        mean_read_s=float(np.mean([r.read_s for r in results])),
        evidence_recall=float(np.mean(hits)) if hits else None,
        per_type=per_type,
        extra=extra or {},
    )


def save_results(
    out_dir: str | Path,
    summary: RunSummary,
    results: List[SampleResult],
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2, default=str), encoding="utf-8"
    )
    with (out_dir / "per_sample.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r), default=str) + "\n")
    if config is not None:
        (out_dir / "config.json").write_text(
            json.dumps(config, indent=2, default=str), encoding="utf-8"
        )
    return out_dir


def append_to_leaderboard(path: str | Path, summary: RunSummary) -> None:
    """One row per run, so plot_pareto.py can read every experiment at once."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary.to_row(), default=str) + "\n")
