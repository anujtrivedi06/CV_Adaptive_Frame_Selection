"""Dataset abstractions shared by every benchmark we plug in."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence


@dataclass
class QASample:
    """One multiple-choice question about one video."""

    qid: str
    video_id: str
    video_path: str
    question: str
    options: List[str]
    answer_idx: int
    qtype: str = "unknown"
    duration: Optional[float] = None
    # Free-form extras. The toy dataset puts the ground-truth evidence window
    # here under "needle_frames", which lets us score a selector without ever
    # running a VLM.
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def answer_text(self) -> str:
        return self.options[self.answer_idx]

    def prompt(self) -> str:
        letters = "ABCDEFGH"
        lines = [self.question]
        for i, opt in enumerate(self.options):
            lines.append(f"{letters[i]}. {opt}")
        return "\n".join(lines)


def evidence_groups(sample: QASample) -> List[List[int]]:
    """Ground-truth evidence windows, when the dataset knows them.

    A question may need evidence from several *separate* moments. Returning a
    list of groups (rather than one flat frame list) is what lets us say
    "retrieved the evidence" only when every required moment was covered.
    """
    groups = sample.meta.get("evidence_groups")
    if groups:
        return [list(g) for g in groups]
    flat = sample.meta.get("needle_frames")
    return [list(flat)] if flat else []


def covers_evidence(
    sample: QASample, idx: Sequence[int], mode: Optional[str] = None
) -> Optional[bool]:
    """Did the selection retrieve the ground-truth evidence?

    ``mode="all"``  every annotated window must be hit. Right when the windows
                    are jointly necessary - e.g. our two-event toy, where you
                    cannot order two events without seeing both.
    ``mode="any"``  one hit is enough. Right for NExT-GQA, whose multiple
                    intervals are alternative groundings of the same evidence,
                    which is also what its IoP-style metric assumes.

    Defaults to the dataset's own ``meta["evidence_mode"]``, else "all".
    Returns None when the dataset has no evidence annotation.
    """
    groups = evidence_groups(sample)
    if not groups:
        return None
    mode = mode or sample.meta.get("evidence_mode", "all")
    chosen = {int(i) for i in idx}
    hits = (bool(set(g) & chosen) for g in groups)
    return all(hits) if mode == "all" else any(hits)


def spans_to_groups(
    spans: Sequence[Sequence[float]], times: Sequence[float]
) -> List[List[int]]:
    """Convert second-intervals to indices into a decoded frame grid.

    Datasets annotate evidence in seconds, but a selector works on indices into
    whatever grid we decoded at (fps, max_candidates). Resolving the two here -
    against the `times` array we cached alongside the features - keeps the
    annotation independent of the sampling rate.

    A span shorter than the frame spacing can contain no sampled frame; rather
    than drop it we snap it to the single nearest frame, so the evidence is
    never silently lost.
    """
    import numpy as np

    t = np.asarray(times, dtype=np.float64)
    out: List[List[int]] = []
    if t.size == 0:
        return out
    for span in spans:
        start, end = float(span[0]), float(span[1])
        inside = np.nonzero((t >= start) & (t <= end))[0]
        if inside.size == 0:
            inside = np.array([int(np.argmin(np.abs(t - (start + end) / 2.0)))])
        out.append([int(i) for i in inside])
    return out


class VideoQADataset:
    """Minimal interface every dataset loader implements."""

    name: str = "base"

    def __init__(self, samples: Sequence[QASample]) -> None:
        self._samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[QASample]:
        return iter(self._samples)

    def __getitem__(self, idx: int) -> QASample:
        return self._samples[idx]

    @property
    def samples(self) -> List[QASample]:
        return self._samples

    def video_ids(self) -> List[str]:
        seen, out = set(), []
        for s in self._samples:
            if s.video_id not in seen:
                seen.add(s.video_id)
                out.append(s.video_id)
        return out

    def subset(self, n: int, seed: int = 0, offset: int = 0) -> "VideoQADataset":
        """Deterministic subsample - we fix question subsets early and keep them.

        ``offset`` carves out *disjoint* subsets from the same shuffle, which is
        how we separate a tuning split from a reporting split. Choosing
        hyperparameters on the questions you then report on is test-set tuning
        and inflates every number; with an offset, `limit=250, offset=0` and
        `limit=250, offset=250` share no question.
        """
        import random

        rng = random.Random(seed)
        picked = sorted(self._samples, key=lambda s: s.qid)
        rng.shuffle(picked)
        clone = VideoQADataset(picked[offset : offset + n])
        clone.name = f"{self.name}-sub{n}@{offset}" if offset else f"{self.name}-sub{n}"
        return clone
