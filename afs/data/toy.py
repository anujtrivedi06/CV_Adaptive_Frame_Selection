"""Synthetic 'needle in a haystack' video-QA dataset.

Why this exists
---------------
It is Phase 5's needle probe, but we build it in Phase 1 because it makes the
whole repository runnable on a laptop with no GPU, no downloads and no VLM:

  * every video is a long stretch of distractor frames with a coloured square
    visible in exactly one short window (the "needle");
  * the question asks for the colour of that square;
  * we store the needle frame indices, so the *exact* oracle is known and a
    mock reader can be scored analytically.

That gives a meaningful accuracy-vs-K curve before any real model is involved,
and it is the cleanest possible unit test for a frame selector: a selector that
cannot find a literal needle will not find a subtle one.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

from afs.data.base import QASample, VideoQADataset
from afs.registry import DATASETS

# BGR, because OpenCV writes BGR.
COLORS: dict[str, Tuple[int, int, int]] = {
    "red": (40, 40, 220),
    "green": (60, 190, 70),
    "blue": (220, 70, 50),
    "yellow": (60, 215, 225),
}
COLOR_NAMES = list(COLORS)


def _distractor_frame(rng: np.random.Generator, h: int, w: int, t: int) -> np.ndarray:
    """Grey noise plus a drifting white blob, so frames are not trivially equal."""
    frame = rng.integers(70, 110, size=(h, w, 3), dtype=np.uint8)
    cx = int((0.5 + 0.4 * np.sin(t * 0.17)) * w)
    cy = int((0.5 + 0.4 * np.cos(t * 0.11)) * h)
    r = max(4, h // 12)
    ys, xs = np.ogrid[:h, :w]
    mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= r * r
    frame[mask] = 200
    return frame


def _paint_needle(frame: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    size = max(8, h // 4)
    y0, x0 = (h - size) // 2, (w - size) // 2
    out[y0 : y0 + size, x0 : x0 + size] = np.array(color, dtype=np.uint8)
    return out


def build_toy_dataset(
    root: str | Path,
    n_videos: int = 24,
    n_frames: int = 120,
    needle_len: int = 4,
    size: Tuple[int, int] = (96, 128),
    fps: int = 1,
    seed: int = 0,
    overwrite: bool = False,
    mode: str = "single",
) -> Path:
    """Write videos + a manifest under ``root``. Returns the manifest path.

    ``mode="single"``
        One coloured square in one window; "what colour is it?". Any selector
        that can score relevance solves this at K=1, so it tests the harness,
        not the method.

    ``mode="order"``
        *Two* squares of different colours, in two well-separated windows;
        "which colour appears first?". Answering needs evidence from **both**
        windows, so a budget spent on near-duplicate frames from one window is
        a wasted budget. This is the setting where relevance-only top-K is
        supposed to fail and a diversity/coverage term is supposed to win -
        i.e. the one that actually tests our claim.
    """
    import cv2

    if mode not in ("single", "order"):
        raise ValueError(f"unknown toy mode {mode!r}")
    root = Path(root)
    (root / "videos").mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not overwrite:
        return manifest_path

    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    h, w = size
    records = []

    for vi in range(n_videos):
        video_id = f"toy{vi:03d}"

        if mode == "single":
            color_name = COLOR_NAMES[vi % len(COLOR_NAMES)]
            # keep the needle away from both ends, otherwise uniform sampling
            # that always takes frame 0 would get it for free
            start = int(pyrng.uniform(0.1, 0.85) * (n_frames - needle_len))
            groups = [list(range(start, start + needle_len))]
            paint = {t: COLORS[color_name] for t in groups[0]}
            question = "What is the colour of the square that appears in the video?"
            answer_name = color_name
            qtype = "needle-colour"
        else:
            first, second = pyrng.sample(COLOR_NAMES, 2)
            s1 = int(pyrng.uniform(0.08, 0.35) * (n_frames - needle_len))
            s2 = int(pyrng.uniform(0.60, 0.90) * (n_frames - needle_len))
            g1 = list(range(s1, s1 + needle_len))
            g2 = list(range(s2, s2 + needle_len))
            groups = [g1, g2]
            paint = {t: COLORS[first] for t in g1}
            paint.update({t: COLORS[second] for t in g2})
            question = (
                "Two coloured squares appear at different times. "
                "Which colour appears first?"
            )
            answer_name = first
            qtype = "two-event-order"

        needle_frames = sorted(t for g in groups for t in g)

        path = root / "videos" / f"{video_id}.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        if not writer.isOpened():  # pragma: no cover - codec missing
            raise RuntimeError(f"cv2 could not open a writer for {path}")
        for t in range(n_frames):
            frame = _distractor_frame(rng, h, w, t)
            if t in paint:
                frame = _paint_needle(frame, paint[t])
            writer.write(frame)
        writer.release()

        options = list(COLOR_NAMES)
        pyrng.shuffle(options)
        records.append(
            {
                "qid": f"{video_id}_q0",
                "video_id": video_id,
                "video_path": str(path.relative_to(root)).replace("\\", "/"),
                "question": question,
                "options": options,
                "answer_idx": options.index(answer_name),
                "qtype": qtype,
                "duration": n_frames / fps,
                "meta": {
                    "needle_frames": needle_frames,
                    "evidence_groups": groups,
                    "n_frames": n_frames,
                    "fps": fps,
                },
            }
        )

    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return manifest_path


@DATASETS.register("toy")
class ToyNeedleDataset(VideoQADataset):
    name = "toy"

    def __init__(self, root: str | Path, build: bool = True, **build_kwargs) -> None:
        root = Path(root)
        manifest = root / "manifest.json"
        if not manifest.exists():
            if not build:
                raise FileNotFoundError(
                    f"{manifest} missing; run scripts/make_toy_dataset.py"
                )
            manifest = build_toy_dataset(root, **build_kwargs)
        records = json.loads(manifest.read_text(encoding="utf-8"))
        samples: List[QASample] = [
            QASample(
                qid=r["qid"],
                video_id=r["video_id"],
                video_path=str(root / r["video_path"]),
                question=r["question"],
                options=r["options"],
                answer_idx=r["answer_idx"],
                qtype=r["qtype"],
                duration=r.get("duration"),
                meta=r.get("meta", {}),
            )
            for r in records
        ]
        super().__init__(samples)
        self.root = root
