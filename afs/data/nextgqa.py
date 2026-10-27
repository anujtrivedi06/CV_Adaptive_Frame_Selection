"""NExT-GQA loader - NExT-QA questions plus ground-truth temporal grounding.

Why this rather than plain NExT-QA
----------------------------------
NExT-GQA is the same multiple-choice questions, but each one also carries the
time spans that actually contain the answer. For a frame-selection project that
is worth a great deal:

  * the **oracle is exact**, not a greedy approximation driven by reader calls;
  * we can measure **evidence recall** - did the selector retrieve the right
    moment? - which isolates selection quality from the reader's reasoning;
  * selection can therefore be evaluated **without a GPU at all**, using the
    grounding-based mock reader, which matters on a laptop.

Layout (downloaded from the `shuzhig/nextgqa` mirror, ~6.6 GB)::

    <root>/annotations/val.csv            video_id,question,answer,qid,type,a0..a4
    <root>/annotations/gsub_val.json      {video_id: {duration, fps, location:{qid: [[s,e],...]}}}
    <root>/videos/<video_id>.mp4

Spans are kept in **seconds** here; `afs.eval` resolves them to frame indices
against the decoded time grid, so they stay correct at any sampling rate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from afs.data.base import QASample, VideoQADataset
from afs.registry import DATASETS

#: NExT-QA's question-type codes, spelled out for per-category tables.
QTYPE_NAMES = {
    "CW": "causal-why",
    "CH": "causal-how",
    "TN": "temporal-next",
    "TC": "temporal-current",
    "TP": "temporal-previous",
    "DC": "descriptive-count",
    "DL": "descriptive-location",
    "DO": "descriptive-other",
}


def _answer_index(raw: object, options: List[str]) -> int:
    """Resolve the csv's `answer` field to an index into `options`.

    The NExT-GQA csvs store the answer as its *text* ("unwrap it"), while some
    NExT-QA releases store an integer index. Accept either. Text is matched
    first, because an option could itself be the string "3".
    """
    text = str(raw).strip()
    if text in options:
        return options.index(text)

    lowered = [o.strip().lower() for o in options]
    if text.lower() in lowered:
        return lowered.index(text.lower())

    try:
        idx = int(text)
    except ValueError:
        raise ValueError(
            f"answer {raw!r} is neither an index nor one of {options}"
        ) from None
    if not 0 <= idx < len(options):
        raise ValueError(f"answer index {idx} out of range for {len(options)} options")
    return idx


@DATASETS.register("nextgqa")
class NextGQADataset(VideoQADataset):
    name = "nextgqa"

    def __init__(
        self,
        root: str | Path,
        split: str = "val",
        video_dir: str = "videos",
        video_ext: str = ".mp4",
        annotations_dir: str = "annotations",
        evidence_mode: str = "any",
        require_grounding: bool = True,
        require_video: bool = True,
        expand_names: bool = True,
    ) -> None:
        import pandas as pd

        root = Path(root)
        ann = root / annotations_dir
        csv_path = ann / f"{split}.csv"
        gsub_path = ann / f"gsub_{split}.json"
        for p in (csv_path, gsub_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found. See README, 'Getting NExT-GQA'."
                )

        df = pd.read_csv(csv_path)
        gsub = json.loads(gsub_path.read_text(encoding="utf-8"))
        vdir = root / video_dir

        samples: List[QASample] = []
        n_no_ground = n_no_video = 0

        for _, row in df.iterrows():
            video_id = str(row["video_id"])
            qid = str(row["qid"])

            video_path = vdir / f"{video_id}{video_ext}"
            if require_video and not video_path.exists():
                n_no_video += 1
                continue

            entry = gsub.get(video_id, {})
            spans = (entry.get("location") or {}).get(qid)
            if not spans:
                n_no_ground += 1
                if require_grounding:
                    continue

            qtype = str(row.get("type", "unknown"))
            options = [str(row[f"a{i}"]) for i in range(5)]
            samples.append(
                QASample(
                    qid=f"{video_id}_{qid}",
                    video_id=video_id,
                    video_path=str(video_path),
                    question=str(row["question"]).strip(),
                    options=options,
                    answer_idx=_answer_index(row["answer"], options),
                    qtype=QTYPE_NAMES.get(qtype, qtype) if expand_names else qtype,
                    duration=float(entry.get("duration")) if entry.get("duration") else None,
                    meta={
                        "evidence_spans": [[float(s), float(e)] for s, e in spans]
                        if spans
                        else [],
                        "evidence_mode": evidence_mode,
                        "source_fps": entry.get("fps"),
                        "split": split,
                    },
                )
            )

        if not samples:
            raise RuntimeError(
                f"no usable samples in {csv_path}: "
                f"{n_no_video} missing videos, {n_no_ground} without grounding. "
                f"Has the video download finished?"
            )

        super().__init__(samples)
        self.root = root
        self.name = f"nextgqa-{split}"
        self.n_skipped_no_video = n_no_video
        self.n_skipped_no_grounding = n_no_ground
