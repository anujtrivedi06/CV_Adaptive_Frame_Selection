"""NExT-QA loader.

Expected layout (download separately; nothing here is committed)::

    data/nextqa/
        videos/<video_id>.mp4
        val.csv          # official csv: video,frame_count,width,height,question,
                         #               answer,qid,type,a0..a4

Set ``split_file`` in the config to point at whichever csv you use.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from afs.data.base import QASample, VideoQADataset
from afs.registry import DATASETS


@DATASETS.register("nextqa")
class NextQADataset(VideoQADataset):
    name = "nextqa"

    def __init__(
        self,
        root: str | Path,
        split_file: str = "val.csv",
        video_dir: str = "videos",
        video_ext: str = ".mp4",
    ) -> None:
        import pandas as pd

        root = Path(root)
        csv_path = root / split_file
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found. Download NExT-QA and point "
                f"dataset.root at its folder (see README, Phase 1)."
            )
        df = pd.read_csv(csv_path)

        samples: List[QASample] = []
        for _, row in df.iterrows():
            options = [str(row[f"a{i}"]) for i in range(5) if f"a{i}" in row]
            video_id = str(row["video"])
            samples.append(
                QASample(
                    qid=f"{video_id}_{row['qid']}",
                    video_id=video_id,
                    video_path=str(root / video_dir / f"{video_id}{video_ext}"),
                    question=str(row["question"]).strip(),
                    options=options,
                    answer_idx=int(row["answer"]),
                    qtype=str(row.get("type", "unknown")),
                )
            )
        super().__init__(samples)
        self.root = root
