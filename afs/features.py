"""Feature cache.

The single most important engineering decision in the project: decode + encode
each video exactly once, write the features to disk, and let every selector
experiment afterwards run in seconds instead of hours.

Layout::

    cache/<dataset>/<encoder_tag>/fps<fps>/video/<video_id>.npz   feats, times
    cache/<dataset>/<encoder_tag>/text/<sha1>.npy                 text feats
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from afs.data.base import QASample
from afs.decode import decode_video
from afs.encoders.base import Encoder


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class FeatureStore:
    def __init__(
        self,
        cache_dir: str | Path,
        dataset_name: str,
        encoder: Encoder,
        encoder_tag: str,
        fps: float = 1.0,
        max_frames: int | None = 256,
        frame_size: Tuple[int, int] = (224, 224),
    ) -> None:
        self.encoder = encoder
        self.fps = fps
        self.max_frames = max_frames
        self.frame_size = tuple(frame_size)
        root = Path(cache_dir) / dataset_name / encoder_tag
        self.video_dir = root / f"fps{fps:g}" / "video"
        self.text_dir = root / "text"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
        self._text_mem: dict[str, np.ndarray] = {}

    # ---------------- video ----------------

    def video_path_for(self, video_id: str) -> Path:
        return self.video_dir / f"{video_id}.npz"

    def has_video(self, video_id: str) -> bool:
        return self.video_path_for(video_id).exists()

    def build_video(self, sample: QASample, overwrite: bool = False) -> Path:
        out = self.video_path_for(sample.video_id)
        if out.exists() and not overwrite:
            return out
        frames, times = decode_video(
            sample.video_path,
            fps=self.fps,
            max_frames=self.max_frames,
            size=self.frame_size,
        )
        feats = self.encoder.encode_images_batched(frames)
        # note: np.savez_compressed appends ".npz" unless the name already ends
        # in it, so the temp name must keep that suffix or the rename misses.
        tmp = out.with_name(out.name + ".tmp.npz")
        np.savez_compressed(tmp, feats=feats.astype(np.float32), times=times)
        tmp.replace(out)
        return out

    def get_video(self, sample: QASample) -> Tuple[np.ndarray, np.ndarray]:
        path = self.video_path_for(sample.video_id)
        if not path.exists():
            self.build_video(sample)
        with np.load(path) as z:
            return z["feats"].astype(np.float32), z["times"].astype(np.float64)

    # ---------------- text ----------------

    def get_text(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.encoder.dim), dtype=np.float32)
        missing_idx, missing_txt = [], []
        for i, t in enumerate(texts):
            key = _sha1(t)
            if key in self._text_mem:
                out[i] = self._text_mem[key]
                continue
            p = self.text_dir / f"{key}.npy"
            if p.exists():
                v = np.load(p)
                self._text_mem[key] = v
                out[i] = v
            else:
                missing_idx.append(i)
                missing_txt.append(t)
        if missing_txt:
            feats = self.encoder.encode_text(missing_txt)
            for j, i in enumerate(missing_idx):
                key = _sha1(texts[i])
                np.save(self.text_dir / f"{key}.npy", feats[j])
                self._text_mem[key] = feats[j]
                out[i] = feats[j]
        return out

    def query_vector(self, sample: QASample, include_options: bool = True) -> np.ndarray:
        """One unit-norm vector representing the question (+ its options)."""
        texts = [sample.question]
        if include_options:
            texts += [f"{sample.question} {opt}" for opt in sample.options]
        feats = self.get_text(texts)
        v = feats.mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-8)
