"""Video -> candidate frames, at a fixed sampling rate.

This is the only place that touches a video container. Everything downstream
works on the decoded array, so swapping PyAV for decord later changes one file.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def decode_video(
    path: str | Path,
    fps: float = 1.0,
    max_frames: int | None = None,
    size: Tuple[int, int] | None = (224, 224),
    backend: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode ``path`` at ``fps``.

    Returns ``(frames, times)`` where frames is uint8 ``[N, H, W, 3]`` in RGB
    and times is float seconds ``[N]``.
    """
    path = str(path)
    if backend == "auto":
        backend = "pyav"
    try:
        if backend == "pyav":
            frames, times = _decode_pyav(path, fps, max_frames)
        else:
            frames, times = _decode_cv2(path, fps, max_frames)
    except Exception:
        frames, times = _decode_cv2(path, fps, max_frames)

    if size is not None and len(frames):
        frames = _resize(frames, size)
    return frames, times


def _resize(frames: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    import cv2

    h, w = size
    out = np.empty((len(frames), h, w, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        out[i] = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
    return out


def _wanted_times(duration: float, fps: float, max_frames: int | None) -> np.ndarray:
    n = max(1, int(round(duration * fps)))
    times = np.arange(n, dtype=np.float64) / fps
    if max_frames and n > max_frames:
        # keep the full temporal span, just thin it out
        idx = np.linspace(0, n - 1, max_frames).round().astype(int)
        times = times[idx]
    return times


def _decode_pyav(path: str, fps: float, max_frames: int | None):
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        avg = float(stream.average_rate or 30.0)
        total = stream.frames or 0
        duration = (total / avg) if total else float(container.duration or 0) / 1e6
        if duration <= 0:
            duration = 1e9  # unknown: fall back to "take everything, then stride"

        step = max(1, int(round(avg / fps))) if fps > 0 else 1
        frames: List[np.ndarray] = []
        times: List[float] = []
        for i, frame in enumerate(container.decode(video=0)):
            if i % step:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            times.append(float(i) / avg)
            if max_frames and len(frames) > max_frames * 4:
                break

    arr = np.stack(frames) if frames else np.zeros((0, 1, 1, 3), np.uint8)
    t = np.asarray(times, dtype=np.float64)
    if max_frames and len(arr) > max_frames:
        idx = np.linspace(0, len(arr) - 1, max_frames).round().astype(int)
        arr, t = arr[idx], t[idx]
    return arr, t


def _decode_cv2(path: str, fps: float, max_frames: int | None):
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps))) if fps > 0 else 1

    frames: List[np.ndarray] = []
    times: List[float] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            times.append(i / src_fps)
        i += 1
    cap.release()

    arr = np.stack(frames) if frames else np.zeros((0, 1, 1, 3), np.uint8)
    t = np.asarray(times, dtype=np.float64)
    if max_frames and len(arr) > max_frames:
        idx = np.linspace(0, len(arr) - 1, max_frames).round().astype(int)
        arr, t = arr[idx], t[idx]
    return arr, t
