"""Frame/text encoders.

Contract: images and text land in the *same* unit-norm space, so relevance is
a dot product and diversity is a dot product. Every selector in the repo only
ever sees those two matrices, never pixels.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


class Encoder:
    dim: int = 0
    batch_size: int = 64

    def encode_images(self, frames: np.ndarray) -> np.ndarray:
        """uint8 ``[N, H, W, 3]`` RGB -> float32 ``[N, dim]``, unit norm."""
        raise NotImplementedError

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """-> float32 ``[len(texts), dim]``, unit norm."""
        raise NotImplementedError

    def encode_images_batched(self, frames: np.ndarray) -> np.ndarray:
        if len(frames) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: List[np.ndarray] = []
        for i in range(0, len(frames), self.batch_size):
            out.append(self.encode_images(frames[i : i + self.batch_size]))
        return np.concatenate(out, axis=0)
