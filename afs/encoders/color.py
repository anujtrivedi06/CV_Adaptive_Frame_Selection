"""A deliberately cheap CPU encoder: colour-palette occupancy.

It is not a substitute for CLIP on real data, but it is a *real* encoder (not a
random projection), it needs no downloads, and on the toy needle dataset it is
genuinely informative. That means the full pipeline - decode, cache, score,
select, read, plot - can be exercised on a laptop before anyone touches a GPU.

Image embedding: fraction of pixels falling near each palette colour, plus a
final "none of the above" dimension.
Text embedding:  one-hot over the palette colour words present in the string.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from afs.encoders.base import Encoder, l2_normalize
from afs.registry import ENCODERS

# RGB palette (the toy dataset's colours plus common ones)
PALETTE: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "green": (70, 190, 60),
    "blue": (50, 70, 220),
    "yellow": (225, 215, 60),
    "white": (235, 235, 235),
    "black": (20, 20, 20),
}
PALETTE_NAMES = list(PALETTE)


@ENCODERS.register("color")
class ColorPaletteEncoder(Encoder):
    def __init__(self, threshold: float = 60.0, **_: object) -> None:
        self.names = PALETTE_NAMES
        self.dim = len(self.names) + 1
        self.threshold = float(threshold)
        self._anchors = np.array(
            [PALETTE[n] for n in self.names], dtype=np.float32
        )  # [C, 3]

    def encode_images(self, frames: np.ndarray) -> np.ndarray:
        if len(frames) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        x = np.asarray(frames, dtype=np.float32)          # [N,H,W,3]
        n = x.shape[0]
        pixels = x.reshape(n, -1, 3)                       # [N,P,3]
        # distance to each palette anchor
        d = np.linalg.norm(pixels[:, :, None, :] - self._anchors[None, None], axis=-1)
        nearest = d.argmin(axis=-1)                        # [N,P]
        near_enough = d.min(axis=-1) < self.threshold      # [N,P]

        feats = np.zeros((n, self.dim), dtype=np.float32)
        for c in range(len(self.names)):
            feats[:, c] = ((nearest == c) & near_enough).mean(axis=1)
        feats[:, -1] = (~near_enough).mean(axis=1)
        # emphasise rare colours: a tiny red square should not be drowned out
        feats[:, : len(self.names)] = np.sqrt(feats[:, : len(self.names)])
        return l2_normalize(feats)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            low = t.lower()
            hits = [c for c, name in enumerate(self.names) if name in low]
            if hits:
                for c in hits:
                    out[i, c] = 1.0
            else:
                out[i, -1] = 1.0
        return l2_normalize(out)
