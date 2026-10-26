"""CLIP / SigLIP frame+text encoder via HuggingFace transformers.

This is the encoder for real datasets. Imports are lazy so the repo still runs
end-to-end on a machine without `transformers` installed.

Recommended:
    google/siglip-base-patch16-224      (strong, 203M)
    openai/clip-vit-base-patch32        (fast baseline)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from afs.encoders.base import Encoder, l2_normalize
from afs.registry import ENCODERS


@ENCODERS.register("hf_clip")
class HFClipEncoder(Encoder):
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str | None = None,
        batch_size: int = 64,
        dtype: str = "float32",
        **_: object,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        torch_dtype = getattr(torch, dtype)
        if device == "cpu" and torch_dtype == torch.float16:
            torch_dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = (
            AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype)
            .to(device)
            .eval()
        )
        self.dim = int(
            getattr(self.model.config, "projection_dim", None)
            or self.model.config.text_config.hidden_size
        )

    @property
    def _no_grad(self):
        return self.torch.no_grad

    def encode_images(self, frames: np.ndarray) -> np.ndarray:
        if len(frames) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        from PIL import Image

        images = [Image.fromarray(f) for f in frames]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self._no_grad():
            feats = self.model.get_image_features(**inputs)
        return l2_normalize(feats.float().cpu().numpy())

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        inputs = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(self.device)
        with self._no_grad():
            feats = self.model.get_text_features(**inputs)
        return l2_normalize(feats.float().cpu().numpy())
