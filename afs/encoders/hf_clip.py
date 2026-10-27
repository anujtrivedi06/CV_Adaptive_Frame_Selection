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
        # transformers renamed `torch_dtype` to `dtype` in v5; support both.
        try:
            model = AutoModel.from_pretrained(model_name, dtype=torch_dtype)
        except TypeError:
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.model = model.to(device).eval()
        self.dim = int(
            getattr(self.model.config, "projection_dim", None)
            or self.model.config.text_config.hidden_size
        )

    @property
    def _no_grad(self):
        return self.torch.no_grad

    @staticmethod
    def _pool(out):
        """Get the embedding tensor out of whatever the model returned.

        transformers v4 returns a plain tensor from get_image_features /
        get_text_features; v5 returns a BaseModelOutputWithPooling. Support
        both, so the repo is not pinned to one major version.
        """
        if hasattr(out, "shape"):           # v4: already a tensor
            return out
        for key in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(out, key, None)
            if value is not None:
                return value
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is not None:              # last resort: mean-pool tokens
            return hidden.mean(dim=1)
        raise TypeError(f"cannot extract embeddings from {type(out).__name__}")

    def encode_images(self, frames: np.ndarray) -> np.ndarray:
        if len(frames) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        from PIL import Image

        images = [Image.fromarray(f) for f in frames]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self._no_grad():
            feats = self._pool(self.model.get_image_features(**inputs))
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
            feats = self._pool(self.model.get_text_features(**inputs))
        return l2_normalize(feats.float().cpu().numpy())
