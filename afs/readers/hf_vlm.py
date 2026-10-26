"""Frozen vision-language reader (Qwen2.5-VL and friends) via transformers.

Scoring: we build a multiple-choice prompt, run one forward pass, and read the
logits of the option letters ("A", "B", ...) at the first generated position.
One forward pass per question, calibrated per-option scores for free, and no
sampling noise - which matters because we are measuring small differences
between selectors.

GPU strongly recommended. Imports are lazy so the rest of the repo works
without transformers installed.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np

from afs.data.base import QASample
from afs.readers.base import Reader, ReaderOutput
from afs.registry import READERS

LETTERS = "ABCDEFGH"

PROMPT = (
    "Answer the multiple-choice question about the video. "
    "Reply with the letter of the correct option only.\n\n{body}\n\nAnswer:"
)


@READERS.register("hf_vlm")
class HFVLMReader(Reader):
    needs_pixels = True
    # ~0.6 TFLOPs per 224px frame for a 3B VL model at 256 visual tokens;
    # override in the config once you measure it properly with fvcore.
    gflops_per_frame = 600.0

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str | None = None,
        dtype: str = "bfloat16",
        max_pixels: int = 224 * 224,
        **_: object,
    ) -> None:
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        torch_dtype = getattr(torch, dtype)
        if device == "cpu":
            torch_dtype = torch.float32

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
        except ImportError:  # older transformers / different family
            from transformers import AutoModelForVision2Seq as ModelCls

        self.processor = AutoProcessor.from_pretrained(
            model_name, max_pixels=max_pixels
        )
        self.model = (
            ModelCls.from_pretrained(model_name, torch_dtype=torch_dtype)
            .to(device)
            .eval()
        )
        self.tokenizer = self.processor.tokenizer
        self._letter_ids = {
            L: self.tokenizer.encode(L, add_special_tokens=False)[0] for L in LETTERS
        }

    def _build_inputs(self, sample: QASample, frames: np.ndarray):
        from PIL import Image

        images = [Image.fromarray(f) for f in frames]
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": PROMPT.format(body=sample.prompt())})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.processor(
            text=[text], images=images, return_tensors="pt", padding=True
        ).to(self.device)

    def answer(self, sample: QASample, frames: np.ndarray, frame_idx: np.ndarray) -> ReaderOutput:
        t0 = time.perf_counter()
        inputs = self._build_inputs(sample, frames)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        scores: List[float] = [
            float(logits[self._letter_ids[LETTERS[i]]].item())
            for i in range(len(sample.options))
        ]
        return ReaderOutput(
            int(np.argmax(scores)), scores, time.perf_counter() - t0
        )
