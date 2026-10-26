"""Mock readers - let the whole harness run and be unit-tested without a GPU.

``needle`` is the one that matters: on the toy dataset it answers correctly iff
at least one selected frame lies inside the ground-truth evidence window, and
guesses otherwise. That is an idealised reader, but it isolates exactly the
quantity the project is about - *did the selector retrieve the evidence?* - so
the accuracy-vs-K curves it produces are already meaningful.
"""

from __future__ import annotations

import time

import numpy as np

from afs.data.base import QASample, covers_evidence
from afs.readers.base import Reader, ReaderOutput
from afs.registry import READERS
from afs.utils import stable_seed


@READERS.register("needle")
class NeedleMockReader(Reader):
    """Correct iff the selection covers *every* evidence window, else guesses.

    An idealised reader, deliberately: it isolates the one quantity the project
    is about - did the selector retrieve the evidence? - with no confound from
    the reader's own reasoning ability.
    """

    gflops_per_frame = 1.0

    def __init__(self, seed: int = 0, **_: object) -> None:
        self.seed = seed

    def answer(self, sample: QASample, frames: np.ndarray, frame_idx: np.ndarray) -> ReaderOutput:
        t0 = time.perf_counter()
        hit = bool(covers_evidence(sample, frame_idx))
        n_opt = len(sample.options)

        scores = [0.0] * n_opt
        if hit:
            pred = sample.answer_idx
            scores[pred] = 4.0
        else:
            rng = np.random.default_rng(stable_seed(self.seed, sample.qid))
            pred = int(rng.integers(n_opt))
            scores[pred] = 0.2
        return ReaderOutput(pred, scores, time.perf_counter() - t0)


@READERS.register("similarity")
class SimilarityReader(Reader):
    """Dataset-agnostic CPU reader: pick the option whose text embedding is
    closest to the selected frames. Weak, but real - useful as a sanity check
    on a true dataset before the VLM is available."""

    gflops_per_frame = 1.0

    def __init__(self, store=None, **_: object) -> None:
        self.store = store  # a FeatureStore, injected by the evaluator

    def answer(self, sample: QASample, frames: np.ndarray, frame_idx: np.ndarray) -> ReaderOutput:
        t0 = time.perf_counter()
        if self.store is None:
            raise ValueError("SimilarityReader needs a FeatureStore")
        feats, _ = self.store.get_video(sample)
        sel = feats[np.asarray(frame_idx, dtype=int)]
        pooled = sel.mean(axis=0)
        pooled /= np.linalg.norm(pooled) + 1e-8
        opts = self.store.get_text(
            [f"{sample.question} {o}" for o in sample.options]
        )
        scores = (opts @ pooled).astype(float).tolist()
        return ReaderOutput(int(np.argmax(scores)), scores, time.perf_counter() - t0)
