from afs.encoders.base import Encoder, l2_normalize
from afs.encoders import color  # noqa: F401  (registers "color")

__all__ = [
    "Encoder",
    "l2_normalize",
    "build_encoder",
    "build_encoder_lazy",
    "LazyEncoder",
]


def build_encoder(cfg):
    from afs.registry import ENCODERS

    name = cfg["name"]
    if name == "hf_clip":
        from afs.encoders import hf_clip  # noqa: F401  (lazy: needs transformers)
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return ENCODERS.build(name, **kwargs)


class LazyEncoder:
    """Defers constructing the real encoder until something actually needs it.

    Loading SigLIP costs ~200 s. Once features and query vectors are cached,
    a run needs the encoder for nothing at all - but a sweep of 30 runs was
    still paying that cost 30 times, which defeats the entire point of the
    cache. With this wrapper a fully-cached sweep never touches the model.
    """

    def __init__(self, factory) -> None:
        self._factory = factory
        self._encoder = None

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = self._factory()
        return self._encoder

    @property
    def is_loaded(self) -> bool:
        return self._encoder is not None

    @property
    def dim(self) -> int:
        return self.encoder.dim

    @property
    def batch_size(self) -> int:
        return self.encoder.batch_size

    def encode_images(self, frames):
        return self.encoder.encode_images(frames)

    def encode_images_batched(self, frames):
        return self.encoder.encode_images_batched(frames)

    def encode_text(self, texts):
        return self.encoder.encode_text(texts)


def build_encoder_lazy(cfg) -> LazyEncoder:
    return LazyEncoder(lambda: build_encoder(cfg))
