from afs.encoders.base import Encoder, l2_normalize
from afs.encoders import color  # noqa: F401  (registers "color")

__all__ = ["Encoder", "l2_normalize", "build_encoder"]


def build_encoder(cfg):
    from afs.registry import ENCODERS

    name = cfg["name"]
    if name == "hf_clip":
        from afs.encoders import hf_clip  # noqa: F401  (lazy: needs transformers)
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return ENCODERS.build(name, **kwargs)
