from afs.readers.base import Reader, ReaderOutput
from afs.readers import mock  # noqa: F401  (registers "needle", "similarity")

__all__ = ["Reader", "ReaderOutput", "build_reader"]


def build_reader(cfg, store=None):
    from afs.registry import READERS

    name = cfg["name"]
    if name == "hf_vlm":
        from afs.readers import hf_vlm  # noqa: F401  (lazy: needs transformers)
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    if name == "similarity":
        kwargs["store"] = store
    reader = READERS.build(name, **kwargs)
    if "gflops_per_frame" in cfg:
        reader.gflops_per_frame = float(cfg["gflops_per_frame"])
    return reader
