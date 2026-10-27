from afs.data.base import QASample, VideoQADataset
from afs.data import toy  # noqa: F401  (registers "toy")

__all__ = ["QASample", "VideoQADataset", "build_dataset"]


def build_dataset(cfg, seed: int = 0):
    """Build a dataset from the `dataset:` block of a config.

    ``limit`` is handled here rather than by the loaders, so every stage
    (feature caching, evaluation, plotting) sees exactly the same fixed
    question subset.
    """
    from afs.registry import DATASETS

    kwargs = {k: v for k, v in cfg.items() if k not in ("name", "limit")}
    name = cfg["name"]
    if name == "nextqa":
        from afs.data import nextqa  # noqa: F401  (lazy: needs pandas + the download)
    elif name == "nextgqa":
        from afs.data import nextgqa  # noqa: F401  (lazy: needs pandas + the download)
    dataset = DATASETS.build(name, **kwargs)

    limit = cfg.get("limit")
    if limit:
        dataset = dataset.subset(int(limit), seed=seed)
    return dataset
