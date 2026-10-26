from afs.registry import SELECTORS
from afs.selectors.base import SelectionContext, Selector
from afs.selectors import baselines  # noqa: F401
from afs.selectors import submodular  # noqa: F401
from afs.selectors import oracle  # noqa: F401
from afs.selectors.adaptive import allocate_budgets, difficulty_score

__all__ = [
    "SELECTORS",
    "SelectionContext",
    "Selector",
    "build_selector",
    "allocate_budgets",
    "difficulty_score",
]


def build_selector(cfg):
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return SELECTORS.build(cfg["name"], **kwargs)
