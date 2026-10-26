"""Name -> class registries, so configs can say `selector.name: submodular`."""

from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: Dict[str, type] = {}

    def register(self, name: str) -> Callable[[type], type]:
        def wrap(cls: type) -> type:
            if name in self._items:
                raise KeyError(f"{self.kind} {name!r} already registered")
            self._items[name] = cls
            cls.registry_name = name  # type: ignore[attr-defined]
            return cls

        return wrap

    def build(self, name: str, **kwargs):
        if name not in self._items:
            raise KeyError(
                f"unknown {self.kind} {name!r}; available: {sorted(self._items)}"
            )
        return self._items[name](**kwargs)

    def get(self, name: str) -> type:
        return self._items[name]

    def names(self):
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


SELECTORS: Registry = Registry("selector")
ENCODERS: Registry = Registry("encoder")
READERS: Registry = Registry("reader")
DATASETS: Registry = Registry("dataset")
