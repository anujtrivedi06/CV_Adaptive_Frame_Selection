"""Small shared helpers."""

from __future__ import annotations

import hashlib


def stable_seed(*parts: object) -> int:
    """A process-independent 32-bit seed derived from arbitrary values.

    Python salts ``hash()`` of strings per process (PYTHONHASHSEED), so using
    it anywhere in an experiment makes results irreproducible between runs.
    Every random draw in this repo goes through here instead.
    """
    payload = "|".join(repr(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha1(payload).digest()[:4], "big")
