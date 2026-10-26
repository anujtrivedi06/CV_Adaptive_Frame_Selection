"""Phase 1: decode every video once and cache its frame features.

    python scripts/build_features.py --config configs/toy.yaml
    python scripts/build_features.py --config configs/nextqa.yaml

Everything downstream reads the cache, so this is the only slow step.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afs.config import load_config  # noqa: E402
from afs.eval import build_features  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("opts", nargs="*", help="dotted overrides, e.g. frames.fps=2")
    args = p.parse_args()

    cfg = load_config(args.config, args.opts)
    t0 = time.perf_counter()
    n = build_features(cfg, overwrite=args.overwrite)
    dt = time.perf_counter() - t0
    print(f"cached {n} videos in {dt:.1f}s -> {cfg.paths['cache_dir']}")


if __name__ == "__main__":
    main()
