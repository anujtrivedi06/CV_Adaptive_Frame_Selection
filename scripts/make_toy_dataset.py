"""Generate the synthetic needle-in-a-haystack videos.

    python scripts/make_toy_dataset.py --root data/toy --n-videos 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afs.data.toy import build_toy_dataset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/toy")
    p.add_argument("--n-videos", type=int, default=32)
    p.add_argument("--n-frames", type=int, default=120)
    p.add_argument("--needle-len", type=int, default=4)
    p.add_argument("--height", type=int, default=96)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mode",
        default="single",
        choices=["single", "order"],
        help="'single' = one needle; 'order' = two separated events, both needed",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    manifest = build_toy_dataset(
        root=args.root,
        n_videos=args.n_videos,
        n_frames=args.n_frames,
        needle_len=args.needle_len,
        size=(args.height, args.width),
        seed=args.seed,
        mode=args.mode,
        overwrite=args.overwrite,
    )
    print(f"wrote {manifest}")
    print(f"videos in {Path(args.root) / 'videos'}")


if __name__ == "__main__":
    main()
