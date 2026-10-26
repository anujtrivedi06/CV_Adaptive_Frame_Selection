"""Turn runs/<dataset>/leaderboard.jsonl into the figures the report needs.

    python scripts/plot_pareto.py --runs runs/toy

Produces accuracy-vs-K and accuracy-vs-GFLOPs, one line per selector, with the
oracle drawn as a dashed ceiling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#1F3A5F", "#C44E52", "#55A868", "#8172B2", "#CCB974", "#64B5CD", "#937860"]


def load_rows(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # keep the last run for each (selector, budget) pair
    dedup = {(r["selector"], r["budget"]): r for r in rows}
    return sorted(dedup.values(), key=lambda r: (r["selector"], r["budget"]))


def plot(rows, xkey: str, xlabel: str, out: Path, logx: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=150)
    selectors = sorted({r["selector"] for r in rows})
    for i, sel in enumerate(selectors):
        pts = sorted(
            [r for r in rows if r["selector"] == sel], key=lambda r: r[xkey]
        )
        is_oracle = sel.startswith("oracle")
        ax.plot(
            [p[xkey] for p in pts],
            [p["accuracy"] for p in pts],
            marker="o" if not is_oracle else "",
            linestyle="--" if is_oracle else "-",
            linewidth=2.0 if is_oracle else 1.6,
            markersize=4,
            color="black" if is_oracle else COLORS[i % len(COLORS)],
            label=sel + (" (ceiling)" if is_oracle else ""),
        )
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("accuracy")
    ax.set_title("Adaptive frame selection: accuracy vs. cost")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="runs/toy")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    runs = Path(args.runs)
    rows = load_rows(runs / "leaderboard.jsonl")
    out_dir = Path(args.out or runs / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot(rows, "mean_frames", "frames fed to the reader (K)", out_dir / "acc_vs_K.png")
    plot(
        rows,
        "mean_total_gflops",
        "total GFLOPs per question (reader + selector overhead)",
        out_dir / "acc_vs_gflops.png",
        logx=True,
    )


if __name__ == "__main__":
    main()
