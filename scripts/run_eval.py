"""Phase 2+: run one configuration, or sweep selectors x budgets.

    # single run
    python scripts/run_eval.py --config configs/toy.yaml selector.name=submodular budget.K=8

    # the whole Pareto sweep (this is the deliverable)
    python scripts/run_eval.py --config configs/toy.yaml --sweep

    # adaptive budget, same average cost as fixed K=8
    python scripts/run_eval.py --config configs/toy.yaml budget.adaptive=true budget.K=8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afs.config import apply_override, load_config  # noqa: E402
from afs.eval import run_eval  # noqa: E402


def _print_row(summary) -> None:
    extra = ""
    if summary.evidence_recall is not None:
        extra = f"  evidence-recall {summary.evidence_recall:6.3f}"
    print(
        f"{summary.selector:<16s} K={summary.budget:<5g} "
        f"acc {summary.accuracy:6.3f}+-{summary.accuracy_stderr:.3f}  "
        f"frames {summary.mean_frames:5.2f}  "
        f"GFLOPs {summary.mean_total_gflops:9.1f}{extra}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--sweep", action="store_true", help="use the config's sweep block")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("opts", nargs="*", help="dotted overrides, e.g. selector.beta=0.5")
    args = p.parse_args()

    cfg = load_config(args.config, args.opts)

    if not args.sweep:
        out = run_eval(cfg, progress=not args.quiet)
        print()
        _print_row(out["summary"])
        print(f"-> {out['out_dir']}")
        return

    selectors = cfg.sweep["selectors"]
    budgets = cfg.sweep["budgets"]
    print(f"sweep: {len(selectors)} selectors x {len(budgets)} budgets\n")
    for name in selectors:
        for k in budgets:
            run_cfg = load_config(args.config, args.opts)
            apply_override(run_cfg, f"selector.name={name}")
            out = run_eval(
                run_cfg, budget=k, run_name=f"{name}_K{k}", progress=False
            )
            _print_row(out["summary"])
    print(f"\nleaderboard -> {Path(cfg.paths['runs_dir']) / 'leaderboard.jsonl'}")


if __name__ == "__main__":
    main()
