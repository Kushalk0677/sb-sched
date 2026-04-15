#!/usr/bin/env python3
"""
scripts/run_all.py

Run all 6 experiments and generate all paper figures.
Expects dataset paths to be set in configs/config.yaml.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --skip 5 6   # Skip experiments 5 and 6
"""

import argparse
import yaml
import sys
import time
from pathlib import Path

# Put the project root on sys.path so "from src.experiments..." works
# consistently for both run_all.py and run_experiment.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip", nargs="*", type=int, default=[],
        help="Experiment numbers to skip (e.g. --skip 5 6)",
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to YAML config (relative to project root)",
    )
    parser.add_argument(
        "--paper-only", action="store_true",
        help="For Exp 1, use only the paper-safe baseline subset",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Imports use the src package via the project root on sys.path
    from src.experiments.exp1_baseline_comparison import run_exp1
    from src.experiments.exp2_6 import run_exp2, run_exp3, run_exp4, run_exp5, run_exp6
    from src.experiments.extended_experiments import run_exp7_outage_robustness, run_exp8_latency_jitter, run_exp9_resource_budget, run_exp10_motion_transitions, run_exp11_pareto
    from src.utils.plotting import (
        plot_exp1_vr_rr,
        plot_exp2_pmax_sensitivity,
        plot_exp4_motion_stress,
        plot_exp5_overhead,
        plot_exp7_outage_robustness,
        plot_exp8_latency_jitter,
        plot_exp9_resource_budget,
        plot_exp10_motion_transitions,
        plot_exp11_pareto,
    )

    results = {}
    t_start = time.time()

    def run(exp_num, fn):
        if exp_num in args.skip:
            print(f"\n[Exp {exp_num}] SKIPPED")
            return None
        print(f"\n{'=' * 60}")
        print(f"  EXPERIMENT {exp_num}")
        print(f"{'=' * 60}")
        t0 = time.time()
        result = fn(config)
        print(f"[Exp {exp_num}] Done in {time.time() - t0:.1f}s")
        return result

    results[1] = run(1, lambda cfg: run_exp1(cfg, include_advanced=not args.paper_only))
    results[2] = run(2, run_exp2)
    results[3] = run(3, run_exp3)
    results[4] = run(4, run_exp4)
    results[5] = run(5, run_exp5)
    results[6] = run(6, run_exp6)
    results[7] = run(7, run_exp7_outage_robustness)
    results[8] = run(8, run_exp8_latency_jitter)
    results[9] = run(9, run_exp9_resource_budget)
    results[10] = run(10, run_exp10_motion_transitions)
    results[11] = run(11, run_exp11_pareto)

    # Generate figures
    print("\n\nGenerating figures...")
    if results[1] is not None and not results[1].empty:
        plot_exp1_vr_rr(results[1])
    if results[2] is not None and not results[2].empty:
        plot_exp2_pmax_sensitivity(results[2])
    if results[4] is not None and not results[4].empty:
        plot_exp4_motion_stress(results[4])
    if results[5] is not None and not results[5].empty:
        plot_exp5_overhead(results[5])
    if results[7] is not None and not results[7].empty:
        plot_exp7_outage_robustness(results[7])
    if results[8] is not None and not results[8].empty:
        plot_exp8_latency_jitter(results[8])
    if results[9] is not None and not results[9].empty:
        plot_exp9_resource_budget(results[9])
    if results[10] is not None and not results[10].empty:
        plot_exp10_motion_transitions(results[10])
    if results[11] is not None and not results[11].empty:
        plot_exp11_pareto(results[11])

    total = time.time() - t_start
    print(f"\n\nAll experiments complete in {total / 60:.1f} minutes.")
    print("Results in results/  |  Figures in results/expN/")


if __name__ == "__main__":
    main()
