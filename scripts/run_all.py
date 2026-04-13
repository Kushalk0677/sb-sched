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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", type=int, default=[],
                        help="Experiment numbers to skip")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results = {}
    t_start = time.time()

    def run(exp_num, fn):
        if exp_num in args.skip:
            print(f"\n[Exp {exp_num}] SKIPPED")
            return None
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT {exp_num}")
        print(f"{'='*60}")
        t0 = time.time()
        result = fn(config)
        print(f"[Exp {exp_num}] Done in {time.time()-t0:.1f}s")
        return result

    from experiments.exp1_baseline_comparison import run_exp1
    from experiments.exp2_6 import run_exp2, run_exp3, run_exp4, run_exp5, run_exp6
    from utils.plotting import (plot_exp1_vr_rr, plot_exp2_pmax_sensitivity,
                                 plot_exp4_motion_stress, plot_exp5_overhead)

    results[1] = run(1, run_exp1)
    results[2] = run(2, run_exp2)
    results[3] = run(3, run_exp3)
    results[4] = run(4, run_exp4)
    results[5] = run(5, run_exp5)
    results[6] = run(6, run_exp6)

    # Generate figures
    print("\n\nGenerating figures...")
    if results[1] is not None:
        plot_exp1_vr_rr(results[1])
    if results[2] is not None:
        plot_exp2_pmax_sensitivity(results[2])
    if results[4] is not None:
        plot_exp4_motion_stress(results[4])
    if results[5] is not None:
        plot_exp5_overhead(results[5])

    total = time.time() - t_start
    print(f"\n\nAll experiments complete in {total/60:.1f} minutes.")
    print("Results in results/  |  Figures in results/expN/")


if __name__ == "__main__":
    main()
