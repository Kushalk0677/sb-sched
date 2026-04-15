#!/usr/bin/env python3
"""
scripts/plot_results.py

Generate all paper figures from saved CSV results.

Usage:
    python scripts/plot_results.py --exp 1
    python scripts/plot_results.py --all
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    def load(path):
        p = Path(path)
        if not p.exists():
            print(f"  [WARN] {path} not found — run experiment first")
            return None
        return pd.read_csv(p)

    exps_to_plot = list(range(1, 12)) if args.all else [args.exp]

    for exp in exps_to_plot:
        if exp == 1:
            df = load("results/exp1/exp1_results.csv")
            if df is not None:
                plot_exp1_vr_rr(df)
        elif exp == 2:
            df = load("results/exp2/exp2_results.csv")
            if df is not None:
                plot_exp2_pmax_sensitivity(df)
        elif exp == 4:
            df = load("results/exp4/exp4_results.csv")
            if df is not None:
                plot_exp4_motion_stress(df)
        elif exp == 5:
            df = load("results/exp5/exp5_results.csv")
            if df is not None:
                plot_exp5_overhead(df)
        elif exp == 7:
            df = load("results/exp7/exp7_outage_robustness.csv")
            if df is not None:
                plot_exp7_outage_robustness(df)
        elif exp == 8:
            df = load("results/exp8/exp8_latency_jitter.csv")
            if df is not None:
                plot_exp8_latency_jitter(df)
        elif exp == 9:
            df = load("results/exp9/exp9_resource_budget.csv")
            if df is not None:
                plot_exp9_resource_budget(df)
        elif exp == 10:
            df = load("results/exp10/exp10_motion_transitions.csv")
            if df is not None:
                plot_exp10_motion_transitions(df)
        elif exp == 11:
            df = load("results/exp11/exp11_pareto.csv")
            if df is not None:
                plot_exp11_pareto(df)
        else:
            print(f"  [INFO] Exp {exp} — no dedicated plot function (check results CSV)")


if __name__ == "__main__":
    main()
