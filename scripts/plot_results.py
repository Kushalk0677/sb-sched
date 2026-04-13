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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.plotting import (
    plot_exp1_vr_rr,
    plot_exp2_pmax_sensitivity,
    plot_exp4_motion_stress,
    plot_exp5_overhead,
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

    exps_to_plot = list(range(1, 7)) if args.all else [args.exp]

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
        else:
            print(f"  [INFO] Exp {exp} — no dedicated plot function (check results CSV)")


if __name__ == "__main__":
    main()
