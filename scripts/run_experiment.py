#!/usr/bin/env python3
"""
scripts/run_experiment.py

Run a single experiment.

Usage:
    python scripts/run_experiment.py --exp 1 --dataset euroc --sequence MH_01_easy
    python scripts/run_experiment.py --exp 2
    python scripts/run_experiment.py --exp 1 --all
"""

import argparse
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, required=True,
                        choices=[1, 2, 3, 4, 5, 6],
                        help="Experiment number (1–6)")
    parser.add_argument("--dataset", default="euroc",
                        choices=["euroc", "kitti", "tumvi"])
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--all", action="store_true",
                        help="Run on all sequences")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.exp == 1:
        from experiments.exp1_baseline_comparison import run_exp1
        df = run_exp1(config)
        print(df.groupby(["method", "dataset"])[["vr","rmse_m"]].mean())

    elif args.exp == 2:
        from experiments.exp2_6 import run_exp2
        df = run_exp2(config)
        print(df)

    elif args.exp == 3:
        from experiments.exp2_6 import run_exp3
        df = run_exp3(config)
        print(df)

    elif args.exp == 4:
        from experiments.exp2_6 import run_exp4
        df = run_exp4(config)
        print(df)

    elif args.exp == 5:
        from experiments.exp2_6 import run_exp5
        df = run_exp5(config)
        print(df)

    elif args.exp == 6:
        from experiments.exp2_6 import run_exp6
        df = run_exp6(config)
        print(df)


if __name__ == "__main__":
    main()
