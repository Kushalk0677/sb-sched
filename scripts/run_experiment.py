#!/usr/bin/env python3
"""
scripts/run_experiment.py

Run a single experiment with optional dataset and sequence filters.

Usage:
    # Single sequence (fast iteration during development)
    python scripts/run_experiment.py --exp 1 --dataset euroc --sequence MH_01_easy

    # Single dataset, all its sequences
    python scripts/run_experiment.py --exp 1 --dataset euroc

    # All datasets and sequences
    python scripts/run_experiment.py --exp 1 --all

    # Other experiments (always run on fixed subset internally)
    python scripts/run_experiment.py --exp 2
"""

import argparse
import yaml
import sys
from pathlib import Path

# Ensure the project root is on sys.path so "from src.experiments..." works.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path: str = "configs/config.yaml") -> dict:
    config_path = PROJECT_ROOT / path
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SB-Sched experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--exp", type=int, required=True, choices=range(1, 12),
        help="Experiment number (1–11)",
    )
    parser.add_argument(
        "--dataset", default=None, choices=["euroc", "kitti", "tumvi"],
        help="Restrict to one dataset (default: all)",
    )
    parser.add_argument(
        "--sequence", default=None,
        help="Restrict to one sequence name (requires --dataset)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Explicitly run all datasets and sequences (default behaviour)",
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to YAML config (relative to project root)",
    )
    args = parser.parse_args()

    # Validate: --sequence requires --dataset
    if args.sequence is not None and args.dataset is None:
        parser.error("--sequence requires --dataset to be specified")

    config = load_config(args.config)

    if args.exp == 1:
        from src.experiments.exp1_baseline_comparison import run_exp1
        df = run_exp1(
            config,
            filter_dataset=args.dataset,
            filter_sequence=args.sequence,
        )
        if not df.empty:
            print("\n=== Mean VR and RMSE per method / dataset ===")
            print(df.groupby(["method", "dataset"])[["vr", "rmse_m"]].mean().to_string())

    elif args.exp == 2:
        from src.experiments.exp2_6 import run_exp2
        df = run_exp2(config)
        print(df.to_string())

    elif args.exp == 3:
        from src.experiments.exp2_6 import run_exp3
        df = run_exp3(config)
        print(df.to_string())

    elif args.exp == 4:
        from src.experiments.exp2_6 import run_exp4
        df = run_exp4(config)
        print(df.to_string())

    elif args.exp == 5:
        from src.experiments.exp2_6 import run_exp5
        df = run_exp5(config)
        print(df.to_string())

    elif args.exp == 6:
        from src.experiments.exp2_6 import run_exp6
        df = run_exp6(config)
        print(df.to_string())

    elif args.exp == 7:
        from src.experiments.extended_experiments import run_exp7_outage_robustness
        df = run_exp7_outage_robustness(config)
        print(df.to_string())

    elif args.exp == 8:
        from src.experiments.extended_experiments import run_exp8_latency_jitter
        df = run_exp8_latency_jitter(config)
        print(df.to_string())

    elif args.exp == 9:
        from src.experiments.extended_experiments import run_exp9_resource_budget
        df = run_exp9_resource_budget(config)
        print(df.to_string())

    elif args.exp == 10:
        from src.experiments.extended_experiments import run_exp10_motion_transitions
        df = run_exp10_motion_transitions(config)
        print(df.to_string())

    elif args.exp == 11:
        from src.experiments.extended_experiments import run_exp11_pareto
        df = run_exp11_pareto(config)
        print(df.to_string())


if __name__ == "__main__":
    main()