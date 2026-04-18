#!/usr/bin/env python3
"""Run realistic-measurement experiments for EuRoC, KITTI, UCI gas drift, or one-month gas.

Examples
--------
EuRoC using the repository's current clean camera proxy plus realistic corruptions:
    python scripts/run_realism_benchmark.py \
        --dataset euroc \
        --sequence MH_03_medium

EuRoC using a precomputed external VO trajectory CSV (timestamp,x,y,z):
    python scripts/run_realism_benchmark.py \
        --dataset euroc \
        --sequence MH_03_medium \
        --precomputed-csv D:/data/euroc_vo/openvins_MH_03_medium.csv

KITTI with GPS realism profiles:
    python scripts/run_realism_benchmark.py \
        --dataset kitti \
        --sequence 2011_09_26_drive_0005_sync
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(PROJECT_ROOT / path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EuRoC / KITTI realism benchmarks for SB-Sched.")
    parser.add_argument("--dataset", required=True, choices=["euroc", "kitti", "uci_gas", "onemonth_gas"], help="Dataset to benchmark")
    parser.add_argument("--sequence", required=True, help="Sequence name. KITTI expects date_drive_sync format. UCI expects batch name like batch1. one-month gas can use any label (kept for output naming).")
    parser.add_argument("--config", default="configs/config.yaml", help="Project config YAML")
    parser.add_argument(
        "--precomputed-csv",
        default=None,
        help="Optional CSV with timestamp,x,y,z from ORB-SLAM/OpenVINS/VINS output for EuRoC",
    )
    parser.add_argument("--sensor-id", type=int, default=0, help="one-month gas: sensor_id to track")
    parser.add_argument("--feature", default="rs_r0_ratio", help="one-month gas: column to track")
    parser.add_argument("--uci-feature-index", type=int, default=2, help="uci_gas: 1-based feature index from sparse file")
    parser.add_argument("--results-dir", default="results/realism", help="Where CSV outputs should be written")
    args = parser.parse_args()

    config = load_config(args.config)
    from src.experiments.realism_benchmark import run_realism_benchmark

    full_df, summary_df = run_realism_benchmark(
        config=config,
        dataset=args.dataset,
        sequence=args.sequence,
        precomputed_csv=args.precomputed_csv,
        results_dir=args.results_dir,
        sensor_id=args.sensor_id,
        feature=args.feature,
        uci_feature_index=args.uci_feature_index,
    )

    print("\n=== SB-Sched realism summary ===")
    print(summary_df.to_string(index=False))
    print("\n=== All method rows saved ===")
    print(full_df[["dataset", "sequence", "scheduler", "setting", "vr", "aot", "rmse_m", "esr_primary"]].to_string(index=False))


if __name__ == "__main__":
    main()
