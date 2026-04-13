"""
src/experiments/exp1_baseline_comparison.py

Experiment 1: Core baseline comparison.
Runs all 10 methods on all sequences across all 3 datasets.
Produces the primary results table: VR, RMSE, ESR, RR.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from ..baselines.baselines import (
    FixedHighScheduler, FixedLowScheduler, FixedMatchedScheduler,
    HeuristicAdaptiveScheduler, AoIMinScheduler, WhittleIndexScheduler,
    PeriodicOptimalScheduler, EventTriggeredScheduler, CDKFGradientScheduler,
    MIQPOptimalScheduler, DRLScheduler, DelayAwarePolicyScheduler,
)
from ..scheduler.sb_sched import StalenessScheduler
from ..datasets.models import get_model
from .runner import run_single
from ..utils.metrics import summarise_results


METHODS = {
    "fixed_high":    (FixedHighScheduler,    {}),
    "fixed_low":     (FixedLowScheduler,     {}),
    "fixed_matched": (FixedMatchedScheduler, {}),
    "heuristic":     (HeuristicAdaptiveScheduler, {}),
    "aoi_min":       (AoIMinScheduler,       {}),
    "whittle":       (WhittleIndexScheduler, {"Q_diag": None}),  # Set at runtime
    "periodic_opt":  (PeriodicOptimalScheduler, {"native_rates": None}),
    "event_trigger": (EventTriggeredScheduler, {}),
    "cd_kf":         (CDKFGradientScheduler, {"native_rates": None}),
    # ── new baselines (must cite) ──────────────────────────────────────────
    # Dutta 2023: MIQP with joint rate budget; native_rates injected at runtime
    "miqp_opt":      (MIQPOptimalScheduler,  {"native_rates": None}),
    # Alali 2024: DRL policy with online TD fine-tuning; no extra kwargs needed
    "drl_sched":     (DRLScheduler,          {}),
    # arXiv 2601 (Jan 2026): delay-aware staleness budget; zero-delay baseline
    # (d_i = 0 for all sensors → reduces to SB-Sched with delta).
    # For full delay experiments supply delays={"camera": 3, "imu": 1} etc.
    "delay_aware":   (DelayAwarePolicyScheduler, {}),
    # ── ours ──────────────────────────────────────────────────────────────
    "sb_sched":      (StalenessScheduler,    {}),
}

SEQUENCES = {
    "euroc": ["MH_01_easy", "MH_02_easy", "MH_03_medium",
              "MH_04_medium", "MH_05_difficult",
              "V1_01_easy", "V1_02_medium", "V1_03_difficult",
              "V2_01_easy", "V2_02_medium", "V2_03_difficult"],
    "kitti": ["2011_09_26_drive_0001_sync", "2011_09_26_drive_0002_sync",
              "2011_09_26_drive_0005_sync", "2011_09_26_drive_0019_sync",
              "2011_09_26_drive_0020_sync", "2011_09_26_drive_0027_sync",
              "2011_09_26_drive_0028_sync"],
    "tumvi": ["room1", "room2", "room3", "corridor1", "corridor2"],
}


def run_exp1(config: dict, results_dir: str = "results/exp1") -> pd.DataFrame:
    """
    Run Experiment 1 across all methods, datasets, and sequences.

    Args:
        config: Loaded YAML config dict
        results_dir: Where to save CSV outputs
    Returns:
        DataFrame with all results
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    for dataset, sequences in SEQUENCES.items():
        root = config["datasets"][dataset]["root"]
        native_rates = {
            "euroc":  {"camera": 20.0, "imu": 200.0},
            "kitti":  {"gps": 10.0, "imu": 100.0},
            "tumvi":  {"camera": 20.0, "imu": 200.0},
        }[dataset]

        # First: run Fixed-High to get baseline ESR
        fixed_high_results = {}
        for seq in sequences:
            result = run_single(
                scheduler_cls=FixedHighScheduler,
                scheduler_kwargs={},
                dataset=dataset, sequence=seq,
                dataset_root=root,
                method_name="fixed_high",
            )
            fixed_high_results[seq] = result.esr
            all_results.append(result)

        # Run all other methods
        for method_name, (cls, kwargs) in METHODS.items():
            if method_name == "fixed_high":
                continue   # Already done above

            # Inject runtime-dependent kwargs
            if "native_rates" in kwargs:
                kwargs = {**kwargs, "native_rates": native_rates}
            if "Q_diag" in kwargs:
                model_fn = get_model(dataset)
                F, Q, _, _, _ = model_fn()
                kwargs = {**kwargs, "Q_diag": np.diag(Q)}

            for seq in sequences:
                result = run_single(
                    scheduler_cls=cls,
                    scheduler_kwargs=kwargs,
                    dataset=dataset, sequence=seq,
                    dataset_root=root,
                    method_name=method_name,
                    fixed_high_esr=fixed_high_results.get(seq),
                )
                all_results.append(result)
                print(f"[{method_name}] {dataset}/{seq}: "
                      f"VR={result.violation_rate:.3f}, "
                      f"RMSE={result.rmse_m:.4f}m")

    # Build DataFrame
    rows = []
    for r in all_results:
        sensor_names = list(r.esr.keys())
        row = {
            "method": r.method,
            "dataset": r.dataset,
            "sequence": r.sequence,
            "vr": r.violation_rate,
            "rmse_m": r.rmse_m,
            "p_max": r.p_max,
        }
        for sname in sensor_names:
            row[f"esr_{sname}"] = r.esr.get(sname, 0)
            row[f"rr_{sname}"] = r.rate_reduction.get(sname, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/exp1_results.csv", index=False)
    print(f"\nResults saved to {results_dir}/exp1_results.csv")

    # Print summary table
    summary = summarise_results(all_results)
    print("\n--- Summary ---")
    for key, s in summary.items():
        print(f"{key:40s}  VR={s['vr_mean']:.3f}±{s['vr_std']:.3f}  "
              f"RMSE={s['rmse_mean']:.4f}±{s['rmse_std']:.4f}")

    return df
