"""
src/experiments/exp1_baseline_comparison.py

Experiment 1: Core baseline comparison.
Runs all methods on selected (or all) sequences across selected datasets.
Produces the primary results table: VR, RMSE, ESR, RR.
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

from baselines.main import (
    FixedHighScheduler,
    FixedLowScheduler,
    FixedMatchedScheduler,
    HeuristicAdaptiveScheduler,
    EventTriggeredScheduler,
)
from baselines.advanced import (
    AoIMinScheduler,
    WhittleIndexScheduler,
    PeriodicOptimalScheduler,
    CDKFGradientScheduler,
    MIQPOptimalScheduler,
    DRLScheduler,
    DelayAwarePolicyScheduler,
)
from scheduler.sb_sched import StalenessScheduler
from datasets.models import get_model
from experiments.runner import run_single
from utils.metrics import summarise_results

METHODS = {
    "fixed_high":    (FixedHighScheduler,    {}),
    "fixed_low":     (FixedLowScheduler,     {}),
    "fixed_matched": (FixedMatchedScheduler, {"native_rates": None}),
    "heuristic":     (HeuristicAdaptiveScheduler, {}),
    "aoi_min":       (AoIMinScheduler,       {}),
    "whittle":       (WhittleIndexScheduler, {"Q_diag": None}),
    "periodic_opt":  (PeriodicOptimalScheduler, {"native_rates": None}),
    "event_trigger": (EventTriggeredScheduler, {}),
    "cd_kf":         (CDKFGradientScheduler, {"native_rates": None}),
    "miqp_opt":      (MIQPOptimalScheduler,  {"native_rates": None}),
    "drl_sched":     (DRLScheduler,          {}),
    "delay_aware":   (DelayAwarePolicyScheduler, {}),
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
              "2011_09_26_drive_0028_sync",
              "2011_09_30_drive_0028_sync", "2011_09_30_drive_0033_sync",
              "2011_10_03_drive_0027_sync"],
    "tumvi": ["room1", "room2", "room3", "corridor1", "corridor2"],
}


def parse_kitti_sequence(sequence: str) -> tuple[str, str]:
    """
    Parse a KITTI sequence string of the form
    '2011_09_26_drive_0001_sync' into (date, drive).

    Returns:
        date  – e.g. "2011_09_26"
        drive – e.g. "0001"

    Raises:
        ValueError if the string does not match the expected pattern.
    """
    m = re.fullmatch(
        r"(\d{4}_\d{2}_\d{2})_drive_(\d{4})_sync",
        sequence.strip(),
    )
    if not m:
        raise ValueError(
            f"Cannot parse KITTI sequence '{sequence}'. "
            "Expected format: YYYY_MM_DD_drive_NNNN_sync"
        )
    return m.group(1), m.group(2)


def run_exp1(
    config: dict,
    results_dir: str = "results/exp1",
    filter_dataset: str | None = None,
    filter_sequence: str | None = None,
) -> pd.DataFrame:
    """
    Run Experiment 1 across methods, datasets, and sequences.

    Args:
        config:          Loaded YAML config dict
        results_dir:     Directory for CSV outputs
        filter_dataset:  If given, restrict to this dataset ("euroc", "kitti", "tumvi")
        filter_sequence: If given, restrict to this single sequence name
    Returns:
        DataFrame with all results
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    # ── Dataset filter ────────────────────────────────────────────────────
    sequences_to_run = {
        ds: seqs
        for ds, seqs in SEQUENCES.items()
        if filter_dataset is None or ds == filter_dataset.lower()
    }
    if not sequences_to_run:
        raise ValueError(
            f"filter_dataset='{filter_dataset}' matched no known datasets. "
            f"Choose from: {list(SEQUENCES)}"
        )

    for dataset, sequences in sequences_to_run.items():

        # ── Sequence filter ───────────────────────────────────────────────
        if filter_sequence is not None:
            if filter_sequence not in sequences:
                print(
                    f"[warn] sequence '{filter_sequence}' not in {dataset} "
                    f"sequence list — skipping"
                )
                continue
            sequences = [filter_sequence]

        root = config["datasets"][dataset]["root"]
        native_rates = {
            "euroc":  {"camera": 20.0, "imu": 200.0},
            "kitti":  {"gps": 10.0,    "imu": 100.0},
            "tumvi":  {"camera": 20.0, "imu": 200.0},
        }[dataset]

        # ── First pass: Fixed-High to get baseline ESR ────────────────────
        fixed_high_results = {}
        for seq in sequences:
            kw = {}
            if dataset == "kitti":
                date, drive = parse_kitti_sequence(seq)
                kw = {"date": date, "drive": drive}

            result = run_single(
                scheduler_cls=FixedHighScheduler,
                scheduler_kwargs={},
                dataset=dataset,
                sequence=seq,
                dataset_root=root,
                method_name="fixed_high",
                **kw,
            )
            fixed_high_results[seq] = result.esr
            all_results.append(result)

        # ── Second pass: all other methods ────────────────────────────────
        for method_name, (cls, base_kwargs) in METHODS.items():
            if method_name == "fixed_high":
                continue  # Already done above

            # Inject runtime-dependent kwargs (shallow-copy to avoid mutation)
            kwargs = dict(base_kwargs)
            if "native_rates" in kwargs:
                kwargs["native_rates"] = native_rates
            if "Q_diag" in kwargs:
                model_fn = get_model(dataset)
                _, Q, _, _, _ = model_fn()
                kwargs["Q_diag"] = np.diag(Q)

            for seq in sequences:
                loader_kw = {}
                if dataset == "kitti":
                    date, drive = parse_kitti_sequence(seq)
                    loader_kw = {"date": date, "drive": drive}

                result = run_single(
                    scheduler_cls=cls,
                    scheduler_kwargs=kwargs,
                    dataset=dataset,
                    sequence=seq,
                    dataset_root=root,
                    method_name=method_name,
                    fixed_high_esr=fixed_high_results.get(seq),
                    **loader_kw,
                )
                all_results.append(result)
                print(
                    f"[{method_name}] {dataset}/{seq}: "
                    f"VR={result.violation_rate:.3f}, "
                    f"RMSE={result.rmse_m:.4f}m"
                )

    if not all_results:
        print("[warn] No results collected — check dataset/sequence filters "
              "and that the dataset root path is correct.")
        return pd.DataFrame()

    # ── Build result DataFrame ────────────────────────────────────────────
    rows = []
    for r in all_results:
        sensor_names = list(r.esr.keys())
        row = {
            "method":   r.method,
            "dataset":  r.dataset,
            "sequence": r.sequence,
            "vr":       r.violation_rate,
            "rmse_m":   r.rmse_m,
            "p_max":    r.p_max,
        }
        for sname in sensor_names:
            row[f"esr_{sname}"]  = r.esr.get(sname, 0)
            row[f"rr_{sname}"]   = r.rate_reduction.get(sname, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/exp1_results.csv", index=False)
    print(f"\nResults saved to {results_dir}/exp1_results.csv")

    summary = summarise_results(all_results)
    print("\n--- Summary ---")
    for key, s in summary.items():
        print(
            f"{key:40s}  VR={s['vr_mean']:.3f}±{s['vr_std']:.3f}  "
            f"RMSE={s['rmse_mean']:.4f}±{s['rmse_std']:.4f}"
        )

    return df
