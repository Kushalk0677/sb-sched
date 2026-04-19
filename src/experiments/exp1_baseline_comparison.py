
"""
src/experiments/exp1_baseline_comparison.py

Experiment 1: Core baseline comparison.
Runs all paper baselines including Clairvoyant Lookahead (CL) (upper bound) and
Variance-Threshold (tuned event-trigger).
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

# Paper-safe baselines
from ..baselines.main import (
    FixedHighScheduler,
    FixedLowScheduler,
    FixedMatchedScheduler,
    HeuristicAdaptiveScheduler,
    AoIMinScheduler,
    PeriodicOptimalScheduler,
    EventTriggeredScheduler,
    DelayAwarePolicyScheduler,
    ClairvoyantLookaheadScheduler,
    VarianceThresholdScheduler,
)

from ..scheduler.sb_sched import StalenessScheduler
from ..datasets.models import get_model
from .runner import run_single
from ..utils.metrics import summarise_results


PAPER_METHODS = {
    "fixed_high":    (FixedHighScheduler,    {}),
    "fixed_low":     (FixedLowScheduler,     {}),
    "fixed_matched": (FixedMatchedScheduler, {"native_rates": None}),
    "heuristic":     (HeuristicAdaptiveScheduler, {}),
    "aoi_min":       (AoIMinScheduler,       {}),
    "periodic_opt":  (PeriodicOptimalScheduler, {"native_rates": None}),
    "event_trigger": (EventTriggeredScheduler, {}),
    "delay_aware":   (DelayAwarePolicyScheduler, {}),
    "var_threshold": (VarianceThresholdScheduler, {}),
    "clairvoyant_lookahead": (ClairvoyantLookaheadScheduler, {}),
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


def _native_rates_for(dataset: str) -> dict[str, float]:
    return {
        "euroc":  {"camera": 20.0, "imu": 200.0},
        "kitti":  {"gps": 10.0,    "imu": 100.0},
        "tumvi":  {"camera": 20.0, "imu": 200.0},
    }[dataset]




def _resolve_dataset_root(config: dict, dataset: str) -> str:
    configured = Path(config["datasets"][dataset]["root"]).expanduser()
    if configured.exists():
        return str(configured)

    project_root = Path(__file__).resolve().parents[2]
    local_map = {"euroc": "euroc", "kitti": "kitti_raw", "tumvi": "tumvi"}
    local_root = project_root / "data" / local_map[dataset]
    if local_root.exists():
        print(f"[info] Using bundled data path for {dataset}: {local_root}")
        return str(local_root)

    return str(configured)

def _resolve_methods() -> dict[str, tuple[type, dict]]:
    return dict(PAPER_METHODS)


def _inject_runtime_kwargs(dataset: str, base_kwargs: dict, native_rates: dict) -> dict:
    """
    Fill runtime-dependent kwargs without mutating the source dict.
    """
    kwargs = dict(base_kwargs)

    if "native_rates" in kwargs:
        kwargs["native_rates"] = native_rates

    if "Q_diag" in kwargs:
        model_fn = get_model(dataset)
        _, Q, _, _, _ = model_fn()
        kwargs["Q_diag"] = np.diag(Q)

    return kwargs


def run_exp1(
    config: dict,
    results_dir: str = "results/exp1",
    filter_dataset: str | None = None,
    filter_sequence: str | None = None,
) -> pd.DataFrame:
    """
    Run Experiment 1 across methods, datasets, and sequences.

    Args:
        config: Loaded YAML config dict.
        results_dir: Directory for CSV outputs.
        filter_dataset: Restrict to one dataset ("euroc", "kitti", "tumvi").
        filter_sequence: Restrict to one sequence name.

    Returns:
        DataFrame with all results.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    methods = _resolve_methods()
    all_results = []

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
        if filter_sequence is not None:
            if filter_sequence not in sequences:
                print(
                    f"[warn] sequence '{filter_sequence}' not in {dataset} "
                    f"sequence list — skipping"
                )
                continue
            sequences = [filter_sequence]

        root = _resolve_dataset_root(config, dataset)
        native_rates = _native_rates_for(dataset)
        p_max_multiplier = float(config.get("experiments", {}).get("p_max_multiplier", 3.0))

        # First pass: always run fixed_high to define reference ESR / RR baseline
        fixed_high_results = {}
        for seq in sequences:
            loader_kw = {}
            if dataset == "kitti":
                date, drive = parse_kitti_sequence(seq)
                loader_kw = {"date": date, "drive": drive}

            result = run_single(
                scheduler_cls=FixedHighScheduler,
                scheduler_kwargs={},
                dataset=dataset,
                sequence=seq,
                dataset_root=root,
                method_name="fixed_high",
                p_max_multiplier=p_max_multiplier,
                **loader_kw,
            )
            fixed_high_results[seq] = result.esr
            all_results.append(result)

        # Second pass: all remaining selected methods
        for method_name, (cls, base_kwargs) in methods.items():
            if method_name == "fixed_high":
                continue

            kwargs = _inject_runtime_kwargs(dataset, base_kwargs, native_rates)

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
                    p_max_multiplier=p_max_multiplier,
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
        print(
            "[warn] No results collected — check dataset/sequence filters "
            "and that the dataset root path is correct."
        )
        return pd.DataFrame()

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
            row[f"esr_{sname}"] = r.esr.get(sname, 0)
            row[f"rr_{sname}"] = r.rate_reduction.get(sname, 0)

        primary_sensor = next((name for name in ("camera", "gps") if name in r.esr), None)
        row["esr_measurement"] = r.esr.get(primary_sensor, np.nan) if primary_sensor else np.nan
        row["rr_measurement"] = r.rate_reduction.get(primary_sensor, np.nan) if primary_sensor else np.nan
        row.setdefault("esr_camera", np.nan)
        row.setdefault("rr_camera", np.nan)
        row.setdefault("esr_gps", np.nan)
        row.setdefault("rr_gps", np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    output_name = "exp1_results.csv"
    out_path = Path(results_dir) / output_name
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    summary = summarise_results(all_results)
    print("\n--- Summary ---")
    for key, s in summary.items():
        print(
            f"{key:40s}  VR={s['vr_mean']:.3f}±{s['vr_std']:.3f}  "
            f"RMSE={s['rmse_mean']:.4f}±{s['rmse_std']:.4f}"
        )

    return df
