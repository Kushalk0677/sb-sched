from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import inspect

import numpy as np
import pandas as pd

from ..baselines.main import (
    DelayAwarePolicyScheduler,
    EventTriggeredScheduler,
    FixedHighScheduler,
    FixedLowScheduler,
    FixedMatchedScheduler,
    HeuristicAdaptiveScheduler,
)
from ..datasets.loaders import get_loader
from ..datasets.models import get_model
from ..kalman.kalman_filter import KalmanFilter
from ..scheduler.sb_sched import StalenessScheduler
from ..utils.metrics import (
    compute_area_over_threshold,
    compute_max_overshoot,
    compute_rate_reduction,
    compute_recovery_durations,
    compute_rmse,
    compute_violation_rate,
    pareto_frontier_mask,
)
from .runner import WARMUP_STEPS, _IMU_HZ, _estimate_steady_state_trace
from .exp2_6 import _resolve_dataset_root, _paper_p_max_multiplier


SEQUENCE = "MH_03_medium"
METHODS = [
    ("fixed_low", FixedLowScheduler),
    ("fixed_matched", FixedMatchedScheduler),
    ("heuristic", HeuristicAdaptiveScheduler),
    ("event_trigger", EventTriggeredScheduler),
    ("delay_aware", DelayAwarePolicyScheduler),
    ("sb_sched", StalenessScheduler),
]


def _build_kf(dataset: str):
    ds = dataset.lower()
    F, Q, sensors, x0, P0 = get_model(ds)()
    return KalmanFilter(F=F, Q=Q, sensors=sensors, x0=x0, P0=P0)


def _instantiate_scheduler(scheduler_cls, kf, p_max: float, sensor_names: list[str], scheduler_kwargs: dict):
    extra: dict = {}
    if "high_rate_hz" in inspect.signature(scheduler_cls.__init__).parameters:
        extra["high_rate_hz"] = _IMU_HZ.get(next(iter(["euroc" if "camera" in sensor_names else "kitti"])), 200.0)
    return scheduler_cls(kf=kf, p_max=p_max, sensor_names=sensor_names, **scheduler_kwargs, **extra)


def _scenario_rng(seed: int | None):
    return np.random.default_rng(0 if seed is None else seed)


def _scenario_drop_measurement(step: int, rng, scenario: dict) -> bool:
    if not scenario:
        return False
    if scenario.get("dropout_prob", 0.0) > 0.0 and rng.random() < float(scenario["dropout_prob"]):
        return True
    burst_every = int(scenario.get("burst_every_steps", 0) or 0)
    burst_len = int(scenario.get("burst_length_steps", 0) or 0)
    burst_start = int(scenario.get("burst_start_step", WARMUP_STEPS) or WARMUP_STEPS)
    if burst_every > 0 and burst_len > 0 and step >= burst_start:
        phase = (step - burst_start) % burst_every
        if phase < burst_len:
            return True
    return False


def _measurement_delay_steps(rng, scenario: dict) -> int:
    if not scenario:
        return 0
    base = int(scenario.get("fixed_delay_steps", 0) or 0)
    jitter = int(scenario.get("jitter_steps", 0) or 0)
    if jitter <= 0:
        return base
    return max(0, base + int(rng.integers(0, jitter + 1)))


def _run_single_detailed(
    scheduler_cls,
    scheduler_kwargs: dict,
    dataset: str,
    sequence: str,
    dataset_root: str,
    p_max_multiplier: float,
    method_name: str,
    scenario: dict | None = None,
    fixed_high_esr: dict | None = None,
    date: str | None = None,
    drive: str | None = None,
):
    scenario = scenario or {}
    rng = _scenario_rng(int(scenario.get("seed", 0)))
    imu_hz = _IMU_HZ.get(dataset.lower(), 200.0)

    kf = _build_kf(dataset)
    steady_state_trace = _estimate_steady_state_trace(kf, imu_hz=imu_hz)
    p_max = steady_state_trace * p_max_multiplier
    sensor_names = list(kf.sensors.keys())

    # Scheduler-specific kwargs needed by some baselines
    local_kwargs = dict(scheduler_kwargs)
    if scheduler_cls is FixedMatchedScheduler and "native_rates" not in local_kwargs:
        local_kwargs["native_rates"] = {n: kf.sensors[n].native_hz for n in sensor_names}
    if scheduler_cls is DelayAwarePolicyScheduler and "delays" not in local_kwargs:
        delay_steps = int(scenario.get("fixed_delay_steps", 0) or 0)
        local_kwargs["delays"] = {n: delay_steps for n in sensor_names}

    scheduler = _instantiate_scheduler(scheduler_cls, kf, p_max, sensor_names, local_kwargs)
    loader = get_loader(dataset.lower(), dataset_root, sequence, date=date, drive=drive)

    pending = defaultdict(dict)
    p_history = []
    eval_estimates = []
    eval_gt = []
    step_trigger_counts = []
    step_timestamps = []
    blackout_windows = []
    in_blackout = False
    blackout_start = None

    step = 0
    measurement_seen = {name: 0 for name in sensor_names}
    budget_window_steps = int(scenario.get("budget_window_steps", 0) or 0)
    max_updates_per_window = int(scenario.get("max_updates_per_window", -1))
    current_budget_window = -1
    updates_used_in_window = 0

    for reading in loader.stream():
        if reading.sensor_name != "imu":
            measurement_seen[reading.sensor_name] += 1
            dropped = _scenario_drop_measurement(step, rng, scenario)
            if dropped:
                if not in_blackout and int(scenario.get("burst_length_steps", 0) or 0) > 0:
                    in_blackout = True
                    blackout_start = step
            else:
                if in_blackout:
                    blackout_windows.append((blackout_start, step))
                    in_blackout = False
                    blackout_start = None
                delay_steps = _measurement_delay_steps(rng, scenario)

                    # FIX: ensure scheduler sees measurement at correct step
                delivery_step = step + 2 + delay_steps
                pending[delivery_step][reading.sensor_name] = reading.z

            if reading.ground_truth is not None:
                eval_estimates.append(kf.state.x[:3].copy())
                eval_gt.append(reading.ground_truth[:3].copy())
            continue

        # High-rate predict step
        kf.predict()
        step += 1
        step_timestamps.append(float(reading.timestamp))

        if budget_window_steps > 0:
            win = step // budget_window_steps
            if win != current_budget_window:
                current_budget_window = win
                updates_used_in_window = 0

        available = pending.pop(step, {})
        if budget_window_steps > 0 and max_updates_per_window >= 0 and updates_used_in_window >= max_updates_per_window:
            available = {}

        before = len(scheduler.trigger_log)
        scheduler.step(step, available)
        after = len(scheduler.trigger_log)
        n_updates = after - before
        updates_used_in_window += max(0, n_updates)

        p_history.append(kf.trace_P)
        step_trigger_counts.append(n_updates)

    if in_blackout:
        blackout_windows.append((blackout_start, step))

    estimates = np.array(eval_estimates) if eval_estimates else np.zeros((1, 3))
    gt_positions = np.array(eval_gt) if eval_gt else np.zeros((1, 3))
    dt = 1.0 / imu_hz
    esr = scheduler.effective_sample_rates(step, dt, warmup_steps=WARMUP_STEPS)
    vr = compute_violation_rate(p_history, p_max, warmup_steps=WARMUP_STEPS)
    rmse = compute_rmse(estimates, gt_positions)
    rr = compute_rate_reduction(esr, fixed_high_esr or esr)
    area = compute_area_over_threshold(p_history, p_max, warmup_steps=WARMUP_STEPS)
    max_overshoot = compute_max_overshoot(p_history, p_max, warmup_steps=WARMUP_STEPS)
    recovery = compute_recovery_durations(p_history, p_max, warmup_steps=WARMUP_STEPS, blackout_windows=blackout_windows)

    return {
        "method": method_name,
        "dataset": dataset,
        "sequence": sequence,
        "violation_rate": vr,
        "rmse_m": rmse,
        "esr": esr,
        "rate_reduction": rr,
        "p_history": p_history,
        "p_max": p_max,
        "step_trigger_counts": step_trigger_counts,
        "step_timestamps": step_timestamps,
        "area_over_threshold": area,
        "max_overshoot": max_overshoot,
        "recovery_steps_mean": float(np.mean(recovery)) if recovery else np.nan,
        "recovery_steps_max": float(np.max(recovery)) if recovery else np.nan,
        "blackout_windows": blackout_windows,
    }


def run_exp7_outage_robustness(config: dict, results_dir: str = "results/exp7") -> pd.DataFrame:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    root = _resolve_dataset_root(config, "euroc")
    pmax = _paper_p_max_multiplier(config)
    rows = []
    scenarios = [
        {"scenario": "random_drop_10", "dropout_prob": 0.10, "seed": 7},
        {"scenario": "random_drop_30", "dropout_prob": 0.30, "seed": 7},
        {"scenario": "random_drop_50", "dropout_prob": 0.50, "seed": 7},
        {"scenario": "burst_1s", "burst_every_steps": 1200, "burst_length_steps": 200, "burst_start_step": 700, "seed": 7},
        {"scenario": "burst_3s", "burst_every_steps": 1800, "burst_length_steps": 600, "burst_start_step": 700, "seed": 7},
    ]

    baseline = _run_single_detailed(FixedHighScheduler, {}, "euroc", SEQUENCE, root, pmax, "fixed_high")
    base_esr = baseline["esr"]

    for scenario in scenarios:
        for method_name, cls in METHODS:
            result = _run_single_detailed(cls, {}, "euroc", SEQUENCE, root, pmax, method_name, scenario=scenario, fixed_high_esr=base_esr)
            rows.append({
                "scenario": scenario["scenario"],
                "method": method_name,
                "vr": result["violation_rate"],
                "rmse_m": result["rmse_m"],
                "area_over_threshold": result["area_over_threshold"],
                "max_overshoot": result["max_overshoot"],
                "recovery_steps_mean": result["recovery_steps_mean"],
                "recovery_steps_max": result["recovery_steps_max"],
                **{f"esr_{k}": v for k, v in result["esr"].items()},
            })
    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/exp7_outage_robustness.csv", index=False)
    return df


def run_exp8_latency_jitter(config: dict, results_dir: str = "results/exp8") -> pd.DataFrame:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    root = _resolve_dataset_root(config, "euroc")
    pmax = _paper_p_max_multiplier(config)
    rows = []

    # Stronger stress cases than plain delay-only:
    # - much larger fixed delays
    # - delay+jitter combinations
    # - delay+dropout combinations to force meaningful constraint stress
    scenarios = [
        {"scenario": "delay_10", "fixed_delay_steps": 10, "seed": 11},
        {"scenario": "delay_20", "fixed_delay_steps": 20, "seed": 11},
        {"scenario": "delay_50", "fixed_delay_steps": 50, "seed": 11},
        {"scenario": "delay_10_jitter_10", "fixed_delay_steps": 10, "jitter_steps": 10, "seed": 11},
        {"scenario": "delay_20_jitter_20", "fixed_delay_steps": 20, "jitter_steps": 20, "seed": 11},
        {"scenario": "delay_10_drop_30", "fixed_delay_steps": 10, "dropout_prob": 0.30, "seed": 11},
        {"scenario": "delay_20_drop_30", "fixed_delay_steps": 20, "dropout_prob": 0.30, "seed": 11},
        {"scenario": "delay_20_drop_50", "fixed_delay_steps": 20, "dropout_prob": 0.50, "seed": 11},
        {
            "scenario": "delay_20_jitter_20_drop_30",
            "fixed_delay_steps": 20,
            "jitter_steps": 20,
            "dropout_prob": 0.30,
            "seed": 11,
        },
    ]

    baseline = _run_single_detailed(FixedHighScheduler, {}, "euroc", SEQUENCE, root, pmax, "fixed_high")
    base_esr = baseline["esr"]

    for scenario in scenarios:
        for method_name, cls in METHODS:
            result = _run_single_detailed(
                cls,
                {},
                "euroc",
                SEQUENCE,
                root,
                pmax,
                method_name,
                scenario=scenario,
                fixed_high_esr=base_esr,
            )
            rows.append({
                "scenario": scenario["scenario"],
                "method": method_name,
                "fixed_delay_steps": scenario.get("fixed_delay_steps", 0),
                "jitter_steps": scenario.get("jitter_steps", 0),
                "dropout_prob": scenario.get("dropout_prob", 0.0),
                "vr": result["violation_rate"],
                "rmse_m": result["rmse_m"],
                "area_over_threshold": result["area_over_threshold"],
                "max_overshoot": result["max_overshoot"],
                "recovery_steps_mean": result["recovery_steps_mean"],
                "recovery_steps_max": result["recovery_steps_max"],
                **{f"esr_{k}": v for k, v in result["esr"].items()},
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/exp8_latency_jitter.csv", index=False)
    return df

def run_exp9_resource_budget(config: dict, results_dir: str = "results/exp9") -> pd.DataFrame:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    root = _resolve_dataset_root(config, "euroc")
    pmax = _paper_p_max_multiplier(config)
    rows = []
    # 20Hz camera on 200Hz IMU => ~200 predict steps per second
    scenarios = [
        {"scenario": "budget_1_per_s", "budget_window_steps": 200, "max_updates_per_window": 1, "seed": 13},
        {"scenario": "budget_2_per_s", "budget_window_steps": 200, "max_updates_per_window": 2, "seed": 13},
        {"scenario": "budget_5_per_s", "budget_window_steps": 200, "max_updates_per_window": 5, "seed": 13},
        {"scenario": "budget_10_per_s", "budget_window_steps": 200, "max_updates_per_window": 10, "seed": 13},
    ]

    baseline = _run_single_detailed(FixedHighScheduler, {}, "euroc", SEQUENCE, root, pmax, "fixed_high")
    base_esr = baseline["esr"]

    for scenario in scenarios:
        for method_name, cls in METHODS:
            result = _run_single_detailed(cls, {}, "euroc", SEQUENCE, root, pmax, method_name, scenario=scenario, fixed_high_esr=base_esr)
            rows.append({
                "scenario": scenario["scenario"],
                "method": method_name,
                "budget_window_steps": scenario["budget_window_steps"],
                "max_updates_per_window": scenario["max_updates_per_window"],
                "vr": result["violation_rate"],
                "rmse_m": result["rmse_m"],
                "area_over_threshold": result["area_over_threshold"],
                "max_overshoot": result["max_overshoot"],
                **{f"esr_{k}": v for k, v in result["esr"].items()},
            })
    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/exp9_resource_budget.csv", index=False)
    return df


def _interpolate_positions(gt_t: np.ndarray, gt_pos: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.interp(query_t, gt_t, gt_pos[:, 0]),
        np.interp(query_t, gt_t, gt_pos[:, 1]),
        np.interp(query_t, gt_t, gt_pos[:, 2]),
    ])


def _motion_phase_labels(dataset_root: str, sequence: str, step_timestamps: np.ndarray):
    loader = get_loader("euroc", dataset_root, sequence)
    gt_t, gt_pos = loader.ground_truth_positions()
    pos = _interpolate_positions(gt_t, gt_pos, step_timestamps)
    dpos = np.diff(pos, axis=0, prepend=pos[:1])
    dt = np.diff(step_timestamps, prepend=step_timestamps[:1])
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.005
    speed = np.linalg.norm(dpos, axis=1) / dt
    q1, q2 = np.quantile(speed, [0.33, 0.66])
    labels = np.where(speed <= q1, "low", np.where(speed <= q2, "medium", "high"))
    return labels, speed


def run_exp10_motion_transitions(config: dict, results_dir: str = "results/exp10") -> pd.DataFrame:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    root = _resolve_dataset_root(config, "euroc")
    pmax = _paper_p_max_multiplier(config)
    rows = []
    window = 100

    for method_name, cls in METHODS:
        result = _run_single_detailed(cls, {}, "euroc", SEQUENCE, root, pmax, method_name)
        step_ts = np.array(result["step_timestamps"])
        if len(step_ts) == 0:
            continue
        labels, speed = _motion_phase_labels(root, SEQUENCE, step_ts)
        triggers = np.array(result["step_trigger_counts"])
        p_hist = np.array(result["p_history"])
        p_max = float(result["p_max"])

        for phase in ["low", "medium", "high"]:
            idx = labels == phase
            if not np.any(idx):
                continue
            rows.append({
                "method": method_name,
                "analysis": "phase_summary",
                "phase": phase,
                "vr": float(np.mean(p_hist[idx] > p_max)),
                "mean_trace_ratio": float(np.mean(p_hist[idx] / p_max)),
                "trigger_rate_hz": float(np.mean(triggers[idx]) * _IMU_HZ["euroc"]),
                "mean_speed": float(np.mean(speed[idx])),
                "lag_steps": np.nan,
                "post_max_overshoot": np.nan,
            })

        change_idx = np.where(labels[1:] != labels[:-1])[0] + 1
        for idx in change_idx:
            prev_phase, next_phase = labels[idx - 1], labels[idx]
            post_end = min(len(triggers), idx + window)
            lag = next((j - idx for j in range(idx, post_end) if triggers[j] > 0), np.nan)
            post_overshoot = float(np.max(np.maximum(p_hist[idx:post_end] - p_max, 0.0))) if post_end > idx else 0.0
            rows.append({
                "method": method_name,
                "analysis": "transition",
                "phase": f"{prev_phase}_to_{next_phase}",
                "vr": float(np.mean(p_hist[idx:post_end] > p_max)) if post_end > idx else 0.0,
                "mean_trace_ratio": float(np.mean(p_hist[idx:post_end] / p_max)) if post_end > idx else 0.0,
                "trigger_rate_hz": float(np.mean(triggers[idx:post_end]) * _IMU_HZ["euroc"]) if post_end > idx else 0.0,
                "mean_speed": float(np.mean(speed[idx:post_end])) if post_end > idx else 0.0,
                "lag_steps": lag,
                "post_max_overshoot": post_overshoot,
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/exp10_motion_transitions.csv", index=False)
    return df


def run_exp11_pareto(config: dict, results_dir: str = "results/exp11") -> pd.DataFrame:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    inputs = [
        (Path("results/exp7/exp7_outage_robustness.csv"), "outage"),
        (Path("results/exp8/exp8_latency_jitter.csv"), "latency"),
        (Path("results/exp9/exp9_resource_budget.csv"), "budget"),
    ]
    frames = []
    for path, family in inputs:
        if path.exists():
            df = pd.read_csv(path)
            df["family"] = family
            frames.append(df)
    if not frames:
        raise FileNotFoundError("Run Exp7-Exp9 first to generate Pareto inputs")

    df = pd.concat(frames, ignore_index=True)
    esr_col = next((c for c in df.columns if c.startswith("esr_")), None)
    if esr_col is None:
        raise ValueError("No esr_* column found in Pareto inputs")
    grouped = (
        df.groupby(["family", "method"], as_index=False)
        .agg({esr_col: "mean", "rmse_m": "mean", "area_over_threshold": "mean", "vr": "mean"})
        .rename(columns={esr_col: "esr_measurement"})
    )
    grouped["pareto_area"] = False
    grouped["pareto_rmse"] = False
    for family in grouped["family"].unique():
        sub_idx = grouped["family"] == family
        area_mask = pareto_frontier_mask(grouped.loc[sub_idx, ["esr_measurement", "area_over_threshold"]].to_numpy())
        rmse_mask = pareto_frontier_mask(grouped.loc[sub_idx, ["esr_measurement", "rmse_m"]].to_numpy())
        grouped.loc[sub_idx, "pareto_area"] = area_mask
        grouped.loc[sub_idx, "pareto_rmse"] = rmse_mask
    grouped.to_csv(f"{results_dir}/exp11_pareto.csv", index=False)
    return grouped
