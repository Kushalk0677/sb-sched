"""Realism benchmark for EuRoC and KITTI.

This experiment suite keeps the repository's KF + scheduler machinery but
replaces the clean low-rate measurements with more realistic ones. It outputs
both per-method metrics and the paper-ready per-setting summary table.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable
import inspect

import numpy as np
import pandas as pd

from ..baselines.main import FixedHighScheduler, EventTriggeredScheduler, ClairvoyantLookaheadScheduler, VarianceThresholdScheduler
from ..datasets.loaders import get_loader
from ..datasets.models import euroc_model, kitti_model, gas_signal_model
from ..datasets.realistic_measurements import (
    default_euroc_profiles,
    default_kitti_profiles,
    default_gas_profiles,
    make_euroc_realistic_loader,
    make_kitti_realistic_loader,
    RealismProfile,
    RealisticMeasurementStream,
)
from ..datasets.gas_loaders import OneMonthGasLoader, UCIGasDriftLoader
from ..experiments.exp1_baseline_comparison import parse_kitti_sequence
from ..experiments.runner import WARMUP_STEPS, _IMU_HZ, _estimate_steady_state_trace
from ..kalman.kalman_filter import KalmanFilter
from ..scheduler.sb_sched import StalenessScheduler
from ..utils.metrics import (
    compute_area_over_threshold,
    compute_rate_reduction,
    compute_rmse,
    compute_violation_rate,
)


DEFAULT_METHODS: dict[str, type] = {
    "fixed_high": FixedHighScheduler,
    "event_trigger": EventTriggeredScheduler,
    "var_threshold": VarianceThresholdScheduler,
    "sb_sched": StalenessScheduler,
    "clairvoyant_lookahead": ClairvoyantLookaheadScheduler,
}


def _build_realistic_kf(dataset: str, profile: RealismProfile) -> KalmanFilter:
    dataset = dataset.lower()
    if dataset == "euroc":
        F, Q, sensors, x0, P0 = euroc_model(imu_hz=200.0, cam_hz=20.0)
        sensors[0].R = np.eye(3) * max(profile.measurement_noise_std_m, 0.10) ** 2
    elif dataset == "kitti":
        F, Q, sensors, x0, P0 = kitti_model(imu_hz=10.0, gps_hz=10.0)
        sensors[0].R = np.eye(3) * max(profile.measurement_noise_std_m, 0.10) ** 2
    elif dataset in {"uci_gas", "onemonth_gas"}:
        F, Q, sensors, x0, P0 = gas_signal_model(sample_hz=1.0, measurement_hz=1.0)
        sensors[0].R = np.array([[max(profile.measurement_noise_std_m, 0.05) ** 2]], dtype=float)
    else:
        raise ValueError(f"Unsupported dataset for realism benchmark: {dataset}")
    return KalmanFilter(F=F, Q=Q, sensors=sensors, x0=x0, P0=P0)


def _resolve_root(config: dict, dataset: str) -> str:
    root = config["datasets"][dataset]["root"]
    return str(Path(root))


def _make_loader(
    dataset: str,
    config: dict,
    sequence: str,
    profile: RealismProfile,
    precomputed_csv: str | None = None,
    sensor_id: int = 0,
    feature: str = 'rs_r0_ratio',
    uci_feature_index: int = 2,
):
    dataset = dataset.lower()
    root = _resolve_root(config, dataset)
    if dataset == 'euroc':
        return make_euroc_realistic_loader(root, sequence, profile, precomputed_trajectory_csv=precomputed_csv)
    if dataset == 'kitti':
        date, drive = parse_kitti_sequence(sequence)
        return make_kitti_realistic_loader(root, date, drive, profile)
    if dataset == 'onemonth_gas':
        base = OneMonthGasLoader(root, sensor_id=sensor_id, feature=feature)
        return RealisticMeasurementStream(base, measurement_sensor='gas', native_hz=base.metadata.native_hz, profile=profile)
    if dataset == 'uci_gas':
        base = UCIGasDriftLoader(root, batch=sequence, feature_index=uci_feature_index, dt_s=1.0)
        return RealisticMeasurementStream(base, measurement_sensor='gas', native_hz=base.metadata.native_hz, profile=profile)
    raise ValueError(f"Unsupported dataset for realism benchmark: {dataset}")


def _scheduler_kwargs(method_name: str, config: dict) -> dict:
    if method_name == "event_trigger":
        return {
            "innovation_threshold": float(config.get("baselines", {}).get("event_trigger", {}).get("innovation_threshold", 3.0))
        }
    return {}


def _run_single_realistic(
    dataset: str,
    sequence: str,
    profile: RealismProfile,
    scheduler_cls,
    scheduler_name: str,
    config: dict,
    precomputed_csv: str | None = None,
    sensor_id: int = 0,
    feature: str = "rs_r0_ratio",
    uci_feature_index: int = 1,
):
    imu_hz = _IMU_HZ.get(dataset.lower(), 1.0)
    kf = _build_realistic_kf(dataset, profile)
    steady_state_trace = _estimate_steady_state_trace(kf, imu_hz=imu_hz)
    p_max_multiplier = float(config.get("experiments", {}).get("p_max_multiplier", 3.0))
    p_max = steady_state_trace * p_max_multiplier

    sensor_names = list(kf.sensors.keys())
    kwargs = _scheduler_kwargs(scheduler_name, config)
    extra = {}
    if "high_rate_hz" in inspect.signature(scheduler_cls.__init__).parameters:
        extra["high_rate_hz"] = imu_hz
    scheduler = scheduler_cls(kf=kf, p_max=p_max, sensor_names=sensor_names, **kwargs, **extra)

    loader = _make_loader(dataset, config, sequence, profile, precomputed_csv=precomputed_csv, sensor_id=sensor_id, feature=feature, uci_feature_index=uci_feature_index)
    run_warmup = min(WARMUP_STEPS, max(25, int(getattr(loader.metadata, "n_samples", WARMUP_STEPS) * 0.1))) if dataset in {"uci_gas", "onemonth_gas"} else WARMUP_STEPS
    if hasattr(scheduler, "warmup_steps"):
        scheduler.warmup_steps = run_warmup
    p_history: list[float] = []
    eval_estimates: list[np.ndarray] = []
    eval_gt: list[np.ndarray] = []
    step = 0

    for reading in loader.stream():
        if reading.sensor_name == "imu":
            kf.predict()
            step += 1
            p_history.append(kf.trace_P)
        else:
            scheduler.step(step, {reading.sensor_name: reading.z})

        if reading.ground_truth is not None:
            gt_vec = np.asarray(reading.ground_truth, dtype=float).reshape(-1)
            eval_estimates.append(kf.state.x[: gt_vec.size].copy())
            eval_gt.append(gt_vec.copy())

    estimates = np.array(eval_estimates) if eval_estimates else np.zeros((1, 3))
    gt = np.array(eval_gt) if eval_gt else np.zeros((1, 3))
    dt = 1.0 / imu_hz
    esr = scheduler.effective_sample_rates(step, dt, warmup_steps=run_warmup)

    result = {
        "dataset": dataset,
        "sequence": sequence,
        "scheduler": scheduler_name,
        "setting": profile.name,
        "vr": compute_violation_rate(p_history, p_max, warmup_steps=run_warmup),
        "aot": compute_area_over_threshold(p_history, p_max, warmup_steps=run_warmup),
        "rmse_m": compute_rmse(estimates, gt),
        "p_max": p_max,
        **{f"esr_{k}": v for k, v in esr.items()},
        "esr_primary": next(iter(esr.values())) if esr else 0.0,
        "n_eval_points": len(eval_gt),
        "n_predict_steps": step,
        "profile_description": profile.description,
        "measurement_noise_std_m": profile.measurement_noise_std_m,
        "dropout_prob": profile.dropout_prob,
        "fixed_delay_events": profile.fixed_delay_events,
        "jitter_delay_events": profile.jitter_delay_events,
        "outlier_prob": profile.outlier_prob,
        "outlier_std_m": profile.outlier_std_m,
        "bias_walk_std_m": profile.bias_walk_std_m,
        "external_measurement_source": getattr(loader.metadata, "external_source", None),
        "delivered_measurements": loader.metadata.delivered_measurement_count,
        "dropped_measurements": loader.metadata.dropped_measurement_count,
        "warmup_steps": run_warmup,
    }
    return result


def run_realism_benchmark(
    config: dict,
    dataset: str,
    sequence: str,
    precomputed_csv: str | None = None,
    methods: dict[str, type] | None = None,
    results_dir: str = "results/realism",
    sensor_id: int = 0,
    feature: str = "rs_r0_ratio",
    uci_feature_index: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the realistic-measurement benchmark and return two dataframes.

    Returns
    -------
    full_df
        Per-method, per-setting metrics.
    summary_df
        Paper-style summary for SB-Sched only with columns Setting / VR / AOT / RMSE / ESR.
    """
    dataset = dataset.lower()
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    methods = DEFAULT_METHODS if methods is None else methods

    if dataset == "euroc":
        profiles = default_euroc_profiles(seed=int(config.get("experiments", {}).get("random_seed", 42)))
    elif dataset == "kitti":
        profiles = default_kitti_profiles(seed=int(config.get("experiments", {}).get("random_seed", 42)))
    elif dataset in {"uci_gas", "onemonth_gas"}:
        profiles = default_gas_profiles(seed=int(config.get("experiments", {}).get("random_seed", 42)))
    else:
        raise ValueError("Supported datasets are 'euroc', 'kitti', 'uci_gas', and 'onemonth_gas'.")

    rows = []
    for profile in profiles.values():
        for method_name, cls in methods.items():
            rows.append(
                _run_single_realistic(
                    dataset=dataset,
                    sequence=sequence,
                    profile=profile,
                    scheduler_cls=cls,
                    scheduler_name=method_name,
                    config=config,
                    precomputed_csv=precomputed_csv,
                    sensor_id=sensor_id,
                    feature=feature,
                    uci_feature_index=uci_feature_index,
                )
            )

    full_df = pd.DataFrame(rows)
    tag = f"{dataset}_{sequence}"
    if precomputed_csv:
        full_df["external_measurement_source"] = str(precomputed_csv)

    full_path = Path(results_dir) / f"{tag}_realism_full.csv"
    full_df.to_csv(full_path, index=False)

    sb = full_df[full_df["scheduler"] == "sb_sched"].copy()
    display_names = {
        "oracle": "Oracle (current)",
        "noisy": "Noisy",
        "noisy_dropout": "Noisy + dropout",
        "noisy_delay": "Noisy + delay",
        "gps_raw": "GPS raw (current)",
        "gps_noisy": "GPS + noise/drift",
        "gps_dropout": "GPS + dropout",
        "gps_delay": "GPS + delay",
        "gas_raw": "Gas raw (current)",
        "gas_dropout": "Gas + dropout",
        "gas_delay": "Gas + delay",
        "gas_dropout_delay": "Gas + dropout + delay",
    }
    sb["Setting"] = sb["setting"].map(lambda s: display_names.get(s, s))
    summary_df = sb[["Setting", "vr", "aot", "rmse_m", "esr_primary", "delivered_measurements", "dropped_measurements"]].rename(
        columns={
            "vr": "VR",
            "aot": "AOT",
            "rmse_m": "RMSE",
            "esr_primary": "ESR",
            "delivered_measurements": "DeliveredMeasurements",
            "dropped_measurements": "DroppedMeasurements",
        }
    )
    summary_path = Path(results_dir) / f"{tag}_realism_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return full_df, summary_df