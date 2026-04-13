"""
src/experiments/runner.py

Generic experiment runner: loads a dataset, runs a scheduler,
collects metrics. Used by all 6 experiment scripts.
"""

import numpy as np
import time
from tqdm import tqdm

from ..kalman.kalman_filter import KalmanFilter
from ..datasets.loaders import get_loader
from ..datasets.models import get_model
from ..utils.metrics import RunResult, compute_rmse, compute_violation_rate, \
    compute_rate_reduction


def run_single(
    scheduler_cls,
    scheduler_kwargs: dict,
    dataset: str,
    sequence: str,
    dataset_root: str,
    p_max_multiplier: float = 3.0,
    method_name: str = "method",
    fixed_high_esr: dict = None,
    date: str = None,
    drive: str = None,
) -> RunResult:
    """
    Run one scheduler on one dataset sequence and return metrics.

    Args:
        scheduler_cls:    Class of the scheduler to instantiate
        scheduler_kwargs: Extra kwargs beyond (kf, p_max, sensor_names)
        dataset:          "euroc" | "kitti" | "tumvi"
        sequence:         Sequence name string
        dataset_root:     Root path to dataset
        p_max_multiplier: P_max = p_max_multiplier * steady_state_trace(P)
        method_name:      String label for this run
        fixed_high_esr:   ESR of Fixed-High run (for rate reduction calc)
        date, drive:      KITTI-specific identifiers
    Returns:
        RunResult with all metrics
    """
    # Build model
    model_fn = get_model(dataset)
    if dataset.lower() == "kitti":
        F, Q, sensors, x0, P0 = model_fn()
    else:
        F, Q, sensors, x0, P0 = model_fn()

    # Initialise Kalman filter
    kf = KalmanFilter(F=F, Q=Q, sensors=sensors, x0=x0, P0=P0)

    # Compute P_max from steady-state covariance of fully-updated filter
    p_max = _estimate_steady_state_trace(kf) * p_max_multiplier

    # Build scheduler
    sensor_names = [s.name for s in sensors]
    scheduler = scheduler_cls(
        kf=kf,
        p_max=p_max,
        sensor_names=sensor_names,
        **scheduler_kwargs,
    )

    # Load data
    loader = get_loader(dataset, dataset_root, sequence, date=date, drive=drive)

    # Run
    p_history = []
    estimates = []
    gt_positions = []
    step = 0

    for reading in tqdm(loader.stream(), desc=f"{method_name}/{sequence}",
                        leave=False):
        # Predict step (always runs at high rate — the IMU drives prediction)
        if reading.sensor_name == "imu":
            kf.predict()
            step += 1
            p_history.append(kf.trace_P)

        # Measurement step (scheduler decides whether to update)
        measurements = {}
        if reading.sensor_name != "imu":
            measurements[reading.sensor_name] = reading.z

        scheduler.step(step, measurements)

        # Collect position estimate and GT
        pos_est = kf.state.x[:3]
        estimates.append(pos_est.copy())
        if reading.ground_truth is not None:
            gt_positions.append(reading.ground_truth[:3].copy())

    estimates = np.array(estimates)
    gt_positions = np.array(gt_positions) if gt_positions else estimates

    # Align lengths
    min_len = min(len(estimates), len(gt_positions))
    estimates = estimates[:min_len]
    gt_positions = gt_positions[:min_len]

    # Metrics
    native_rates = {s.name: s.native_hz for s in sensors}
    imu_hz = 200.0 if dataset != "kitti" else 100.0
    dt = 1.0 / imu_hz
    esr = scheduler.effective_sample_rates(step, dt)
    vr = compute_violation_rate(p_history, p_max)
    rmse = compute_rmse(estimates, gt_positions)
    rr = compute_rate_reduction(esr, fixed_high_esr or esr)

    return RunResult(
        method=method_name,
        dataset=dataset,
        sequence=sequence,
        violation_rate=vr,
        rmse_m=rmse,
        esr=esr,
        rate_reduction=rr,
        p_history=p_history,
        pos_errors=list(np.linalg.norm(estimates - gt_positions, axis=1)),
        total_steps=step,
        p_max=p_max,
    )


def _estimate_steady_state_trace(kf: KalmanFilter, steps: int = 2000) -> float:
    """
    Run KF with all sensors updating at native rate until P converges.
    Returns steady-state trace(P).
    """
    P = kf.state.P.copy()
    for _ in range(steps):
        P = kf.F @ P @ kf.F.T + kf.Q
        for sensor in kf.sensors.values():
            H, R = sensor.H, sensor.R
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            P = (np.eye(kf.n) - K @ H) @ P
    return float(np.trace(P))
