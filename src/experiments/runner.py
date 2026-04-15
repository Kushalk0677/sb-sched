"""
src/experiments/runner.py

Generic experiment runner: loads a dataset, runs a scheduler,
collects metrics. Used by all 6 experiment scripts.
"""

import inspect
import numpy as np
from tqdm import tqdm

from ..kalman.kalman_filter import KalmanFilter
from ..datasets.loaders import get_loader
from ..datasets.models import get_model
from ..utils.metrics import (
    RunResult,
    compute_rmse,
    compute_violation_rate,
    compute_rate_reduction,
)

# IMU (predict-step) rate for each dataset.
# KITTI: pykitti exposes OXTS packets at ~10 Hz; there is no separate
#        high-rate IMU stream, so the KF predicts at 10 Hz, not 100 Hz.
_IMU_HZ: dict[str, float] = {
    "euroc": 200.0,
    "kitti":  10.0,   # OXTS packet rate — see KITTILoader docstring
    "tumvi": 200.0,
}

# Steps to exclude from VR calculation (initial convergence phase).
WARMUP_STEPS = 500


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
    measurement_subsample_factor: int = 1,
) -> RunResult:
    """
    Run one scheduler on one dataset sequence and return metrics.

    Args:
        scheduler_cls:    Scheduler class to instantiate
        scheduler_kwargs: Extra kwargs beyond (kf, p_max, sensor_names)
        dataset:          "euroc" | "kitti" | "tumvi"
        sequence:         Sequence name string
        dataset_root:     Root path to dataset on disk
        p_max_multiplier: P_max = multiplier × steady-state trace(P)
        method_name:      Label string for this run
        fixed_high_esr:   ESR dict from Fixed-High run (for RR calculation)
        date, drive:      KITTI-specific sequence identifiers
        measurement_subsample_factor:
                         Keep only every Nth non-IMU measurement event.
                         Used by Exp 3 to emulate larger sensor-rate asymmetry
                         while leaving the IMU predict stream unchanged.
    Returns:
        RunResult with all metrics
    """
    if measurement_subsample_factor < 1:
        raise ValueError("measurement_subsample_factor must be >= 1")

    ds = dataset.lower()
    imu_hz = _IMU_HZ.get(ds, 200.0)

    # Build model and KF
    model_fn = get_model(ds)
    F, Q, sensors, x0, P0 = model_fn()
    kf = KalmanFilter(F=F, Q=Q, sensors=sensors, x0=x0, P0=P0)

    # Derive P_max from a realistic steady-state covariance under native-rate updates
    steady_state_trace = _estimate_steady_state_trace(kf, imu_hz=imu_hz)
    p_max = steady_state_trace * p_max_multiplier

    # Build scheduler — inject high_rate_hz for schedulers that need it
    sensor_names = [s.name for s in sensors]
    extra: dict = {}
    if "high_rate_hz" in inspect.signature(scheduler_cls.__init__).parameters:
        extra["high_rate_hz"] = imu_hz

    scheduler = scheduler_cls(
        kf=kf,
        p_max=p_max,
        sensor_names=sensor_names,
        **scheduler_kwargs,
        **extra,
    )

    # Load data stream
    loader = get_loader(ds, dataset_root, sequence, date=date, drive=drive)


    # ── Main loop ────────────────────────────────────────────────────────
    p_history: list[float] = []
    eval_estimates: list[np.ndarray] = []
    eval_gt: list[np.ndarray] = []
    step = 0
    measurement_seen = {name: 0 for name in sensor_names}

    for reading in tqdm(loader.stream(), desc=f"{method_name}/{sequence}",
                        leave=False):

        if reading.sensor_name == "imu":
            # IMU drives the predict step
            kf.predict()
            step += 1
            p_history.append(kf.trace_P)

        else:
            # Optional measurement thinning for asymmetry experiments.
            measurement_seen[reading.sensor_name] += 1
            if measurement_seen[reading.sensor_name] % measurement_subsample_factor != 0:
                continue

            # Non-IMU sensor: offer measurement to scheduler
            scheduler.step(step, {reading.sensor_name: reading.z})

        # Collect evaluation pair aligned at every GT-tagged event.
        # Both arrays grow together so estimates[i] ↔ gt[i] by construction.
        if reading.ground_truth is not None:
            eval_estimates.append(kf.state.x[:3].copy())
            eval_gt.append(reading.ground_truth[:3].copy())

    # ── Metrics ──────────────────────────────────────────────────────────
    if eval_estimates:
        estimates    = np.array(eval_estimates)
        gt_positions = np.array(eval_gt)
    else:
        # No GT events — degenerate fallback (shouldn't happen with real data)
        estimates = gt_positions = np.zeros((1, 3))

    dt  = 1.0 / imu_hz
    esr = scheduler.effective_sample_rates(step, dt, warmup_steps=WARMUP_STEPS)
    vr  = compute_violation_rate(p_history, p_max, warmup_steps=WARMUP_STEPS)
    rmse = compute_rmse(estimates, gt_positions)
    pos_errors = list(np.linalg.norm(estimates - gt_positions, axis=1))
    rr  = compute_rate_reduction(esr, fixed_high_esr or esr)

    return RunResult(
        method=method_name,
        dataset=dataset,
        sequence=sequence,
        violation_rate=vr,
        rmse_m=rmse,
        esr=esr,
        rate_reduction=rr,
        p_history=p_history,
        pos_errors=pos_errors,
        total_steps=step,
        p_max=p_max,
    )


def _estimate_steady_state_trace(
    kf: KalmanFilter,
    imu_hz: float,
    steps: int = 8000,
    tail_steps: int = 1000,
) -> float:
    """
    Estimate a realistic steady-state trace(P) using each sensor's native update
    rate rather than assuming every sensor updates on every predict step.

    We return the peak trace(P) over the converged tail window to obtain a
    conservative reference for P_max on hard sequences.
    """
    P = kf.state.P.copy()
    I = np.eye(kf.n)

    sensor_periods = {}
    for name, sensor in kf.sensors.items():
        native_hz = float(getattr(sensor, "native_hz", imu_hz) or imu_hz)
        native_hz = min(native_hz, imu_hz)
        sensor_periods[name] = max(1, int(round(imu_hz / native_hz)))

    tail = []
    for step in range(steps):
        P = kf.F @ P @ kf.F.T + kf.Q
        for name, sensor in kf.sensors.items():
            if step % sensor_periods[name] != 0:
                continue
            H, R = sensor.H, sensor.R
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            I_KH = I - K @ H
            P = I_KH @ P @ I_KH.T + K @ R @ K.T

        if step >= steps - tail_steps:
            tail.append(float(np.trace(P)))

    if not tail:
        return float(np.trace(P))
    return float(max(tail))
