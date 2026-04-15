"""
Combined implementations for Experiments 2–6.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..baselines.main import (
    EventTriggeredScheduler,
    FixedHighScheduler,
    FixedLowScheduler,
    FixedMatchedScheduler,
    HeuristicAdaptiveScheduler,
)
from ..datasets.loaders import get_loader
from ..kalman.ekf import ExtendedKalmanFilter
from ..kalman.kalman_filter import KalmanFilter, SensorModel
from ..scheduler.sb_sched import StalenessScheduler
from ..utils.metrics import RunResult, compute_rate_reduction, compute_rmse, compute_violation_rate
from .runner import WARMUP_STEPS, run_single


SEQUENCE = "MH_03_medium"


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


def _paper_p_max_multiplier(config: dict) -> float:
    return float(config.get("experiments", {}).get("p_max_multiplier", 3.0))


def _p_max_sweep(config: dict) -> list[float]:
    return list(config.get("experiments", {}).get("p_max_sweep", [1.5, 2.0, 3.0, 5.0, 10.0]))


def run_exp2(config: dict, results_dir: str = "results/exp2") -> pd.DataFrame:
    """Experiment 2: ESR and VR as P_max tightness varies."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    root = _resolve_dataset_root(config, "euroc")
    rows = []

    for mult in _p_max_sweep(config):
        for method_name, cls in [
            ("sb_sched", StalenessScheduler),
            ("heuristic", HeuristicAdaptiveScheduler),
        ]:
            result = run_single(
                scheduler_cls=cls,
                scheduler_kwargs={},
                dataset="euroc",
                sequence=SEQUENCE,
                dataset_root=root,
                p_max_multiplier=mult,
                method_name=method_name,
            )
            rows.append(
                {
                    "method": method_name,
                    "p_max_mult": mult,
                    "vr": result.violation_rate,
                    "rmse_m": result.rmse_m,
                    **{f"esr_{k}": v for k, v in result.esr.items()},
                }
            )
            print(
                f"[Exp2] {method_name} pmax_mult={mult}: "
                f"VR={result.violation_rate:.3f}, RMSE={result.rmse_m:.4f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/exp2_results.csv", index=False)
    return df


def run_exp3(config: dict, results_dir: str = "results/exp3") -> pd.DataFrame:
    """
    Experiment 3: performance under 10:1 vs 20:1 sensor rate asymmetry.

    The original draft incorrectly passed `subsample_factor` into
    FixedHighScheduler, which does not accept that keyword.  The real knob
    for asymmetry is data availability, not scheduler configuration, so we
    emulate 20:1 by thinning every second non-IMU measurement in the runner.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    root = _resolve_dataset_root(config, "euroc")
    p_max_multiplier = _paper_p_max_multiplier(config)

    for measurement_subsample_factor in [1, 2]:  # 1 = 10:1, 2 = 20:1
        rate_ratio = f"{10 * measurement_subsample_factor}:1"
        for method_name, cls in [
            ("fixed_high", FixedHighScheduler),
            ("heuristic", HeuristicAdaptiveScheduler),
            ("event_trigger", EventTriggeredScheduler),
            ("sb_sched", StalenessScheduler),
        ]:
            result = run_single(
                scheduler_cls=cls,
                scheduler_kwargs={},
                dataset="euroc",
                sequence=SEQUENCE,
                dataset_root=root,
                method_name=method_name,
                p_max_multiplier=p_max_multiplier,
                measurement_subsample_factor=measurement_subsample_factor,
            )
            rows.append(
                {
                    "method": method_name,
                    "rate_ratio": rate_ratio,
                    "measurement_subsample_factor": measurement_subsample_factor,
                    "vr": result.violation_rate,
                    "rmse_m": result.rmse_m,
                    **{f"esr_{k}": v for k, v in result.esr.items()},
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/exp3_results.csv", index=False)
    return df


def run_exp4(config: dict, results_dir: str = "results/exp4") -> pd.DataFrame:
    """Experiment 4: VR and RMSE vs motion intensity."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    p_max_multiplier = _paper_p_max_multiplier(config)

    # High-motion: EuRoC hard + TUM-VI corridors
    # Low-motion: EuRoC easy + TUM-VI rooms
    test_cases = [
        ("euroc", "MH_01_easy", "low", _resolve_dataset_root(config, "euroc")),
        ("euroc", "MH_03_medium", "medium", config["datasets"]["euroc"]["root"]),
        ("euroc", "MH_05_difficult", "high", config["datasets"]["euroc"]["root"]),
        ("tumvi", "room1", "low", _resolve_dataset_root(config, "tumvi")),
        ("tumvi", "corridor1", "high", _resolve_dataset_root(config, "tumvi")),
    ]

    for dataset, seq, motion_level, root in test_cases:
        for method_name, cls in [
            ("fixed_low", FixedLowScheduler),
            ("fixed_matched", FixedMatchedScheduler),
            ("heuristic", HeuristicAdaptiveScheduler),
            ("event_trigger", EventTriggeredScheduler),
            ("sb_sched", StalenessScheduler),
        ]:
            result = run_single(
                scheduler_cls=cls,
                scheduler_kwargs={},
                dataset=dataset,
                sequence=seq,
                dataset_root=root,
                method_name=method_name,
                p_max_multiplier=p_max_multiplier,
            )
            rows.append(
                {
                    "method": method_name,
                    "dataset": dataset,
                    "sequence": seq,
                    "motion_level": motion_level,
                    "vr": result.violation_rate,
                    "rmse_m": result.rmse_m,
                    **{f"esr_{k}": v for k, v in result.esr.items()},
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/exp4_results.csv", index=False)
    return df


def run_exp5(config: dict, results_dir: str = "results/exp5") -> pd.DataFrame:
    """
    Experiment 5: computational overhead of scheduler logic vs KF steps.

    This version avoids the earlier apples-to-oranges comparison between
    `SB-Sched compute_budget()` and `Heuristic step()` by reporting both idle
    decision cost and budget recomputation cost explicitly.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    n_warmup = 200
    n_measure = 20_000
    n = 9

    F = np.eye(n)
    Q = np.eye(n) * 0.01
    H = np.zeros((3, n))
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0
    R = np.eye(3) * 0.1
    sensor = SensorModel("gps", H, R, 10.0)
    z = np.zeros(3)
    p_max = 0.5

    rows = []

    def make_kf() -> KalmanFilter:
        return KalmanFilter(F, Q, [sensor], np.zeros(n), np.eye(n))

    def time_block(fn, n_reps: int) -> float:
        for _ in range(n_warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(n_reps):
            fn()
        return (time.perf_counter() - t0) / n_reps * 1e6

    kf_predict = make_kf()
    t_pred = time_block(lambda: kf_predict.predict(), n_measure)
    rows.append({"operation": "KF predict", "time_us": t_pred})

    kf_update = make_kf()
    t_upd = time_block(lambda: kf_update.update("gps", z), n_measure)
    rows.append({"operation": "KF update", "time_us": t_upd})

    kf_sb_idle = make_kf()
    sb_idle = StalenessScheduler(kf_sb_idle, p_max, ["gps"])
    sb_idle.schedule["gps"].trigger_step = 10**9
    t_sb_idle = time_block(lambda: sb_idle.step(0, {}), n_measure)
    rows.append({"operation": "SB-Sched idle step", "time_us": t_sb_idle})

    kf_sb_budget = make_kf()
    sb_budget = StalenessScheduler(kf_sb_budget, p_max, ["gps"])
    t_budget = time_block(lambda: sb_budget.compute_budget("gps"), n_measure)
    rows.append({"operation": "SB-Sched budget", "time_us": t_budget})

    kf_h_idle = make_kf()
    h_idle = HeuristicAdaptiveScheduler(kf_h_idle, p_max, ["gps"])
    t_h_idle = time_block(lambda: h_idle.step(0, {}), n_measure)
    rows.append({"operation": "Heuristic idle step", "time_us": t_h_idle})

    df = pd.DataFrame(rows)
    df["pct_of_predict"] = df["time_us"] / t_pred * 100.0
    df["pct_of_predict_plus_update"] = df["time_us"] / (t_pred + t_upd) * 100.0
    df.to_csv(f"{results_dir}/exp5_results.csv", index=False)
    print(df.to_string(index=False))
    return df


def run_exp6(config: dict, results_dir: str = "results/exp6") -> pd.DataFrame:
    """
    Experiment 6: generalisation to EKF using a lightweight nonlinear motion
    model over the same EuRoC observation stream.

    Important scope note:
    this is a true EKF experiment, but it is still a simplified proxy for the
    paper's full IMU-preintegration EKF described in docs/ekf_extension.md.
    It validates that SB-Sched can operate on EKF covariance predictions and
    remain conservative under mild nonlinearity.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    root = _resolve_dataset_root(config, "euroc")

    sequences = [
        "MH_01_easy",
        "MH_02_easy",
        "MH_03_medium",
        "MH_04_medium",
        "MH_05_difficult",
    ]
    rows = []

    for sequence in sequences:
        fixed_high_result = _run_single_ekf(
            scheduler_cls=FixedHighScheduler,
            scheduler_kwargs={},
            dataset_root=root,
            sequence=sequence,
            method_name="fixed_high",
        )
        rows.append(_ekf_row(fixed_high_result))

        baseline_esr = fixed_high_result.esr
        for method_name, cls in [
            ("heuristic", HeuristicAdaptiveScheduler),
            ("sb_sched", StalenessScheduler),
        ]:
            result = _run_single_ekf(
                scheduler_cls=cls,
                scheduler_kwargs={},
                dataset_root=root,
                sequence=sequence,
                method_name=method_name,
                fixed_high_esr=baseline_esr,
            )
            rows.append(_ekf_row(result))

    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/exp6_results.csv", index=False)
    print("[Exp6] EKF results written to", f"{results_dir}/exp6_results.csv")
    return df


# ---------------------------------------------------------------------------
# EKF helpers
# ---------------------------------------------------------------------------


def _ekf_row(result: RunResult) -> dict:
    return {
        "method": result.method,
        "dataset": result.dataset,
        "sequence": result.sequence,
        "vr": result.violation_rate,
        "rmse_m": result.rmse_m,
        "rate_reduction": result.rate_reduction,
        **{f"esr_{k}": v for k, v in result.esr.items()},
    }


def _build_position_ekf(imu_hz: float = 200.0, meas_hz: float = 20.0):
    dt = 1.0 / imu_hz
    n = 6
    alpha = 0.15

    def f(x: np.ndarray) -> np.ndarray:
        x_next = x.copy()
        vel = x[3:6]
        accel = alpha * np.sin(vel)
        x_next[0:3] = x[0:3] + vel * dt + 0.5 * accel * (dt ** 2)
        x_next[3:6] = vel + accel * dt
        return x_next

    def jac_f(x: np.ndarray) -> np.ndarray:
        F = np.eye(n)
        vel = x[3:6]
        dacc_dv = alpha * np.cos(vel)
        for i in range(3):
            F[i, 3 + i] = dt + 0.5 * dacc_dv[i] * (dt ** 2)
            F[3 + i, 3 + i] = 1.0 + dacc_dv[i] * dt
        return F

    def h_pos(x: np.ndarray) -> np.ndarray:
        return x[:3]

    def jac_h_pos(_x: np.ndarray) -> np.ndarray:
        H = np.zeros((3, n))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        return H

    q_pos = 0.001
    q_vel = 0.010
    Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]) * dt
    R = np.eye(3) * 0.01
    sensor = SensorModel(name="camera", H=np.eye(3, n), R=R, native_hz=meas_hz)
    x0 = np.zeros(n)
    P0 = np.eye(n) * 0.003

    ekf = ExtendedKalmanFilter(
        f=f,
        jac_f=jac_f,
        Q=Q,
        sensors=[(sensor, h_pos, jac_h_pos)],
        x0=x0,
        P0=P0,
    )
    return ekf, [sensor]


def _estimate_ekf_steady_state_trace(ekf: ExtendedKalmanFilter, sensor_name: str, steps: int = 5000) -> float:
    P = ekf.state.P.copy()
    x = ekf.state.x.copy()
    sensor_model, _h, jac_h = ekf.sensor_map[sensor_name]
    I = np.eye(ekf.n)

    for _ in range(steps):
        F_k = ekf.jac_f(x)
        x = ekf.f(x)
        P = F_k @ P @ F_k.T + ekf.Q
        H_k = jac_h(x)
        S = H_k @ P @ H_k.T + sensor_model.R
        K = P @ H_k.T @ np.linalg.inv(S)
        P = (I - K @ H_k) @ P @ (I - K @ H_k).T + K @ sensor_model.R @ K.T
    return float(np.trace(P))


def _run_single_ekf(
    scheduler_cls,
    scheduler_kwargs: dict,
    dataset_root: str,
    sequence: str,
    method_name: str,
    p_max_multiplier: float = 3.0,
    fixed_high_esr: dict | None = None,
) -> RunResult:
    imu_hz = 200.0
    ekf, sensors = _build_position_ekf(imu_hz=imu_hz, meas_hz=20.0)
    p_max = _estimate_ekf_steady_state_trace(ekf, "camera") * p_max_multiplier
    sensor_names = [s.name for s in sensors]

    scheduler = scheduler_cls(
        kf=ekf,
        p_max=p_max,
        sensor_names=sensor_names,
        **scheduler_kwargs,
    )

    loader = get_loader("euroc", dataset_root, sequence)
    p_history: list[float] = []
    eval_estimates: list[np.ndarray] = []
    eval_gt: list[np.ndarray] = []
    step = 0

    for reading in loader.stream():
        if reading.sensor_name == "imu":
            ekf.predict()
            step += 1
            p_history.append(ekf.trace_P)
        else:
            scheduler.step(step, {reading.sensor_name: reading.z})

        if reading.ground_truth is not None:
            eval_estimates.append(ekf.state.x[:3].copy())
            eval_gt.append(reading.ground_truth[:3].copy())

    if eval_estimates:
        estimates = np.array(eval_estimates)
        gt_positions = np.array(eval_gt)
    else:
        estimates = gt_positions = np.zeros((1, 3))

    dt = 1.0 / imu_hz
    esr = scheduler.effective_sample_rates(step, dt, warmup_steps=WARMUP_STEPS)
    vr = compute_violation_rate(p_history, p_max, warmup_steps=WARMUP_STEPS)
    rmse = compute_rmse(estimates, gt_positions)
    pos_errors = list(np.linalg.norm(estimates - gt_positions, axis=1))
    rr = compute_rate_reduction(esr, fixed_high_esr or esr)

    return RunResult(
        method=method_name,
        dataset="euroc_ekf",
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
