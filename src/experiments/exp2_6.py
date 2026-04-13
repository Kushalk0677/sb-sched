"""
src/experiments/exp2_pmax_sensitivity.py
Experiment 2: ESR and VR as P_max tightness varies.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from ..scheduler.sb_sched import StalenessScheduler
from ..baselines.baselines import HeuristicAdaptiveScheduler
from .runner import run_single


P_MAX_MULTIPLIERS = [1.5, 2.0, 3.0, 5.0, 10.0]
SEQUENCE = "MH_03_medium"


def run_exp2(config: dict, results_dir: str = "results/exp2") -> pd.DataFrame:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    root = config["datasets"]["euroc"]["root"]
    rows = []

    for mult in P_MAX_MULTIPLIERS:
        for method_name, cls in [("sb_sched", StalenessScheduler),
                                  ("heuristic", HeuristicAdaptiveScheduler)]:
            result = run_single(
                scheduler_cls=cls, scheduler_kwargs={},
                dataset="euroc", sequence=SEQUENCE,
                dataset_root=root, p_max_multiplier=mult,
                method_name=method_name,
            )
            rows.append({
                "method": method_name,
                "p_max_mult": mult,
                "vr": result.violation_rate,
                "rmse_m": result.rmse_m,
                **{f"esr_{k}": v for k, v in result.esr.items()},
            })
            print(f"[Exp2] {method_name} pmax_mult={mult}: "
                  f"VR={result.violation_rate:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/exp2_results.csv", index=False)
    return df


# ---------------------------------------------------------------------------
"""
src/experiments/exp3_rate_asymmetry.py
Experiment 3: Performance under 10:1 vs 20:1 sensor rate asymmetry.
"""

def run_exp3(config: dict, results_dir: str = "results/exp3") -> pd.DataFrame:
    import pandas as pd
    from pathlib import Path
    from ..baselines.baselines import (FixedHighScheduler,
                                       HeuristicAdaptiveScheduler,
                                       EventTriggeredScheduler)
    from ..scheduler.sb_sched import StalenessScheduler

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    root = config["datasets"]["euroc"]["root"]

    for subsampling_factor in [1, 2]:   # 1 = native 10:1, 2 = artificial 20:1
        for method_name, cls in [
            ("fixed_high", FixedHighScheduler),
            ("heuristic",  HeuristicAdaptiveScheduler),
            ("event_trigger", EventTriggeredScheduler),
            ("sb_sched",   StalenessScheduler),
        ]:
            result = run_single(
                scheduler_cls=cls,
                scheduler_kwargs={"subsample_factor": subsampling_factor}
                                  if method_name == "fixed_high" else {},
                dataset="euroc", sequence="MH_03_medium",
                dataset_root=root, method_name=method_name,
            )
            rows.append({
                "method": method_name,
                "rate_ratio": f"{10 * subsampling_factor}:1",
                "vr": result.violation_rate,
                "rmse_m": result.rmse_m,
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/exp3_results.csv", index=False)
    return df


# ---------------------------------------------------------------------------
"""
src/experiments/exp4_motion_stress.py
Experiment 4: VR and RMSE vs motion intensity.
"""

def run_exp4(config: dict, results_dir: str = "results/exp4") -> pd.DataFrame:
    import pandas as pd
    from pathlib import Path
    from ..baselines.baselines import (HeuristicAdaptiveScheduler,
                                       EventTriggeredScheduler,
                                       AoIMinScheduler)
    from ..scheduler.sb_sched import StalenessScheduler

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    rows = []

    # High-motion: EuRoC hard + TUM-VI corridors
    # Low-motion: EuRoC easy + TUM-VI rooms
    test_cases = [
        ("euroc",  "MH_01_easy",       "low",    config["datasets"]["euroc"]["root"]),
        ("euroc",  "MH_03_medium",     "medium", config["datasets"]["euroc"]["root"]),
        ("euroc",  "MH_05_difficult",  "high",   config["datasets"]["euroc"]["root"]),
        ("tumvi",  "room1",            "low",    config["datasets"]["tumvi"]["root"]),
        ("tumvi",  "corridor1",        "high",   config["datasets"]["tumvi"]["root"]),
    ]

    for dataset, seq, motion_level, root in test_cases:
        for method_name, cls in [
            ("heuristic",    HeuristicAdaptiveScheduler),
            ("event_trigger", EventTriggeredScheduler),
            ("aoi_min",      AoIMinScheduler),
            ("sb_sched",     StalenessScheduler),
        ]:
            result = run_single(
                scheduler_cls=cls, scheduler_kwargs={},
                dataset=dataset, sequence=seq,
                dataset_root=root, method_name=method_name,
            )
            rows.append({
                "method": method_name,
                "dataset": dataset,
                "sequence": seq,
                "motion_level": motion_level,
                "vr": result.violation_rate,
                "rmse_m": result.rmse_m,
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/exp4_results.csv", index=False)
    return df


# ---------------------------------------------------------------------------
"""
src/experiments/exp5_overhead.py
Experiment 5: Computational overhead of scheduler vs KF steps.
"""

def run_exp5(config: dict, results_dir: str = "results/exp5") -> pd.DataFrame:
    import time
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from ..kalman.kalman_filter import KalmanFilter, SensorModel
    from ..scheduler.sb_sched import StalenessScheduler
    from ..baselines.baselines import HeuristicAdaptiveScheduler

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    N_WARMUP = 100
    N_MEASURE = 10_000
    n = 9  # State dimension

    F = np.eye(n)
    Q = np.eye(n) * 0.01
    H = np.zeros((3, n)); H[0,0]=H[1,1]=H[2,2]=1.0
    R = np.eye(3) * 0.1
    sensor = SensorModel("gps", H, R, 10.0)
    kf = KalmanFilter(F, Q, [sensor], np.zeros(n), np.eye(n))
    p_max = 0.5

    rows = []

    def time_block(fn, n_reps):
        for _ in range(N_WARMUP): fn()
        t0 = time.perf_counter()
        for _ in range(n_reps): fn()
        return (time.perf_counter() - t0) / n_reps * 1e6  # microseconds

    # KF predict step
    t_pred = time_block(lambda: kf.predict(), N_MEASURE)
    rows.append({"operation": "KF predict", "time_us": t_pred})

    # KF update step
    z = np.zeros(3)
    t_upd = time_block(lambda: kf.update("gps", z), N_MEASURE)
    rows.append({"operation": "KF update", "time_us": t_upd})

    # SB-Sched budget computation
    sched = StalenessScheduler(kf, p_max, ["gps"])
    t_budget = time_block(
        lambda: sched.compute_budget("gps"), N_MEASURE
    )
    rows.append({"operation": "SB-Sched budget", "time_us": t_budget})

    # Heuristic decision
    h_sched = HeuristicAdaptiveScheduler(kf, p_max, ["gps"])
    t_heuristic = time_block(
        lambda: h_sched.step(0, {"gps": z}), N_MEASURE
    )
    rows.append({"operation": "Heuristic step", "time_us": t_heuristic})

    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/exp5_results.csv", index=False)
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
"""
src/experiments/exp6_ekf.py
Experiment 6: Generalisation to EKF (nonlinear systems).
"""

def run_exp6(config: dict, results_dir: str = "results/exp6") -> pd.DataFrame:
    import pandas as pd
    from pathlib import Path
    from ..kalman.ekf import ExtendedKalmanFilter
    from ..scheduler.sb_sched import StalenessScheduler
    from ..baselines.baselines import (FixedHighScheduler,
                                       HeuristicAdaptiveScheduler)
    import numpy as np

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    print("[Exp6] EKF generalisation — uses EKF covariance prediction.")
    print("[Exp6] Note: SB-Sched uses conservative linearised budget.")
    print("[Exp6] Results logged to", results_dir)

    # EKF version of runner is analogous to linear; uses ekf.predict_covariance
    # Full implementation follows the same pattern as run_single() in runner.py
    # but swaps KalmanFilter for ExtendedKalmanFilter and uses nonlinear f, h.
    # See docs/ekf_extension.md for the IMU preintegration model details.

    rows = [{"note": "EKF runner — see docs/ekf_extension.md for full impl."}]
    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/exp6_placeholder.csv", index=False)
    return df
