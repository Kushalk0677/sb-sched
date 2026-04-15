#!/usr/bin/env python3
"""
scripts/demo.py

Synthetic demo — runs all schedulers on a simulated 2-sensor system
without requiring any real dataset downloads.

Usage:
    python scripts/demo.py

Outputs a summary table and a covariance-over-time plot to results/demo/.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

matplotlib.rcParams.update({"font.size": 10, "figure.dpi": 120})

from src.kalman.kalman_filter import KalmanFilter, SensorModel
from src.scheduler.sb_sched import StalenessScheduler
from src.baselines.baselines import (
    FixedHighScheduler, FixedLowScheduler, FixedMatchedScheduler,
    HeuristicAdaptiveScheduler, AoIMinScheduler,
    EventTriggeredScheduler,
)


def make_system(seed=0):
    """
    2D position-velocity system.
    State: [px, py, vx, vy]
    High-rate sensor: IMU (predict, 200 Hz)
    Low-rate sensor:  GPS position (10 Hz)
    """
    np.random.seed(seed)
    dt = 1 / 200.0   # 200 Hz IMU
    n = 4
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]])
    Q = np.diag([1e-4, 1e-4, 1e-3, 1e-3])
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    R = np.eye(2) * 0.09   # 30 cm GPS noise
    sensor = SensorModel("gps", H, R, 10.0)
    x0 = np.zeros(n)
    P0 = np.eye(n) * 0.5
    return F, Q, [sensor], x0, P0


def run_scheduler(name, scheduler_cls, scheduler_kwargs,
                  F, Q, sensors, x0, P0,
                  total_steps=2000, gps_period=20, p_max=0.6):
    kf = KalmanFilter(F, Q, sensors, x0, P0)
    sched = scheduler_cls(kf=kf, p_max=p_max,
                          sensor_names=["gps"], **scheduler_kwargs)
    p_history = []
    true_pos = np.zeros(2)
    estimates = []
    gt = []

    for step in range(total_steps):
        # Simulate true dynamics
        noise = np.random.randn(4) * 0.01
        true_pos += np.array([0.05, 0.03]) + noise[:2]

        # IMU predict
        kf.predict()
        p_history.append(kf.trace_P)

        # GPS available every gps_period steps
        measurements = {}
        if step % gps_period == 0:
            z = true_pos + np.random.randn(2) * 0.3
            measurements["gps"] = z

        sched.step(step, measurements)
        estimates.append(kf.state.x[:2].copy())
        gt.append(true_pos.copy())

    estimates = np.array(estimates)
    gt = np.array(gt)
    rmse = float(np.sqrt(np.mean(np.sum((estimates - gt)**2, axis=1))))
    vr = sum(1 for p in p_history if p > p_max) / len(p_history)
    esr = sched.effective_sample_rates(total_steps, 1/200.0)
    n_updates = len(sched.trigger_log)
    return {
        "method": name, "vr": vr, "rmse": rmse,
        "esr_gps": esr.get("gps", 0), "n_updates": n_updates,
        "p_history": p_history,
    }


def main():
    out_dir = Path("results/demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    F, Q, sensors, x0, P0 = make_system()
    p_max = 0.6

    METHODS = [
        ("Fixed-High",     FixedHighScheduler,       {}),
        ("Fixed-Low",      FixedLowScheduler,         {}),
        ("Fixed-Matched",  FixedMatchedScheduler,
         {"native_rates": {"gps": 10.0}}),
        ("Heuristic",      HeuristicAdaptiveScheduler, {}),
        ("AoI-Min",        AoIMinScheduler,           {}),
        ("Event-Trigger",  EventTriggeredScheduler,   {}),
        ("SB-Sched (ours)",StalenessScheduler,        {}),
    ]

    results = []
    for name, cls, kwargs in METHODS:
        r = run_scheduler(name, cls, kwargs, F, Q, sensors, x0, P0,
                          p_max=p_max)
        results.append(r)
        print(f"  {name:22s}  VR={r['vr']:.3f}  "
              f"RMSE={r['rmse']:.4f}  "
              f"ESR={r['esr_gps']:.1f}Hz  "
              f"Updates={r['n_updates']}")

    # --- Plot: covariance traces ---
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(10, 12), sharex=True)
    colors = ["#2c7bb6","#abd9e9","#74add1","#fdae61",
              "#f46d43","#fee090","#1a9641"]

    for ax, r, col in zip(axes, results, colors):
        ax.plot(r["p_history"], color=col, linewidth=0.6, alpha=0.8)
        ax.axhline(p_max, color="red", linewidth=1.0, linestyle="--",
                   label=f"P_max={p_max}")
        ax.set_ylabel(r["method"], fontsize=8, rotation=0,
                      labelpad=80, va="center")
        ax.set_ylim(0, p_max * 3)
        vr_pct = r["vr"] * 100
        color_vr = "green" if vr_pct == 0 else "red"
        ax.annotate(f"VR={vr_pct:.1f}%", xy=(0.98, 0.85),
                    xycoords="axes fraction", ha="right",
                    color=color_vr, fontsize=8)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Covariance trace(P) over time\n(red dashed = P_max boundary)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "demo_covariance_traces.png")
    fig.savefig(out_dir / "demo_covariance_traces.pdf")
    print(f"\nFigure saved to {out_dir}/demo_covariance_traces.png")

    # --- Summary table ---
    print("\n--- Summary ---")
    print(f"{'Method':<24} {'VR':>6} {'RMSE(m)':>9} {'ESR(Hz)':>9} {'Updates':>9}")
    print("-" * 62)
    for r in results:
        star = " ✓" if r["vr"] == 0 else ""
        print(f"{r['method']:<24} {r['vr']:>6.3f} {r['rmse']:>9.4f}"
              f" {r['esr_gps']:>9.1f} {r['n_updates']:>9}{star}")
    print("\n✓ = VR=0 (quality guarantee met)")
    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
