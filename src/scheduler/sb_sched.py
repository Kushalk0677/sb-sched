"""
src/scheduler/sb_sched.py

Staleness-Budget Scheduler (SB-Sched) — the core novel contribution.

Algorithm:
    At each timestep t:
    1. For each sensor i, simulate predict-only covariance k steps ahead.
    2. Find τ*_i = smallest k s.t. trace(P_{t+k}) > δ·P_max   (δ < 1).
    3. Schedule sensor i to trigger at t + τ*_i.
    4. On trigger (called *after* predict), call kf.update(sensor_i, z_i).

Key property: guarantees trace(P(t)) ≤ P_max at all times once the
filter has converged from its initial conditions (post-warmup).

One-step lookahead gap and why δ closes it:
    compute_budget() is called immediately after an update, when
    trace(P) ≤ P_max.  But step() is invoked *after* predict, so by the
    time the trigger fires at t + τ*, the filter has already executed one
    predict step beyond the state at which the budget was computed.  Using
    the hard threshold P_max therefore allows trace(P) to exceed P_max for
    that one step before the update pulls it back.

    Setting the internal threshold to δ·P_max (δ < 1) ensures the sensor
    fires while there is still headroom: after the one intervening predict
    step, trace(P) reaches at most P_max rather than overshooting it.
    δ = 0.85 is the recommended default; Experiment 6 showed δ = 0.90
    already eliminates violations for the EKF — the linear KF is better
    conditioned, so 0.85 is conservative and safe across all experiments.

Note on warmup:
    The guarantee is a steady-state property. During the initial convergence
    phase (while P > P_max due to uncertain initialisation), the scheduler
    fires every step. Once P ≤ P_max the budget mechanism takes over and
    the formal guarantee holds. Set P0 ≤ P_max to skip warmup entirely,
    or use warmup_steps to track the post-warmup violation rate separately.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ScheduleEntry:
    """Next scheduled trigger for a sensor."""
    sensor_name: str
    trigger_step: int       # Absolute step at which to fire
    budget_steps: int       # τ*_i computed at scheduling time


class StalenessScheduler:
    """
    SB-Sched: Staleness-Budget Sensor Sampling Scheduler.

    Args:
        kf:            KalmanFilter (or EKF) instance
        p_max:         Quality threshold — trace(P) must stay ≤ p_max
        sensor_names:  List of sensor name strings to schedule
        max_lookahead: Maximum steps to look ahead when computing τ*
                       (if budget exceeds this, sensor idles until next check)
        warmup_steps:  Steps to exclude from VR calculation (initial convergence)
        delta:         Safety margin applied to the internal budget threshold.
                       compute_budget() triggers when trace(P_pred) > delta·p_max
                       rather than p_max, closing the one-step lookahead gap that
                       arises because step() is called after predict().
                       Must satisfy 0 < delta < 1.  Default 0.85.
    """

    def __init__(
        self,
        kf,
        p_max: float,
        sensor_names: list,
        max_lookahead: int = 200,
        warmup_steps: int = 100,
        delta: float = 0.85,
    ):
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.kf = kf
        self.p_max = p_max
        self.delta = delta
        self._p_threshold = delta * p_max   # internal trigger level
        self.sensor_names = sensor_names
        self.max_lookahead = max_lookahead
        self.warmup_steps = warmup_steps

        # Initialise schedule — trigger all sensors at step 0
        self.schedule: dict = {
            name: ScheduleEntry(name, 0, 0) for name in sensor_names
        }
        self.trigger_log: list = []

    def compute_budget(self, sensor_name: str) -> int:
        """
        Compute τ*_i: how many steps this sensor can idle before firing.

        Searches for the first k where trace(P_{t+k}) > δ·p_max (the
        internal threshold self._p_threshold).  Using δ·p_max rather than
        p_max absorbs the one-step lookahead gap: step() fires the update
        *after* predict, so the covariance has already grown by one step
        beyond the post-update state used here.  With δ = 0.85 that headroom
        is enough that trace(P) stays at or below p_max when the update lands.

        Uses kf.predict_covariance() which does NOT modify internal state.

        Returns:
            Integer ≥ 1.  Returns max_lookahead if budget exceeds window.
        """
        for k in range(1, self.max_lookahead + 1):
            P_pred = self.kf.predict_covariance(k)
            if np.trace(P_pred) > self._p_threshold:
                # k-1 steps are safe; fire at k-1 so the update lands
                # before trace(P) reaches even delta·p_max.
                return max(1, k - 1)
        return self.max_lookahead

    def step(self, current_step: int, measurements: dict) -> dict:
        """
        Main scheduler step. Call once per high-rate timestep (after predict).

        Args:
            current_step:  Current discrete timestep index
            measurements:  Dict of available measurements {sensor_name: z}

        Returns:
            Dict of sensors triggered this step {name: budget_used}.
        """
        triggered = {}

        for name in self.sensor_names:
            entry = self.schedule[name]

            if current_step >= entry.trigger_step:
                if name in measurements:
                    self.kf.update(name, measurements[name])
                    budget = self.compute_budget(name)
                    self.schedule[name] = ScheduleEntry(
                        sensor_name=name,
                        trigger_step=current_step + budget,
                        budget_steps=budget,
                    )
                    triggered[name] = budget
                    self.trigger_log.append({
                        "step": current_step,
                        "sensor": name,
                        "budget": budget,
                        "trace_P": self.kf.trace_P,
                    })
                else:
                    # Measurement missing — retry next step
                    self.schedule[name] = ScheduleEntry(name, current_step + 1, 1)

        return triggered

    def effective_sample_rates(self, total_steps: int, dt: float, warmup_steps: int | None = None) -> dict:
        """
        Compute effective sampling rate (Hz) for each sensor.
        Only counts post-warmup triggers to match VR metric.
        """
        counts = {name: 0 for name in self.sensor_names}
        for entry in self.trigger_log:
            if entry["step"] >= self.warmup_steps:
                counts[entry["sensor"]] += 1
        total_time = max(total_steps - self.warmup_steps, 1) * dt
        return {name: counts[name] / total_time for name in self.sensor_names}

    def violation_rate(self, p_history: list, warmup_steps: int | None = None) -> float:
        """
        Fraction of post-warmup steps where trace(P) > p_max.

        Args:
            p_history: List of trace(P) values recorded after each predict+update.
        """
        warmup = self.warmup_steps if warmup_steps is None else max(0, int(warmup_steps))
        post = p_history[warmup:]
        if not post:
            return 0.0
        return sum(1 for p in post if p > self.p_max) / len(post)
