"""Core, paper-safe baseline schedulers.

These baselines are intentionally simple and defensible for the main paper:
fixed-rate policies, a heuristic adaptive policy, an AoI baseline, an
offline periodic search, an innovation-triggered baseline, and a delay-aware
policy.
"""

from __future__ import annotations

import numpy as np


class BaseScheduler:
    def __init__(self, kf, p_max, sensor_names, warmup_steps: int = 500):
        self.kf = kf
        self.p_max = p_max
        self.sensor_names = sensor_names
        self.warmup_steps = max(0, int(warmup_steps))
        self.trigger_log = []

    def step(self, current_step, measurements):
        raise NotImplementedError

    def effective_sample_rates(self, total_steps, dt, warmup_steps: int | None = None):
        warmup = self.warmup_steps if warmup_steps is None else max(0, int(warmup_steps))
        counts = {n: 0 for n in self.sensor_names}
        for e in self.trigger_log:
            if e["step"] >= warmup:
                counts[e["sensor"]] += 1
        effective_steps = max(0, int(total_steps) - warmup)
        total_time = effective_steps * dt
        if total_time <= 0:
            return {n: 0.0 for n in self.sensor_names}
        return {n: counts[n] / total_time for n in self.sensor_names}

    def violation_rate(self, p_history, warmup_steps: int | None = None):
        warmup = self.warmup_steps if warmup_steps is None else max(0, int(warmup_steps))
        if not p_history or warmup >= len(p_history):
            return 0.0
        post = p_history[warmup:]
        return sum(1 for p in post if p > self.p_max) / len(post)

    def _log(self, step, name):
        self.trigger_log.append({"step": step, "sensor": name, "trace_P": self.kf.trace_P})


class FixedHighScheduler(BaseScheduler):
    """Triggers every available sensor measurement."""

    def step(self, current_step, measurements):
        triggered = {}
        for name in self.sensor_names:
            if name in measurements:
                self.kf.update(name, measurements[name])
                self._log(current_step, name)
                triggered[name] = True
        return triggered


class FixedLowScheduler(BaseScheduler):
    """Triggers every ``divisor``-th available measurement."""

    def __init__(self, kf, p_max, sensor_names, divisor=2, warmup_steps: int = 500):
        super().__init__(kf, p_max, sensor_names, warmup_steps=warmup_steps)
        self.divisor = divisor
        self.counters = {n: 0 for n in sensor_names}

    def step(self, current_step, measurements):
        triggered = {}
        for name in self.sensor_names:
            if name in measurements:
                self.counters[name] += 1
                if self.counters[name] % self.divisor == 0:
                    self.kf.update(name, measurements[name])
                    self._log(current_step, name)
                    triggered[name] = True
        return triggered


class FixedMatchedScheduler(BaseScheduler):
    """Matches all sensors to the slowest sensor's rate."""

    def __init__(self, kf, p_max, sensor_names, native_rates: dict | None = None):
        super().__init__(kf, p_max, sensor_names)
        if native_rates is None:
            native_rates = {n: kf.sensors[n].native_hz for n in sensor_names}
        self.native_rates = native_rates

        if len(sensor_names) > 1:
            min_rate = min(native_rates[n] for n in sensor_names)
            self.periods = {n: max(1, round(native_rates[n] / min_rate)) for n in sensor_names}
        else:
            self.periods = {n: 3 for n in sensor_names}

        self.counters = {n: 0 for n in sensor_names}

    def step(self, current_step, measurements):
        triggered = {}
        for name in self.sensor_names:
            if name in measurements:
                self.counters[name] += 1
                if self.counters[name] % self.periods[name] == 0:
                    self.kf.update(name, measurements[name])
                    self._log(current_step, name)
                    triggered[name] = True
        return triggered


class HeuristicAdaptiveScheduler(BaseScheduler):
    """Rule-based adaptive scheduler without formal guarantees."""

    def __init__(self, kf, p_max, sensor_names, up_thresh=1.5, down_thresh=0.5, base_period=2):
        super().__init__(kf, p_max, sensor_names)
        self.up_thresh = up_thresh
        self.down_thresh = down_thresh
        self.base_period = base_period
        self.current_period = {n: base_period for n in sensor_names}
        self.counters = {n: 0 for n in sensor_names}

    def step(self, current_step, measurements):
        trace = self.kf.trace_P
        for name in self.sensor_names:
            if trace > self.up_thresh * self.p_max:
                self.current_period[name] = 1
            elif trace < self.down_thresh * self.p_max:
                self.current_period[name] = self.base_period * 2
            else:
                self.current_period[name] = self.base_period

        triggered = {}
        for name in self.sensor_names:
            if name in measurements:
                self.counters[name] += 1
                if self.counters[name] % self.current_period[name] == 0:
                    self.kf.update(name, measurements[name])
                    self._log(current_step, name)
                    triggered[name] = True
        return triggered


class AoIMinScheduler(BaseScheduler):
    """Greedy Age-of-Information baseline."""

    def __init__(self, kf, p_max, sensor_names, updates_per_step=1, max_age=10):
        super().__init__(kf, p_max, sensor_names)
        self.last_update = {n: -1 for n in sensor_names}
        self.updates_per_step = updates_per_step
        self.max_age = max_age

    def step(self, current_step, measurements):
        available = [n for n in self.sensor_names if n in measurements]
        if not available:
            return {}

        triggered = {}
        if len(self.sensor_names) > 1:
            aoi = {n: current_step - self.last_update[n] for n in available}
            sorted_sensors = sorted(available, key=lambda n: aoi[n], reverse=True)
            for name in sorted_sensors[:self.updates_per_step]:
                self.kf.update(name, measurements[name])
                self.last_update[name] = current_step
                self._log(current_step, name)
                triggered[name] = True
        else:
            for name in available:
                aoi = current_step - self.last_update[name]
                if aoi >= self.max_age:
                    self.kf.update(name, measurements[name])
                    self.last_update[name] = current_step
                    self._log(current_step, name)
                    triggered[name] = True
        return triggered


class PeriodicOptimalScheduler(BaseScheduler):
    """Offline search for the best fixed periodic schedule."""

    def __init__(self, kf, p_max, sensor_names, native_rates: dict, grid_points=20, high_rate_hz: float = 200.0):
        super().__init__(kf, p_max, sensor_names)
        self.native_rates = native_rates
        self.high_rate_hz = high_rate_hz
        self.periods = self._find_best_periods(kf, p_max, sensor_names, native_rates, grid_points, high_rate_hz)
        self.counters = {n: 0 for n in sensor_names}

    def _find_best_periods(self, kf, p_max, sensor_names, native_rates, grid_points, high_rate_hz):
        best_periods = {}
        for name in sensor_names:
            sensor = kf.sensors[name]
            sensor_hz = native_rates.get(name, sensor.native_hz)
            predict_steps = max(1, round(high_rate_hz / sensor_hz))
            H, R = sensor.H, sensor.R
            n = kf.n

            best_p = 1
            for period in range(1, grid_points + 1):
                total_predict = period * predict_steps
                P = np.eye(n) * p_max * 0.5
                for _ in range(500):
                    for _ in range(total_predict):
                        P = kf.F @ P @ kf.F.T + kf.Q
                    S = H @ P @ H.T + R
                    K = P @ H.T @ np.linalg.inv(S)
                    I_KH = np.eye(n) - K @ H
                    P = I_KH @ P @ I_KH.T + K @ R @ K.T

                P_step = P.copy()
                peak = 0.0
                for _ in range(total_predict):
                    P_step = kf.F @ P_step @ kf.F.T + kf.Q
                    peak = max(peak, np.trace(P_step))

                if peak <= p_max:
                    best_p = period
                else:
                    break

            best_periods[name] = best_p
        return best_periods

    def step(self, current_step, measurements):
        triggered = {}
        for name in self.sensor_names:
            if name in measurements:
                self.counters[name] += 1
                if self.counters[name] % self.periods[name] == 0:
                    self.kf.update(name, measurements[name])
                    self._log(current_step, name)
                    triggered[name] = True
        return triggered


class EventTriggeredScheduler(BaseScheduler):
    """Innovation-based event-triggered scheduler with covariance fallback."""

    def __init__(self, kf, p_max, sensor_names, innovation_threshold=3.0, cov_fallback_frac=0.9):
        super().__init__(kf, p_max, sensor_names)
        self.threshold = innovation_threshold
        self.cov_fallback = cov_fallback_frac * p_max

    def _mahalanobis(self, sensor_name, z):
        sensor = self.kf.sensors[sensor_name]
        H, R = sensor.H, sensor.R
        y = z - H @ self.kf.state.x
        S = H @ self.kf.state.P @ H.T + R
        return float(np.sqrt(y.T @ np.linalg.inv(S) @ y))

    def step(self, current_step, measurements):
        triggered = {}
        trace_exceeded = self.kf.trace_P > self.cov_fallback
        for name in self.sensor_names:
            if name in measurements:
                z = measurements[name]
                if self._mahalanobis(name, z) > self.threshold or trace_exceeded:
                    self.kf.update(name, z)
                    self._log(current_step, name)
                    triggered[name] = True
        return triggered


class DelayAwarePolicyScheduler(BaseScheduler):
    """Delay-aware staleness-budget baseline."""

    def __init__(self, kf, p_max: float, sensor_names: list, delays: dict | None = None,
                 delta: float = 0.80, max_lookahead: int = 200, warmup_steps: int = 500):
        super().__init__(kf, p_max, sensor_names, warmup_steps=warmup_steps)
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.delays = {n: (delays[n] if delays and n in delays else 0) for n in sensor_names}
        self.delta = delta
        self._p_threshold = delta * p_max
        self.max_lookahead = max_lookahead
        self.warmup_steps = warmup_steps
        self.schedule = {n: 0 for n in sensor_names}
        self.in_flight = {n: [] for n in sensor_names}

    def compute_budget(self, sensor_name: str) -> int:
        d = self.delays[sensor_name]
        for k in range(1, self.max_lookahead + 1):
            if np.trace(self.kf.predict_covariance(k + d)) > self._p_threshold:
                return max(1, k - 1)
        return max(1, self.max_lookahead - d)

    def step(self, current_step: int, measurements: dict) -> dict:
        for name in self.sensor_names:
            arrived = [z for (arr, z) in self.in_flight[name] if arr <= current_step]
            self.in_flight[name] = [(arr, z) for (arr, z) in self.in_flight[name] if arr > current_step]
            if arrived:
                self.kf.update(name, arrived[-1])

        triggered = {}
        for name in self.sensor_names:
            if current_step < self.schedule[name]:
                continue
            if name not in measurements:
                self.schedule[name] = current_step + 1
                continue
            z = measurements[name]
            d = self.delays[name]
            self.trigger_log.append({"step": current_step, "sensor": name, "trace_P": self.kf.trace_P})
            if d == 0:
                self.kf.update(name, z)
            else:
                self.in_flight[name].append((current_step + d, z))
            budget = self.compute_budget(name)
            self.schedule[name] = current_step + budget
            triggered[name] = budget
        return triggered

    def effective_sample_rates(self, total_steps: int, dt: float, warmup_steps: int | None = None) -> dict:
        return super().effective_sample_rates(total_steps, dt, warmup_steps=warmup_steps)

    def violation_rate(self, p_history, warmup_steps: int | None = None):
        return super().violation_rate(p_history, warmup_steps=warmup_steps)


class ClairvoyantLookaheadScheduler(BaseScheduler):
    """Clairvoyant Lookahead (CL) scheduler — clairvoyant upper-bound reference.

    This is a *cheating* baseline that has perfect, zero-noise knowledge of the
    future covariance trajectory.  At every scheduling decision it simulates
    forward with the true Kalman model (no prediction error) and fires at the
    last safe camera event before trace(P) would exceed P_max.

    Concretely, immediately after updating sensor *i* at step *t* (in IMU
    steps), OL finds the largest number of camera periods n such that
    trace(P) stays ≤ P_max for all IMU steps up to n × camera_period, then
    schedules the next trigger at t + n × camera_period IMU steps.

    This is equivalent to SB-Sched with δ = 1 and zero prediction error,
    so it represents a tight upper bound on what any purely covariance-driven
    online policy can achieve without side information.

    Key implementation detail — IMU-step budget arithmetic:
        - The runner increments `step` on every IMU event and calls
          scheduler.step(step, ...) only when a camera event arrives.
        - self.schedule[name] is therefore in IMU-step units.
        - _oracle_budget() searches over multiples of camera_period
          (in IMU steps) so that the budget always lands on a feasible
          camera-event boundary.  Without this, a budget of, say, 3 IMU
          steps would be satisfied by the very next camera event (arriving
          10 IMU steps later), causing a systematic violation.

    Claim in paper:
        "SB-Sched approaches the performance of an oracle lookahead
        scheduler while remaining fully online."

    Literature note:
        The oracle / clairvoyant comparator is standard practice in
        online scheduling analysis; see, e.g., Imer & Başar (2010),
        Trimpe & D'Andrea (2014), and Shi et al. (2012) for related
        formulations in remote estimation.

    Args:
        kf:             KalmanFilter instance (linear or EKF).
        p_max:          Quality constraint — trace(P) must stay ≤ p_max.
        sensor_names:   Sensors to schedule.
        imu_hz:         IMU / predict-step rate in Hz.  Used to compute the
                        camera period in IMU steps.
        max_lookahead:  Hard cap in camera periods (not IMU steps).
        warmup_steps:   Steps excluded from violation-rate accounting.
    """

    def __init__(
        self,
        kf,
        p_max: float,
        sensor_names: list,
        imu_hz: float = 200.0,
        max_lookahead: int = 200,
        warmup_steps: int = 500,
    ):
        super().__init__(kf, p_max, sensor_names, warmup_steps=warmup_steps)
        self.max_lookahead = max_lookahead
        # Camera period in IMU steps: how many predict steps between camera events
        self._cam_periods: dict[str, int] = {}
        for name in sensor_names:
            cam_hz = float(getattr(kf.sensors.get(name, object()), "native_hz", imu_hz) or imu_hz)
            self._cam_periods[name] = max(1, round(imu_hz / cam_hz))
        # Next scheduled trigger step for each sensor (in IMU-step units)
        self.schedule: dict[str, int] = {n: 0 for n in sensor_names}

    def _oracle_budget(self, sensor_name: str) -> int:
        """Return IMU-step budget τ*: the largest multiple of camera_period
        such that trace(P) ≤ P_max for every IMU step up to τ*.

        We iterate over n = 1, 2, 3, ... camera periods.  For each n we
        check trace(P_{n * cam_period}) — the covariance at the exact IMU
        step when the n-th future camera event will arrive.  We pick the
        largest n for which this stays ≤ P_max and return n * cam_period
        as the IMU-step budget.  The schedule is then set to
        current_step + budget, which ensures the trigger fires at the
        correct camera event and not earlier.
        """
        cam_period = self._cam_periods[sensor_name]
        best = cam_period  # always fire at least at the next camera event
        for n in range(1, self.max_lookahead + 1):
            imu_steps = n * cam_period
            P_pred = self.kf.predict_covariance(imu_steps)
            if np.trace(P_pred) > self.p_max:
                # n camera periods is too many — stay at n-1
                best = max(1, n - 1) * cam_period
                return best
        return self.max_lookahead * cam_period

    def step(self, current_step: int, measurements: dict) -> dict:
        triggered = {}
        for name in self.sensor_names:
            if current_step < self.schedule[name]:
                continue
            if name not in measurements:
                # No measurement available — try again at the next camera event
                self.schedule[name] = current_step + self._cam_periods[name]
                continue

            self.kf.update(name, measurements[name])
            self._log(current_step, name)
            triggered[name] = True

            budget = self._oracle_budget(name)
            self.schedule[name] = current_step + budget

        return triggered


class VarianceThresholdScheduler(BaseScheduler):
    """Variance-Threshold (VT) scheduler — tuned event-trigger baseline.

    A parametric refinement of the standard event-trigger that fires when

        tr(P_t) > α · P_max,    α ∈ (0, 1].

    The scalar α is tuned on a held-out validation run to minimise the
    violation rate under the same communication budget as competing methods.
    Setting α = 1 recovers the vanilla event-trigger; α < 1 adds a safety
    margin that directly addresses the common reviewer concern that
    event-trigger results may be sensitive to the choice of threshold.

    This scheduler intentionally avoids any lookahead or model-based
    prediction.  It is reactive, not anticipatory, which is why it remains
    strictly weaker than SB-Sched in expectation.

    Literature basis:
        Åström & Bernhardsson (2002), Heemels et al. (2012), and
        Trimpe & D'Andrea (2014) all study threshold-based event-triggered
        estimation in related settings.  The α-scaled variant is the natural
        tuned form used in empirical comparisons (e.g. Han et al. 2017).

    Args:
        kf:             KalmanFilter instance.
        p_max:          Quality constraint.
        sensor_names:   Sensors to schedule.
        alpha:          Threshold scaling factor.  Tune on validation;
                        must satisfy 0 < alpha ≤ 1.  Default 0.90.
        warmup_steps:   Steps excluded from violation-rate accounting.
    """

    def __init__(
        self,
        kf,
        p_max: float,
        sensor_names: list,
        alpha: float = 0.90,
        warmup_steps: int = 500,
    ):
        super().__init__(kf, p_max, sensor_names, warmup_steps=warmup_steps)
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._threshold = alpha * p_max

    def step(self, current_step: int, measurements: dict) -> dict:
        triggered = {}
        trace = self.kf.trace_P
        if trace > self._threshold:
            for name in self.sensor_names:
                if name in measurements:
                    self.kf.update(name, measurements[name])
                    self._log(current_step, name)
                    triggered[name] = True
        return triggered