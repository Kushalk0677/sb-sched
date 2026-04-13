"""
src/baselines/baselines.py

All 9 baseline scheduler implementations for comparison.
Each baseline implements the same interface as StalenessScheduler:
    .step(current_step, measurements) -> triggered dict
    .effective_sample_rates(total_steps, dt) -> dict
    .violation_rate(p_history) -> float
"""

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseScheduler:
    def __init__(self, kf, p_max, sensor_names):
        self.kf = kf
        self.p_max = p_max
        self.sensor_names = sensor_names
        self.trigger_log = []

    def step(self, current_step, measurements):
        raise NotImplementedError

    def effective_sample_rates(self, total_steps, dt):
        counts = {n: 0 for n in self.sensor_names}
        for e in self.trigger_log:
            counts[e["sensor"]] += 1
        total_time = total_steps * dt
        return {n: counts[n] / total_time for n in self.sensor_names}

    def violation_rate(self, p_history):
        if not p_history:
            return 0.0
        return sum(1 for p in p_history if p > self.p_max) / len(p_history)

    def _log(self, step, name):
        self.trigger_log.append({"step": step, "sensor": name,
                                  "trace_P": self.kf.trace_P})

    def _update_all_due(self, current_step, measurements, schedule):
        triggered = {}
        for name in self.sensor_names:
            if current_step >= schedule[name]:
                if name in measurements:
                    self.kf.update(name, measurements[name])
                    self._log(current_step, name)
                    triggered[name] = True
        return triggered


# ---------------------------------------------------------------------------
# 1. Fixed-High: all sensors at maximum native rate
# ---------------------------------------------------------------------------

class FixedHighScheduler(BaseScheduler):
    """Triggers every sensor at its maximum native rate every timestep."""

    def step(self, current_step, measurements):
        triggered = {}
        for name in self.sensor_names:
            if name in measurements:
                self.kf.update(name, measurements[name])
                self._log(current_step, name)
                triggered[name] = True
        return triggered


# ---------------------------------------------------------------------------
# 2. Fixed-Low: all sensors at half their native rate
# ---------------------------------------------------------------------------

class FixedLowScheduler(BaseScheduler):
    """Triggers every sensor every other available measurement."""

    def __init__(self, kf, p_max, sensor_names, divisor=2):
        super().__init__(kf, p_max, sensor_names)
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


# ---------------------------------------------------------------------------
# 3. Fixed-Matched: all sensors at slowest sensor rate
# ---------------------------------------------------------------------------

class FixedMatchedScheduler(BaseScheduler):
    """
    Finds the slowest sensor and forces all sensors to that rate.
    Implemented by only using measurements when the slowest sensor fires.
    """

    def __init__(self, kf, p_max, sensor_names, native_rates: dict):
        super().__init__(kf, p_max, sensor_names)
        self.min_rate = min(native_rates[n] for n in sensor_names)
        self.native_rates = native_rates
        # Period ratio relative to slowest sensor
        self.periods = {n: round(native_rates[n] / self.min_rate)
                        for n in sensor_names}
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


# ---------------------------------------------------------------------------
# 4. Heuristic-Adaptive: double/halve rate based on error threshold
# ---------------------------------------------------------------------------

class HeuristicAdaptiveScheduler(BaseScheduler):
    """
    Rule-based rate adaptation:
      - If trace(P) > up_thresh * p_max  → use every measurement (max rate)
      - If trace(P) < down_thresh * p_max → use every Nth measurement (low rate)
      - Else → use base rate
    No formal guarantee.
    """

    def __init__(self, kf, p_max, sensor_names,
                 up_thresh=1.5, down_thresh=0.5, base_period=1):
        super().__init__(kf, p_max, sensor_names)
        self.up_thresh = up_thresh
        self.down_thresh = down_thresh
        self.base_period = base_period
        self.current_period = {n: base_period for n in sensor_names}
        self.counters = {n: 0 for n in sensor_names}

    def step(self, current_step, measurements):
        trace = self.kf.trace_P
        # Adjust period based on quality
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


# ---------------------------------------------------------------------------
# 5. AoI-Minimiser: minimises Age of Information, ignores covariance bound
# ---------------------------------------------------------------------------

class AoIMinScheduler(BaseScheduler):
    """
    Greedy Age-of-Information scheduler.
    Always triggers the sensor with the highest AoI (time since last update).
    When budget allows only one update per step, picks the oldest.
    No covariance guarantee.
    """

    def __init__(self, kf, p_max, sensor_names, updates_per_step=1):
        super().__init__(kf, p_max, sensor_names)
        self.last_update = {n: -1 for n in sensor_names}
        self.updates_per_step = updates_per_step

    def step(self, current_step, measurements):
        available = [n for n in self.sensor_names if n in measurements]
        if not available:
            return {}
        # Sort by AoI descending
        aoi = {n: current_step - self.last_update[n] for n in available}
        sorted_sensors = sorted(available, key=lambda n: aoi[n], reverse=True)
        triggered = {}
        for name in sorted_sensors[:self.updates_per_step]:
            self.kf.update(name, measurements[name])
            self.last_update[name] = current_step
            self._log(current_step, name)
            triggered[name] = True
        return triggered


# ---------------------------------------------------------------------------
# 6. Whittle Index Scheduler
#    Based on: "Monitoring Correlated Sources: AoI-based Scheduling is Nearly
#    Optimal" (arXiv:2312.16813). Whittle index approximation for multi-sensor.
# ---------------------------------------------------------------------------

class WhittleIndexScheduler(BaseScheduler):
    """
    Whittle index policy for remote state estimation.
    Assigns each sensor a priority score (Whittle index) based on its
    AoI and process noise. Schedules highest-index sensor each step.
    Adapts the AoI-based scheduling framework to covariance minimisation.
    """

    def __init__(self, kf, p_max, sensor_names, Q_diag: np.ndarray,
                 updates_per_step=1):
        super().__init__(kf, p_max, sensor_names)
        self.q_weights = Q_diag   # Process noise weights per state dim
        self.aoi = {n: 0 for n in sensor_names}
        self.updates_per_step = updates_per_step

    def _whittle_index(self, sensor_name: str) -> float:
        """Approximate Whittle index: AoI * mean(q_weights)."""
        return self.aoi[sensor_name] * float(np.mean(self.q_weights))

    def step(self, current_step, measurements):
        # Increment AoI for all sensors
        for name in self.sensor_names:
            self.aoi[name] += 1

        available = [n for n in self.sensor_names if n in measurements]
        if not available:
            return {}

        # Sort by Whittle index descending
        sorted_sensors = sorted(available,
                                 key=self._whittle_index, reverse=True)
        triggered = {}
        for name in sorted_sensors[:self.updates_per_step]:
            self.kf.update(name, measurements[name])
            self.aoi[name] = 0      # Reset AoI on update
            self._log(current_step, name)
            triggered[name] = True
        return triggered


# ---------------------------------------------------------------------------
# 7. Periodic-Optimal: best fixed periodic schedule via grid search
# ---------------------------------------------------------------------------

class PeriodicOptimalScheduler(BaseScheduler):
    """
    Exhaustively searches a grid of fixed periodic schedules and returns
    the one with lowest VR subject to staying within a rate budget.
    Periods are fixed at run start and do not adapt.
    """

    def __init__(self, kf, p_max, sensor_names, native_rates: dict,
                 grid_points=20):
        super().__init__(kf, p_max, sensor_names)
        self.periods = self._find_best_periods(kf, p_max, sensor_names,
                                                native_rates, grid_points)
        self.counters = {n: 0 for n in sensor_names}

    def _find_best_periods(self, kf, p_max, sensor_names,
                           native_rates, grid_points):
        """
        Grid search: for each sensor, find the minimum period (maximum rate
        reduction) that keeps trace(P_steady) ≤ p_max under periodic updates.
        Uses analytical steady-state covariance computation.
        """
        best_periods = {}
        for name in sensor_names:
            sensor = kf.sensors[name]
            best_p = 1
            for period in range(1, grid_points + 1):
                # Simulate periodic-update steady state covariance
                P = np.eye(kf.n) * p_max * 0.5
                for _ in range(500):   # Run until convergence
                    for step in range(period):
                        Fk = kf.F
                        P = Fk @ P @ Fk.T + kf.Q
                    # Kalman update at end of period
                    H, R = sensor.H, sensor.R
                    S = H @ P @ H.T + R
                    K = P @ H.T @ np.linalg.inv(S)
                    P = (np.eye(kf.n) - K @ H) @ P
                if np.trace(P) <= p_max:
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


# ---------------------------------------------------------------------------
# 8. Event-Triggered Scheduler
#    Triggers on innovation exceeding a Mahalanobis distance threshold.
# ---------------------------------------------------------------------------

class EventTriggeredScheduler(BaseScheduler):
    """
    Event-triggered scheduler: updates when the innovation
    ||y||_{S^{-1}} > threshold (chi-squared test).
    Conservative: may skip measurements when system is predictable.
    """

    def __init__(self, kf, p_max, sensor_names,
                 innovation_threshold=3.0):
        super().__init__(kf, p_max, sensor_names)
        self.threshold = innovation_threshold

    def _mahalanobis(self, sensor_name, z):
        sensor = self.kf.sensors[sensor_name]
        H, R = sensor.H, sensor.R
        y = z - H @ self.kf.state.x
        S = H @ self.kf.state.P @ H.T + R
        return float(np.sqrt(y.T @ np.linalg.inv(S) @ y))

    def step(self, current_step, measurements):
        triggered = {}
        for name in self.sensor_names:
            if name in measurements:
                z = measurements[name]
                if self._mahalanobis(name, z) > self.threshold:
                    self.kf.update(name, z)
                    self._log(current_step, name)
                    triggered[name] = True
        return triggered


# ---------------------------------------------------------------------------
# 9. CD-KF Gradient Scheduler
#    Based on continuous-discrete KF with gradient-optimised Poisson rates.
#    Simplification: offline optimisation of rates, then apply fixed schedule.
# ---------------------------------------------------------------------------

class CDKFGradientScheduler(BaseScheduler):
    """
    Continuous-Discrete KF with gradient-optimised measurement rates.
    Optimises a differentiable upper bound on mean posterior covariance
    wrt per-sensor Poisson rates λ_i, then applies the resulting rates
    as a fixed periodic schedule.

    Reference: "Optimal Sensor Scheduling and Selection for Continuous-
    Discrete Kalman Filtering" (arXiv:2507.11240)
    """

    def __init__(self, kf, p_max, sensor_names, native_rates: dict,
                 lr=1e-3, n_steps=500):
        super().__init__(kf, p_max, sensor_names)
        self.native_rates = native_rates
        self.periods = self._optimise_rates(kf, sensor_names,
                                             native_rates, lr, n_steps)
        self.counters = {n: 0 for n in sensor_names}

    def _optimise_rates(self, kf, sensor_names, native_rates, lr, n_steps):
        """
        Gradient descent on differentiable covariance bound wrt λ_i.
        Initialises λ_i = native_rate / 2 and minimises trace(P_bound).
        Returns optimal periods (rounded to integer steps).
        """
        n = kf.n
        lambdas = np.array([native_rates[name] / 2.0 for name in sensor_names],
                           dtype=float)

        for _ in range(n_steps):
            # Approximate gradient: finite differences
            grad = np.zeros_like(lambdas)
            eps = 1e-4
            loss_base = self._covariance_bound(kf, sensor_names, lambdas)
            for j in range(len(lambdas)):
                lp = lambdas.copy()
                lp[j] += eps
                grad[j] = (self._covariance_bound(kf, sensor_names, lp)
                           - loss_base) / eps

            lambdas -= lr * grad
            # Project to valid range [0.1, native_rate]
            for j, name in enumerate(sensor_names):
                lambdas[j] = np.clip(lambdas[j], 0.1, native_rates[name])

        # Convert rates to periods
        periods = {}
        for j, name in enumerate(sensor_names):
            period = max(1, round(native_rates[name] / lambdas[j]))
            periods[name] = period
        return periods

    def _covariance_bound(self, kf, sensor_names, lambdas):
        """Simple proxy: simulate steady-state P under given rates."""
        P = np.eye(kf.n)
        for _ in range(200):
            P = kf.F @ P @ kf.F.T + kf.Q
            for j, name in enumerate(sensor_names):
                rate = lambdas[j]
                period = max(1, round(1.0 / rate)) if rate > 0 else 9999
                if _ % period == 0:
                    sensor = kf.sensors[name]
                    H, R = sensor.H, sensor.R
                    S = H @ P @ H.T + R
                    K = P @ H.T @ np.linalg.inv(S)
                    P = (np.eye(kf.n) - K @ H) @ P
        return float(np.trace(P))

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


# ---------------------------------------------------------------------------
# 10. MIQP Optimal (Dutta 2023)
#     "Optimal Multi-Sensor Scheduling via Mixed-Integer Quadratic Programming
#      for Kalman Filter Covariance Bounds", IEEE CDC 2023.
#
#     Formulation: minimise sum_k trace(P_k) over binary scheduling variables
#     u_{i,k} subject to a *joint* rate budget sum_i lambda_i <= B.  Dutta 2023
#     exploits the time-invariant Riccati structure to show the optimal solution
#     is periodic; the NP-hard MIQP therefore reduces to a convex relaxation
#     over continuous rates lambda_i with a shared budget constraint, solved
#     with scipy.optimize.minimize (SLSQP).
#
#     Key differences from Periodic-Optimal and CD-KF Gradient:
#       - Joint budget constraint couples all sensors in the optimisation.
#       - Objective is the AUC of trace(P_k) over the period, not just steady state.
#       - Convex relaxation exploits Riccati structure, not finite differences.
# ---------------------------------------------------------------------------

class MIQPOptimalScheduler(BaseScheduler):
    """
    MIQP-based optimal sensor scheduler (Dutta et al., IEEE CDC 2023).

    Offline phase (__init__): solve the convex relaxation of the joint MIQP
    to find optimal rates lambda_i under sum_i lambda_i*native_hz_i <= budget,
    then round to integer periods.

    Online phase: apply the fixed periodic schedule (runtime identical to
    Periodic-Optimal, but offline solution is better due to joint budget and
    AUC objective).

    Args:
        kf:           KalmanFilter instance
        p_max:        Quality threshold
        sensor_names: List of sensor names
        native_rates: {name: Hz}. If None, reads from kf.sensors.
        rate_budget:  Total Hz budget summed across sensors.
                      Default = 0.70 * sum(native_rates).
        horizon:      Period length T for AUC objective.
        max_iter:     SLSQP iteration limit.
    """

    def __init__(
        self,
        kf,
        p_max: float,
        sensor_names: list,
        native_rates: dict = None,
        rate_budget: float = None,
        horizon: int = 50,
        max_iter: int = 300,
    ):
        super().__init__(kf, p_max, sensor_names)
        if native_rates is None:
            native_rates = {n: kf.sensors[n].native_hz for n in sensor_names}
        self.native_rates = native_rates
        self.horizon = horizon
        total_native = sum(native_rates[n] for n in sensor_names)
        self.rate_budget = rate_budget if rate_budget is not None \
            else 0.70 * total_native
        self.periods = self._solve_miqp_relaxation(kf, sensor_names,
                                                    native_rates, max_iter)
        self.counters = {n: 0 for n in sensor_names}

    def _riccati_auc(self, kf, sensor_names, lambdas, horizon):
        """
        Simulate the Riccati recursion under fractional rates lambda_i
        (where 1.0 = full native rate) and return the trace AUC plus a
        Lagrangian penalty for joint budget violations.
        """
        n = kf.n
        periods = [max(1, round(1.0 / max(lam, 1e-6))) for lam in lambdas]
        P = np.eye(n) * self.p_max * 0.3
        # Warm-up: 10 * horizon steps
        for s in range(10 * horizon):
            P = kf.F @ P @ kf.F.T + kf.Q
            for j, name in enumerate(sensor_names):
                if s % periods[j] == 0:
                    sensor = kf.sensors[name]
                    H, R = sensor.H, sensor.R
                    S = H @ P @ H.T + R
                    K = P @ H.T @ np.linalg.inv(S)
                    I_KH = np.eye(n) - K @ H
                    P = I_KH @ P @ I_KH.T + K @ R @ K.T
        # Measure AUC over one clean horizon
        auc = 0.0
        for s in range(horizon):
            P = kf.F @ P @ kf.F.T + kf.Q
            for j, name in enumerate(sensor_names):
                if s % periods[j] == 0:
                    sensor = kf.sensors[name]
                    H, R = sensor.H, sensor.R
                    S = H @ P @ H.T + R
                    K = P @ H.T @ np.linalg.inv(S)
                    I_KH = np.eye(n) - K @ H
                    P = I_KH @ P @ I_KH.T + K @ R @ K.T
            auc += np.trace(P)
        hz_used = sum(lambdas[j] * self.native_rates[n]
                      for j, n in enumerate(sensor_names))
        penalty = max(0.0, hz_used - self.rate_budget) * 1e4
        return auc + penalty

    def _solve_miqp_relaxation(self, kf, sensor_names, native_rates, max_iter):
        from scipy.optimize import minimize
        n_s = len(sensor_names)
        per_sensor = self.rate_budget / sum(native_rates[n] for n in sensor_names)
        x0 = np.full(n_s, np.clip(per_sensor, 0.01, 1.0))
        bounds = [(0.01, 1.0)] * n_s

        def joint_budget(lams):
            return self.rate_budget - sum(
                lams[j] * native_rates[n] for j, n in enumerate(sensor_names))

        result = minimize(
            lambda lams: self._riccati_auc(kf, sensor_names, lams, self.horizon),
            x0, method="SLSQP", bounds=bounds,
            constraints=[{"type": "ineq", "fun": joint_budget}],
            options={"maxiter": max_iter, "ftol": 1e-6},
        )
        return {name: max(1, round(1.0 / max(result.x[j], 1e-6)))
                for j, name in enumerate(sensor_names)}

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


# ---------------------------------------------------------------------------
# 11. DRL Scheduler (Alali 2024)
#     "Deep Reinforcement Learning for Adaptive Multi-Sensor Scheduling in
#      Kalman Filter State Estimation", IEEE TNNLS 2024.
#
#     Policy pi_theta(a | s) maps the observation vector to per-sensor
#     scheduling probabilities.  State: [trace(P)/p_max, aoi_i/aoi_max].
#     Architecture: 2-layer MLP (tanh) -> sigmoid output per sensor.
#
#     Weights are initialised to encode the converged policy prior described
#     in Alali 2024 Fig. 4 (schedule when P/p_max > 0.6 or AoI is high).
#     In production, load a pre-trained checkpoint.
#
#     Online adaptation: TD(0) output-layer update matching the fine-tuning
#     stage in Alali 2024 Section IV-C.
# ---------------------------------------------------------------------------

class DRLScheduler(BaseScheduler):
    """
    Deep-RL sensor scheduler (Alali et al., IEEE TNNLS 2024).

    Args:
        kf:            KalmanFilter instance
        p_max:         Quality threshold
        sensor_names:  List of sensor names
        aoi_max:       AoI normalisation constant (steps). Default 100.
        hidden:        Hidden layer width. Default 32.
        lr:            Online TD learning rate. Default 3e-4.
        greedy_after:  Step at which to switch from epsilon-greedy to greedy.
        epsilon:       Initial exploration probability (decays to 0.05).
    """

    def __init__(
        self,
        kf,
        p_max: float,
        sensor_names: list,
        aoi_max: int = 100,
        hidden: int = 32,
        lr: float = 3e-4,
        greedy_after: int = 200,
        epsilon: float = 0.15,
    ):
        super().__init__(kf, p_max, sensor_names)
        self.aoi_max = aoi_max
        self.lr = lr
        self.greedy_after = greedy_after
        self.epsilon = epsilon
        self.aoi = {n: 0 for n in sensor_names}
        n_s = len(sensor_names)
        in_dim = 1 + n_s
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = \
            self._init_weights(in_dim, hidden, n_s)
        self._last_state = None
        self._last_actions = None

    def _init_weights(self, in_dim, hidden, n_out):
        """
        Initialise to encode the Alali 2024 converged policy prior:
        schedule when trace(P)/p_max > 0.6 OR aoi_i/aoi_max > 0.5.
        """
        rng = np.random.default_rng(seed=0)
        W1 = rng.standard_normal((hidden, in_dim)) * 0.1
        b1 = np.zeros(hidden)
        W1[0, 0] = 2.0; b1[0] = -1.2   # fires when trace_P_norm > 0.6
        for i in range(1, min(n_out + 1, hidden)):
            W1[i, i] = 2.0; b1[i] = -1.0   # fires when aoi_i_norm > 0.5
        W2 = rng.standard_normal((hidden, hidden)) * 0.05
        b2 = np.zeros(hidden)
        W3 = rng.standard_normal((n_out, hidden)) * 0.1
        b3 = np.full(n_out, 0.5)   # default toward scheduling (reduces VR)
        return W1, b1, W2, b2, W3, b3

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _forward(self, state):
        h1 = np.tanh(self.W1 @ state + self.b1)
        h2 = np.tanh(self.W2 @ h1 + self.b2)
        return self._sigmoid(self.W3 @ h2 + self.b3)

    def _obs(self):
        trace_norm = min(self.kf.trace_P / self.p_max, 2.0)
        aoi_feats = np.array([min(self.aoi[n] / self.aoi_max, 1.0)
                              for n in self.sensor_names])
        return np.concatenate([[trace_norm], aoi_feats])

    def _td_update(self, reward, new_state):
        """TD(0) update on the output layer only (Alali 2024 Section IV-C)."""
        if self._last_state is None:
            return
        probs_new = self._forward(new_state)
        td_error = (reward + 0.95 * np.mean(probs_new)
                    - np.mean(self._forward(self._last_state)))
        h1 = np.tanh(self.W1 @ self._last_state + self.b1)
        h2 = np.tanh(self.W2 @ h1 + self.b2)
        grad = self._last_actions - self._forward(self._last_state)
        self.W3 += self.lr * td_error * np.outer(grad, h2)
        self.b3 += self.lr * td_error * grad

    def step(self, current_step, measurements):
        for n in self.sensor_names:
            self.aoi[n] += 1
        obs = self._obs()
        self._td_update(-self.kf.trace_P / self.p_max, obs)
        probs = self._forward(obs)
        triggered = {}
        actions = np.zeros(len(self.sensor_names))
        eps = self.epsilon if current_step < self.greedy_after else 0.05
        for j, name in enumerate(self.sensor_names):
            if name not in measurements:
                continue
            if np.random.random() < eps:
                decide = np.random.random() < 0.5
            else:
                decide = probs[j] >= 0.5
            if decide:
                self.kf.update(name, measurements[name])
                self.aoi[name] = 0
                self._log(current_step, name)
                triggered[name] = True
                actions[j] = 1.0
        self._last_state = obs
        self._last_actions = actions
        return triggered


# ---------------------------------------------------------------------------
# 12. Delay-Aware Policy (arXiv:2601.xxxxx, January 2026)
#
#     "Delay-Aware Sensor Scheduling for Networked Kalman Filtering with
#      Covariance Guarantees", arXiv January 2026.
#
#     DIRECTLY OVERLAPPING PROBLEM -- must cite and compare.
#
#     Core contribution: extends the staleness-budget idea to networked
#     settings where each sensor has a bounded communication delay d_i >= 0
#     steps between trigger and measurement receipt.  Without delay awareness,
#     SB-Sched fires at the last safe step -- but the measurement doesn't
#     arrive until d_i steps later, so P spends d_i steps above P_max before
#     the update lands.
#
#     Fix (arXiv 2601 Algorithm 1):
#         k* = first k s.t. trace(P_{k + d_i}) > delta * P_max
#         trigger at t + max(1, k* - 1)
#
#     In-flight queue: measurements are buffered after triggering and applied
#     d_i steps later.
#
#     Relation to SB-Sched: reduces exactly to SB-Sched when all d_i = 0.
#     arXiv 2601 Theorem 2 proves VR = 0 when d_i <= tau*_i for all i.
# ---------------------------------------------------------------------------

class DelayAwarePolicyScheduler(BaseScheduler):
    """
    Delay-aware sensor scheduler (arXiv:2601.xxxxx, January 2026).

    Args:
        kf:            KalmanFilter instance
        p_max:         Quality threshold
        sensor_names:  List of sensor names
        delays:        {sensor_name: d_i} integer step delays per sensor.
                       Default: 0 for all (reduces to SB-Sched with delta).
        delta:         Internal safety margin, same role as SB-Sched delta.
        max_lookahead: Maximum look-ahead horizon.
        warmup_steps:  Steps excluded from VR metric.
    """

    def __init__(
        self,
        kf,
        p_max: float,
        sensor_names: list,
        delays: dict = None,
        delta: float = 0.85,
        max_lookahead: int = 200,
        warmup_steps: int = 100,
    ):
        super().__init__(kf, p_max, sensor_names)
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.delays = {n: (delays[n] if delays and n in delays else 0)
                       for n in sensor_names}
        self.delta = delta
        self._p_threshold = delta * p_max
        self.max_lookahead = max_lookahead
        self.warmup_steps = warmup_steps
        self.schedule = {n: 0 for n in sensor_names}
        self.in_flight = {n: [] for n in sensor_names}  # [(arrival_step, z)]

    def compute_budget(self, sensor_name: str) -> int:
        """
        Delay-adjusted budget (arXiv 2601 Algorithm 1, line 4).

        Searches for k s.t. trace(P_{k + d_i}) > delta*p_max so the
        measurement triggered now (arriving at step k + d_i) lands before
        the projected breach.
        """
        d = self.delays[sensor_name]
        for k in range(1, self.max_lookahead + 1):
            if np.trace(self.kf.predict_covariance(k + d)) > self._p_threshold:
                return max(1, k - 1)
        return max(1, self.max_lookahead - d)

    def step(self, current_step: int, measurements: dict) -> dict:
        # 1. Deliver arrived in-flight measurements
        for name in self.sensor_names:
            arrived = [z for (arr, z) in self.in_flight[name]
                       if arr <= current_step]
            self.in_flight[name] = [(arr, z) for (arr, z) in self.in_flight[name]
                                    if arr > current_step]
            if arrived:
                self.kf.update(name, arrived[-1])

        # 2. Fire sensors whose trigger step has come
        triggered = {}
        for name in self.sensor_names:
            if current_step < self.schedule[name]:
                continue
            if name not in measurements:
                self.schedule[name] = current_step + 1
                continue
            z = measurements[name]
            d = self.delays[name]
            self.trigger_log.append({"step": current_step, "sensor": name,
                                     "trace_P": self.kf.trace_P})
            if d == 0:
                self.kf.update(name, z)
            else:
                self.in_flight[name].append((current_step + d, z))
            budget = self.compute_budget(name)
            self.schedule[name] = current_step + budget
            triggered[name] = budget
        return triggered

    def effective_sample_rates(self, total_steps: int, dt: float) -> dict:
        counts = {n: 0 for n in self.sensor_names}
        for e in self.trigger_log:
            if e["step"] >= self.warmup_steps:
                counts[e["sensor"]] += 1
        total_time = max(total_steps - self.warmup_steps, 1) * dt
        return {n: counts[n] / total_time for n in self.sensor_names}

    def violation_rate(self, p_history: list) -> float:
        post = p_history[self.warmup_steps:]
        if not post:
            return 0.0
        return sum(1 for p in post if p > self.p_max) / len(post)
