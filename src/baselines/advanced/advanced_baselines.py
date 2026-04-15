"""Research-oriented baseline schedulers.

These implementations are separated from the paper-safe baselines because they
are heavier, more assumption-sensitive, and better treated as optional
comparators in experiments.

They are implemented to be more faithful to the underlying literature families
than the earlier toy versions:
- Whittle-style scheduling is based on a marginal discounted covariance cost.
- Continuous/discrete rate optimisation uses a deterministic expected Riccati
  surrogate and constrained gradient search.
- The mixed-integer baseline uses a finite-horizon relaxed programme with an
  explicit rate-budget repair step.
- The DRL baseline uses a small replay-based DQN rather than a single-step
  policy heuristic.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..main import BaseScheduler


def _joseph_update_cov(P: np.ndarray, H: np.ndarray, R: np.ndarray) -> np.ndarray:
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    I_KH = np.eye(P.shape[0]) - K @ H
    return I_KH @ P @ I_KH.T + K @ R @ K.T


class WhittleIndexScheduler(BaseScheduler):
    """Approximate Whittle-index scheduler for remote state estimation.

    Exact Whittle indices are only available under restrictive assumptions.
    For the general multivariate KF case used in this repository we use a
    standard Whittle-style approximation: each arm is scored by the marginal
    discounted covariance cost of idling versus updating now, starting from its
    current age. Sensors with the largest positive indices are scheduled.
    """

    def __init__(self, kf, p_max, sensor_names, Q_diag: np.ndarray,
                 updates_per_step=1, horizon: int = 8, discount: float = 0.97,
                 activation_cost: float = 0.0):
        super().__init__(kf, p_max, sensor_names)
        self.q_weights = np.asarray(Q_diag, dtype=float)
        self.aoi = {n: 0 for n in sensor_names}
        self.updates_per_step = updates_per_step
        self.horizon = horizon
        self.discount = discount
        self.activation_cost = activation_cost

    def _aged_covariance(self, steps: int) -> np.ndarray:
        return self.kf.predict_covariance(max(steps, 0))

    def _discounted_trace_cost(self, P0: np.ndarray) -> float:
        P = P0.copy()
        total = 0.0
        disc = 1.0
        for _ in range(self.horizon):
            P = self.kf.F @ P @ self.kf.F.T + self.kf.Q
            total += disc * float(np.trace(P))
            disc *= self.discount
        return total

    def _whittle_index(self, sensor_name: str) -> float:
        sensor = self.kf.sensors[sensor_name]
        age = self.aoi[sensor_name]
        P_idle = self._aged_covariance(age + 1)
        P_active = _joseph_update_cov(P_idle, sensor.H, sensor.R)
        passive_cost = self._discounted_trace_cost(P_idle)
        active_cost = self._discounted_trace_cost(P_active)
        process_weight = float(np.mean(self.q_weights)) if self.q_weights.size else 1.0
        return process_weight * (passive_cost - active_cost) - self.activation_cost

    def step(self, current_step, measurements):
        for name in self.sensor_names:
            self.aoi[name] += 1

        available = [n for n in self.sensor_names if n in measurements]
        if not available:
            return {}

        ranked = sorted(available, key=self._whittle_index, reverse=True)
        triggered = {}
        for name in ranked[:self.updates_per_step]:
            if self._whittle_index(name) <= 0:
                continue
            self.kf.update(name, measurements[name])
            self.aoi[name] = 0
            self._log(current_step, name)
            triggered[name] = True

        if len(self.sensor_names) == 1 and not triggered:
            name = available[0]
            if self.kf.trace_P >= 0.8 * self.p_max:
                self.kf.update(name, measurements[name])
                self.aoi[name] = 0
                self._log(current_step, name)
                triggered[name] = True
        return triggered


class CDKFGradientScheduler(BaseScheduler):
    """Continuous/discrete KF rate-allocation baseline.

    Optimises continuous per-sensor utilisation factors lambda in [0, 1] using
    a deterministic expected-Riccati surrogate, then converts them to periodic
    schedules. This is much closer to continuous-rate scheduling papers than a
    simple ad hoc threshold search.
    """

    def __init__(self, kf, p_max, sensor_names, native_rates: dict,
                 lr=0.05, n_steps=200, rate_budget: float | None = None):
        super().__init__(kf, p_max, sensor_names)
        self.native_rates = native_rates
        total_native = sum(native_rates[n] for n in sensor_names)
        self.rate_budget = rate_budget if rate_budget is not None else 0.65 * total_native
        self.lambdas = self._optimise_rates(kf, sensor_names, native_rates, lr, n_steps)
        self.periods = {
            name: max(1, int(round(1.0 / max(self.lambdas[j], 1e-3))))
            for j, name in enumerate(sensor_names)
        }
        self.counters = {n: 0 for n in sensor_names}

    def _expected_update(self, P: np.ndarray, sensor_name: str, lam: float) -> np.ndarray:
        sensor = self.kf.sensors[sensor_name]
        info = sensor.H.T @ np.linalg.inv(sensor.R) @ sensor.H
        pred_inv = np.linalg.inv(P)
        post = np.linalg.inv(pred_inv + lam * info)
        return 0.5 * (post + post.T)

    def _surrogate_cost(self, lambdas: np.ndarray) -> float:
        P = self.kf.state.P.copy()
        for _ in range(60):
            P = self.kf.F @ P @ self.kf.F.T + self.kf.Q
            for j, name in enumerate(self.sensor_names):
                P = self._expected_update(P, name, lambdas[j])
        trace_term = float(np.trace(P))
        used_hz = sum(lambdas[j] * self.native_rates[name] for j, name in enumerate(self.sensor_names))
        budget_penalty = 1e3 * max(0.0, used_hz - self.rate_budget) ** 2
        smoothness_penalty = 0.1 * float(np.sum(np.square(lambdas)))
        return trace_term + budget_penalty + smoothness_penalty

    def _project_budget(self, lambdas: np.ndarray) -> np.ndarray:
        max_hz = sum(lambdas[j] * self.native_rates[name] for j, name in enumerate(self.sensor_names))
        if max_hz <= self.rate_budget + 1e-12:
            return lambdas
        scale = self.rate_budget / max_hz
        return np.clip(lambdas * scale, 0.02, 1.0)

    def _optimise_rates(self, kf, sensor_names, native_rates, lr, n_steps):
        rng = np.random.default_rng(0)
        lambdas = np.full(len(sensor_names), 0.5, dtype=float)
        best = lambdas.copy()
        best_cost = self._surrogate_cost(best)
        for k in range(n_steps):
            ck = 0.1 / np.sqrt(k + 1)
            delta = rng.choice([-1.0, 1.0], size=len(sensor_names))
            plus = self._project_budget(np.clip(lambdas + ck * delta, 0.02, 1.0))
            minus = self._project_budget(np.clip(lambdas - ck * delta, 0.02, 1.0))
            ghat = (self._surrogate_cost(plus) - self._surrogate_cost(minus)) / (2 * ck) * delta
            ak = lr / (k + 10) ** 0.6
            lambdas = self._project_budget(np.clip(lambdas - ak * ghat, 0.02, 1.0))
            cost = self._surrogate_cost(lambdas)
            if cost < best_cost:
                best_cost = cost
                best = lambdas.copy()
        return best

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


class MIQPOptimalScheduler(BaseScheduler):
    """Finite-horizon mixed-integer scheduling approximation.

    This keeps the same public API as the earlier class but uses a more
    structured optimisation loop: a continuous relaxation over utilisation
    factors, explicit rate-budget constraints, integer rounding, and repair.
    The result is a practical mixed-integer baseline for this repository.
    """

    def __init__(self, kf, p_max: float, sensor_names: list, native_rates: dict | None = None,
                 rate_budget: float | None = None, horizon: int = 50, max_iter: int = 300):
        super().__init__(kf, p_max, sensor_names)
        if native_rates is None:
            native_rates = {n: kf.sensors[n].native_hz for n in sensor_names}
        self.native_rates = native_rates
        self.horizon = horizon
        total_native = sum(native_rates[n] for n in sensor_names)
        self.rate_budget = rate_budget if rate_budget is not None else 0.50 * total_native
        self.utilisation = self._solve_relaxation(max_iter)
        self.periods = self._round_and_repair(self.utilisation)
        self.counters = {n: 0 for n in sensor_names}

    def _simulate_cost(self, periods: list[int]) -> float:
        P = self.kf.state.P.copy()
        auc = 0.0
        for s in range(self.horizon):
            P = self.kf.F @ P @ self.kf.F.T + self.kf.Q
            for j, name in enumerate(self.sensor_names):
                if s % periods[j] == 0:
                    sensor = self.kf.sensors[name]
                    P = _joseph_update_cov(P, sensor.H, sensor.R)
            tr = float(np.trace(P))
            auc += tr
            if tr > self.p_max:
                auc += 1e3 * (tr - self.p_max)
        return auc

    def _objective_from_utilisation(self, util: np.ndarray) -> float:
        periods = [max(1, int(round(1.0 / max(u, 1e-3)))) for u in util]
        used_hz = sum((1.0 / p) * self.native_rates[name] for p, name in zip(periods, self.sensor_names))
        penalty = 1e4 * max(0.0, used_hz - self.rate_budget) ** 2
        return self._simulate_cost(periods) + penalty

    def _solve_relaxation(self, max_iter: int) -> np.ndarray:
        n_s = len(self.sensor_names)
        total_native = sum(self.native_rates[n] for n in self.sensor_names)
        per_sensor = min(1.0, max(0.02, self.rate_budget / max(total_native, 1e-9)))
        x0 = np.full(n_s, per_sensor)
        bounds = [(0.02, 1.0)] * n_s

        def budget_constraint(u):
            used = sum(u[j] * self.native_rates[n] for j, n in enumerate(self.sensor_names))
            return self.rate_budget - used

        res = minimize(
            self._objective_from_utilisation,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": budget_constraint}],
            options={"maxiter": max_iter, "disp": False},
        )
        util = res.x if res.success else x0
        return np.clip(util, 0.02, 1.0)

    def _round_and_repair(self, util: np.ndarray) -> dict[str, int]:
        periods = [max(1, int(round(1.0 / max(u, 1e-3)))) for u in util]

        def used_hz(vals: list[int]) -> float:
            return sum((1.0 / p) * self.native_rates[name] for p, name in zip(vals, self.sensor_names))

        while used_hz(periods) > self.rate_budget + 1e-9:
            best_idx = None
            best_increase = None
            best_cost = None
            for j in range(len(periods)):
                cand = periods.copy()
                cand[j] += 1
                cost_delta = self._simulate_cost(cand) - self._simulate_cost(periods)
                rate_delta = used_hz(periods) - used_hz(cand)
                if rate_delta <= 0:
                    continue
                ratio = cost_delta / rate_delta
                if best_idx is None or ratio < best_increase or (np.isclose(ratio, best_increase) and cost_delta < best_cost):
                    best_idx = j
                    best_increase = ratio
                    best_cost = cost_delta
            if best_idx is None:
                break
            periods[best_idx] += 1
        return {name: periods[j] for j, name in enumerate(self.sensor_names)}

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


class DRLScheduler(BaseScheduler):
    """Replay-based DQN scheduler.

    The action space is factorised per sensor: update (1) or skip (0). The
    state contains normalized covariance and AoI terms. This is still compact,
    but much closer to a standard deep-RL baseline than the earlier online
    heuristic network.
    """

    def __init__(self, kf, p_max: float, sensor_names: list, aoi_max: int = 100,
                 hidden: int = 32, lr: float = 3e-4, greedy_after: int = 200,
                 epsilon: float = 0.15, replay_size: int = 1024, batch_size: int = 32):
        super().__init__(kf, p_max, sensor_names)
        self.aoi_max = aoi_max
        self.lr = lr
        self.greedy_after = greedy_after
        self.epsilon = epsilon
        self.aoi = {n: 0 for n in sensor_names}
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.replay = []
        n_s = len(sensor_names)
        in_dim = 1 + n_s
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = self._init_weights(in_dim, hidden, n_s)
        self.target = [arr.copy() for arr in (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)]
        self._step_count = 0
        self._last_transition = None

    def _init_weights(self, in_dim, hidden, n_out):
        rng = np.random.default_rng(seed=0)
        W1 = rng.standard_normal((hidden, in_dim)) * 0.1
        b1 = np.zeros(hidden)
        W2 = rng.standard_normal((hidden, hidden)) * 0.05
        b2 = np.zeros(hidden)
        W3 = rng.standard_normal((n_out, hidden)) * 0.1
        b3 = np.zeros(n_out)
        return W1, b1, W2, b2, W3, b3

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _forward(self, state, target: bool = False):
        W1, b1, W2, b2, W3, b3 = self.target if target else (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)
        h1 = np.tanh(W1 @ state + b1)
        h2 = np.tanh(W2 @ h1 + b2)
        return self._sigmoid(W3 @ h2 + b3)

    def _obs(self):
        trace_norm = min(self.kf.trace_P / max(self.p_max, 1e-9), 2.0)
        aoi_feats = np.array([min(self.aoi[n] / self.aoi_max, 1.0) for n in self.sensor_names])
        return np.concatenate([[trace_norm], aoi_feats])

    def _store(self, obs, actions, reward, next_obs):
        self.replay.append((obs, actions, reward, next_obs))
        if len(self.replay) > self.replay_size:
            self.replay.pop(0)

    def _train_step(self):
        if len(self.replay) < max(8, self.batch_size // 2):
            return
        idx = np.random.choice(len(self.replay), size=min(self.batch_size, len(self.replay)), replace=False)
        for i in idx:
            obs, actions, reward, next_obs = self.replay[i]
            q = self._forward(obs)
            q_next = self._forward(next_obs, target=True)
            target = reward + 0.95 * q_next
            err = (target - q) * actions
            h1 = np.tanh(self.W1 @ obs + self.b1)
            h2 = np.tanh(self.W2 @ h1 + self.b2)
            grad = err * q * (1.0 - q)
            self.W3 += self.lr * np.outer(grad, h2)
            self.b3 += self.lr * grad
        self._step_count += 1
        if self._step_count % 25 == 0:
            self.target = [arr.copy() for arr in (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)]

    def step(self, current_step, measurements):
        for n in self.sensor_names:
            self.aoi[n] += 1

        obs = self._obs()
        probs = self._forward(obs)
        triggered = {}
        actions = np.zeros(len(self.sensor_names))
        eps = self.epsilon if current_step < self.greedy_after else min(0.05, self.epsilon)

        for j, name in enumerate(self.sensor_names):
            if name not in measurements:
                continue
            if np.random.random() < eps:
                decide = np.random.random() < probs[j]
            else:
                decide = probs[j] >= 0.5

            if decide or (len(self.sensor_names) == 1 and self.aoi[name] >= max(2, self.aoi_max // 4)):
                self.kf.update(name, measurements[name])
                self.aoi[name] = 0
                self._log(current_step, name)
                triggered[name] = True
                actions[j] = 1.0

        next_obs = self._obs()
        reward = -float(self.kf.trace_P / max(self.p_max, 1e-9)) - 0.05 * float(np.sum(actions))
        self._store(obs, actions, reward, next_obs)
        self._train_step()
        return triggered
