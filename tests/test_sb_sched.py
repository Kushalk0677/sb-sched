"""
tests/test_sb_sched.py

Unit tests for SB-Sched scheduler and Kalman filter.
Run with:  pytest tests/ -v
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Allow running from project root or from within the tests directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kalman.kalman_filter import KalmanFilter, SensorModel
from kalman.ekf import ExtendedKalmanFilter
from scheduler.sb_sched import StalenessScheduler
from baselines.baselines import (
    FixedHighScheduler,
    FixedLowScheduler,
    HeuristicAdaptiveScheduler,
    AoIMinScheduler,
    EventTriggeredScheduler,
    DelayAwarePolicyScheduler,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_kf(n: int = 6, P0_scale: float = 0.1) -> KalmanFilter:
    """Minimal 6-dim constant-velocity KF with one GPS sensor."""
    dt = 0.01
    F = np.eye(n)
    F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
    Q = np.eye(n) * 0.001 * dt
    H = np.zeros((3, n)); H[0, 0] = H[1, 1] = H[2, 2] = 1.0
    R = np.eye(3) * 0.1
    sensor = SensorModel("gps", H, R, 10.0)
    return KalmanFilter(F, Q, [sensor], x0=np.zeros(n), P0=np.eye(n) * P0_scale)


def make_dual_kf(n: int = 6) -> KalmanFilter:
    """KF with two independent position sensors for multi-sensor tests."""
    dt = 0.01
    F = np.eye(n)
    F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
    Q = np.eye(n) * 0.001 * dt
    H = np.zeros((3, n)); H[0, 0] = H[1, 1] = H[2, 2] = 1.0
    s1 = SensorModel("s1", H.copy(), np.eye(3) * 0.1, 10.0)
    s2 = SensorModel("s2", H.copy(), np.eye(3) * 0.2,  5.0)
    return KalmanFilter(F, Q, [s1, s2], x0=np.zeros(n), P0=np.eye(n) * 0.1)


# ── KalmanFilter: basic properties ────────────────────────────────────────────

class TestKalmanFilterBasic:

    def test_predict_increases_covariance(self):
        kf = make_kf()
        t0 = kf.trace_P
        kf.predict()
        assert kf.trace_P > t0

    def test_update_decreases_covariance(self):
        kf = make_kf()
        kf.predict(10)
        t0 = kf.trace_P
        kf.update("gps", np.zeros(3))
        assert kf.trace_P < t0

    def test_predict_covariance_matches_manual(self):
        kf = make_kf()
        P_pred = kf.predict_covariance(5)
        Fk = np.linalg.matrix_power(kf.F, 5)
        Qk = kf._accumulated_Q(5)
        P_exp = Fk @ kf.state.P @ Fk.T + Qk
        assert np.allclose(P_pred, P_exp, rtol=1e-10)

    def test_predict_covariance_does_not_mutate_state(self):
        kf = make_kf()
        P_before = kf.state.P.copy()
        kf.predict_covariance(10)
        assert np.allclose(kf.state.P, P_before)

    def test_positive_definite_after_many_steps(self):
        rng = np.random.default_rng(42)
        kf = make_kf()
        for _ in range(500):
            kf.predict()
            kf.update("gps", rng.standard_normal(3))
        eigvals = np.linalg.eigvalsh(kf.state.P)
        assert np.all(eigvals > 0)

    def test_joseph_form_preserves_symmetry_under_extreme_innovation(self):
        kf = make_kf()
        kf.predict(50)
        kf.update("gps", np.ones(3) * 100)  # extreme outlier
        diff = kf.state.P - kf.state.P.T
        assert np.max(np.abs(diff)) < 1e-12

    def test_longer_prediction_gives_larger_covariance(self):
        kf = make_kf()
        P1 = kf.predict_covariance(1)
        P5 = kf.predict_covariance(5)
        assert np.trace(P5) > np.trace(P1)

    def test_predict_covariance_k_matches_k_predict_calls(self):
        kf = make_kf()
        P_seq = kf.predict_covariance(3)
        kf2 = make_kf()
        for _ in range(3):
            kf2.predict()
        assert np.allclose(P_seq, kf2.state.P, rtol=1e-8)


# ── StalenessScheduler: configuration ─────────────────────────────────────────

class TestStalenessSchedulerConfig:

    def test_delta_stored_correctly(self):
        sched = StalenessScheduler(make_kf(P0_scale=0.05), p_max=1.0,
                                    sensor_names=["gps"], delta=0.85)
        assert sched.delta == 0.85

    def test_threshold_equals_delta_times_pmax(self):
        sched = StalenessScheduler(make_kf(P0_scale=0.05), p_max=1.0,
                                    sensor_names=["gps"], delta=0.85)
        assert np.isclose(sched._p_threshold, 0.85)

    def test_delta_equal_to_one_raises(self):
        with pytest.raises(ValueError):
            StalenessScheduler(make_kf(), p_max=1.0,
                                sensor_names=["gps"], delta=1.0)

    def test_delta_equal_to_zero_raises(self):
        with pytest.raises(ValueError):
            StalenessScheduler(make_kf(), p_max=1.0,
                                sensor_names=["gps"], delta=0.0)

    def test_lower_delta_gives_smaller_or_equal_budget(self):
        kf = make_kf(P0_scale=0.05)
        s_tight = StalenessScheduler(kf, p_max=1.0, sensor_names=["gps"], delta=0.5)
        s_loose = StalenessScheduler(kf, p_max=1.0, sensor_names=["gps"], delta=0.99)
        assert s_tight.compute_budget("gps") <= s_loose.compute_budget("gps")


# ── StalenessScheduler: budget computation ────────────────────────────────────

class TestStalenessSchedulerBudget:

    def test_budget_is_positive_integer(self):
        sched = StalenessScheduler(make_kf(P0_scale=0.05), p_max=1.0,
                                    sensor_names=["gps"])
        b = sched.compute_budget("gps")
        assert isinstance(b, int) and b >= 1

    def test_tighter_pmax_gives_smaller_budget(self):
        kf = make_kf(P0_scale=0.05)
        s_tight = StalenessScheduler(kf, p_max=0.2, sensor_names=["gps"])
        s_loose = StalenessScheduler(kf, p_max=5.0, sensor_names=["gps"])
        assert s_tight.compute_budget("gps") <= s_loose.compute_budget("gps")

    def test_larger_current_P_gives_smaller_budget(self):
        kf1 = make_kf(P0_scale=0.05)
        kf1.predict(30)  # inflate P
        b_inflated = StalenessScheduler(kf1, p_max=1.0,
                                         sensor_names=["gps"]).compute_budget("gps")
        kf2 = make_kf(P0_scale=0.05)
        b_fresh = StalenessScheduler(kf2, p_max=1.0,
                                      sensor_names=["gps"]).compute_budget("gps")
        assert b_inflated <= b_fresh

    def test_budget_capped_at_max_lookahead(self):
        sched = StalenessScheduler(make_kf(P0_scale=0.05), p_max=1.0,
                                    sensor_names=["gps"], max_lookahead=5)
        assert sched.compute_budget("gps") <= 5


# ── StalenessScheduler: quality guarantee ─────────────────────────────────────

class TestStalenessSchedulerGuarantee:

    def test_vr_zero_post_warmup(self):
        """Core paper claim: VR = 0 after warmup with P0 ≤ P_max."""
        rng = np.random.default_rng(42)
        kf = make_kf(P0_scale=0.05)
        p_max = 1.0
        sched = StalenessScheduler(kf, p_max=p_max, sensor_names=["gps"],
                                    warmup_steps=50)
        p_history = []
        for step in range(2000):
            kf.predict()
            sched.step(step, {"gps": rng.standard_normal(3) * 0.1})
            p_history.append(kf.trace_P)

        vr = sched.violation_rate(p_history)
        assert vr == 0.0, (
            f"VR={vr:.4f}, max post-warmup trace(P)={max(p_history[50:]):.4f}, "
            f"p_max={p_max}"
        )

    def test_trigger_log_non_empty(self):
        rng = np.random.default_rng(0)
        kf = make_kf(P0_scale=0.05)
        sched = StalenessScheduler(kf, p_max=1.0, sensor_names=["gps"])
        for step in range(200):
            kf.predict()
            sched.step(step, {"gps": rng.standard_normal(3) * 0.1})
        assert len(sched.trigger_log) > 0


# ── One-step lookahead gap regression ─────────────────────────────────────────

class TestLookaheadGapRegression:
    """
    Without delta < 1, the budget is computed in the post-update state, but
    step() fires the update *after* predict(), so trace(P) can overshoot p_max
    for one step.  With delta=0.85, this gap must be closed.
    """

    def test_no_exposure_point_violations_with_default_delta(self):
        rng = np.random.default_rng(7)
        kf = make_kf(P0_scale=0.05)
        p_max = 0.8
        sched = StalenessScheduler(kf, p_max=p_max, sensor_names=["gps"],
                                    warmup_steps=50, delta=0.85)
        p_before_update = []  # trace(P) AFTER predict, BEFORE update
        for step in range(1500):
            kf.predict()
            p_before_update.append(kf.trace_P)  # worst-case exposure point
            sched.step(step, {"gps": rng.standard_normal(3) * 0.1})

        post = p_before_update[50:]
        violations = sum(1 for p in post if p > p_max)
        assert violations == 0, (
            f"violations={violations}, max exposed trace(P)={max(post):.4f}, "
            f"p_max={p_max}"
        )


# ── StalenessScheduler: efficiency ────────────────────────────────────────────

class TestStalenessSchedulerEfficiency:

    def test_fewer_samples_than_fixed_high_under_loose_pmax(self):
        rng = np.random.default_rng(0)
        kf_sb = make_kf(P0_scale=0.05)
        kf_fh = make_kf(P0_scale=0.05)
        p_max = 5.0  # loose → SB-Sched should idle frequently
        sched_sb = StalenessScheduler(kf_sb, p_max=p_max, sensor_names=["gps"])
        sched_fh = FixedHighScheduler(kf_fh, p_max=p_max, sensor_names=["gps"])
        for step in range(1000):
            kf_sb.predict(); kf_fh.predict()
            z = {"gps": rng.standard_normal(3) * 0.1}
            sched_sb.step(step, z)
            sched_fh.step(step, z)
        esr_sb = sched_sb.effective_sample_rates(1000, 0.01)
        esr_fh = sched_fh.effective_sample_rates(1000, 0.01)
        assert esr_sb["gps"] <= esr_fh["gps"]

    def test_meaningful_rate_reduction_under_loose_pmax(self):
        rng = np.random.default_rng(0)
        kf_sb = make_kf(P0_scale=0.05)
        kf_fh = make_kf(P0_scale=0.05)
        p_max = 5.0
        sched_sb = StalenessScheduler(kf_sb, p_max=p_max, sensor_names=["gps"])
        sched_fh = FixedHighScheduler(kf_fh, p_max=p_max, sensor_names=["gps"])
        for step in range(1000):
            kf_sb.predict(); kf_fh.predict()
            z = {"gps": rng.standard_normal(3) * 0.1}
            sched_sb.step(step, z); sched_fh.step(step, z)
        esr_sb = sched_sb.effective_sample_rates(1000, 0.01)
        esr_fh = sched_fh.effective_sample_rates(1000, 0.01)
        rr = (esr_fh["gps"] - esr_sb["gps"]) / esr_fh["gps"] * 100
        assert rr > 10, f"RR={rr:.1f}% — expected >10% reduction"


# ── StalenessScheduler: robustness ────────────────────────────────────────────

class TestStalenessSchedulerRobustness:

    def test_missing_measurement_no_crash(self):
        kf = make_kf(P0_scale=0.05)
        sched = StalenessScheduler(kf, p_max=1.0, sensor_names=["gps"])
        kf.predict()
        triggered = sched.step(1, {})
        assert isinstance(triggered, dict)

    def test_missing_measurement_reschedules_next_step(self):
        kf = make_kf(P0_scale=0.05)
        sched = StalenessScheduler(kf, p_max=1.0, sensor_names=["gps"])
        kf.predict()
        sched.step(1, {})
        assert sched.schedule["gps"].trigger_step == 2


# ── Baseline schedulers ───────────────────────────────────────────────────────

class TestFixedHighScheduler:

    def test_uses_every_measurement(self):
        kf = make_kf()
        sched = FixedHighScheduler(kf, p_max=1.0, sensor_names=["gps"])
        for step in range(100):
            kf.predict()
            sched.step(step, {"gps": np.zeros(3)})
        assert len(sched.trigger_log) == 100


class TestFixedLowScheduler:

    def test_divisor_2_uses_half(self):
        kf = make_kf()
        sched = FixedLowScheduler(kf, p_max=1.0, sensor_names=["gps"], divisor=2)
        for step in range(100):
            kf.predict()
            sched.step(step, {"gps": np.zeros(3)})
        assert len(sched.trigger_log) == 50

    def test_divisor_4_uses_quarter(self):
        kf = make_kf()
        sched = FixedLowScheduler(kf, p_max=1.0, sensor_names=["gps"], divisor=4)
        for step in range(100):
            kf.predict()
            sched.step(step, {"gps": np.zeros(3)})
        assert len(sched.trigger_log) == 25


class TestEventTriggeredScheduler:

    def test_fires_on_large_innovation(self):
        kf = make_kf()
        sched = EventTriggeredScheduler(kf, p_max=1.0, sensor_names=["gps"],
                                         innovation_threshold=0.01)
        kf.predict()
        triggered = sched.step(0, {"gps": np.ones(3) * 100})
        assert "gps" in triggered

    def test_suppresses_small_innovation(self):
        kf = make_kf()
        sched = EventTriggeredScheduler(kf, p_max=1.0, sensor_names=["gps"],
                                         innovation_threshold=1e6)
        kf.predict()
        triggered = sched.step(0, {"gps": np.zeros(3)})
        assert "gps" not in triggered


class TestAoIMinScheduler:

    def test_selects_oldest_sensor(self):
        kf = make_dual_kf()
        sched = AoIMinScheduler(kf, p_max=1.0, sensor_names=["s1", "s2"],
                                 updates_per_step=1)
        kf.predict()
        sched.last_update["s1"] = -20  # s1 much older
        sched.last_update["s2"] = -2
        triggered = sched.step(1, {"s1": np.zeros(3), "s2": np.zeros(3)})
        assert "s1" in triggered

    def test_only_one_update_per_step(self):
        kf = make_dual_kf()
        sched = AoIMinScheduler(kf, p_max=1.0, sensor_names=["s1", "s2"],
                                 updates_per_step=1)
        kf.predict()
        triggered = sched.step(1, {"s1": np.zeros(3), "s2": np.zeros(3)})
        assert len(triggered) <= 1


# ── DelayAwarePolicyScheduler ─────────────────────────────────────────────────

class TestDelayAwarePolicyScheduler:

    def test_delays_stored_correctly(self):
        dap = DelayAwarePolicyScheduler(make_kf(P0_scale=0.05), p_max=1.0,
                                         sensor_names=["gps"], delays={"gps": 0})
        assert dap.delays["gps"] == 0

    def test_threshold_equals_delta_times_pmax(self):
        dap = DelayAwarePolicyScheduler(make_kf(P0_scale=0.05), p_max=1.0,
                                         sensor_names=["gps"], delta=0.85)
        assert np.isclose(dap._p_threshold, 0.85)

    def test_zero_delay_budget_matches_sb_sched(self):
        kf = make_kf(P0_scale=0.05)
        dap = DelayAwarePolicyScheduler(kf, p_max=1.0, sensor_names=["gps"],
                                         delays={"gps": 0}, delta=0.85)
        sb = StalenessScheduler(kf, p_max=1.0, sensor_names=["gps"], delta=0.85)
        assert dap.compute_budget("gps") == sb.compute_budget("gps")

    def test_nonzero_delay_reduces_budget(self):
        kf = make_kf(P0_scale=0.05)
        dap_0 = DelayAwarePolicyScheduler(kf, p_max=1.0, sensor_names=["gps"],
                                           delays={"gps": 0}, delta=0.85)
        dap_3 = DelayAwarePolicyScheduler(kf, p_max=1.0, sensor_names=["gps"],
                                           delays={"gps": 3}, delta=0.85)
        assert dap_3.compute_budget("gps") <= dap_0.compute_budget("gps")

    def test_vr_zero_with_zero_delay(self):
        rng = np.random.default_rng(3)
        kf = make_kf(P0_scale=0.05)
        p_max = 0.9
        dap = DelayAwarePolicyScheduler(kf, p_max=p_max, sensor_names=["gps"],
                                         delays={"gps": 0}, delta=0.85,
                                         warmup_steps=50)
        p_hist = []
        for step in range(1500):
            kf.predict()
            p_hist.append(kf.trace_P)
            dap.step(step, {"gps": rng.standard_normal(3) * 0.1})
        assert dap.violation_rate(p_hist) == 0.0

    def test_in_flight_arrival_step_correct(self):
        kf = make_kf(P0_scale=0.05)
        dap = DelayAwarePolicyScheduler(kf, p_max=5.0, sensor_names=["gps"],
                                         delays={"gps": 4}, delta=0.85,
                                         warmup_steps=0)
        kf.predict()
        dap.step(0, {"gps": np.ones(3)})
        if dap.trigger_log:  # sensor fired at step 0
            arrival_steps = [arr for (arr, _) in dap.in_flight["gps"]]
            assert 4 in arrival_steps, (
                f"Expected arrival at step 4, got {arrival_steps}"
            )


# ── ExtendedKalmanFilter ──────────────────────────────────────────────────────

class TestExtendedKalmanFilter:
    """Minimal EKF sanity checks using a linear model for ground truth."""

    @pytest.fixture
    def ekf(self):
        dt = 0.01
        F = np.eye(4); F[0, 2] = dt; F[1, 3] = dt

        def f(x): return F @ x
        def jac_f(x): return F
        def h(x): return x[:2]
        def jac_h(x):
            H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1.0; return H

        Q = np.eye(4) * 0.001
        sensor_model = SensorModel("gps", np.zeros((2, 4)), np.eye(2) * 0.1, 10.0)
        return ExtendedKalmanFilter(
            f=f, jac_f=jac_f, Q=Q,
            sensors=[(sensor_model, h, jac_h)],
            x0=np.zeros(4), P0=np.eye(4) * 0.5,
        )

    def test_predict_increases_covariance(self, ekf):
        t0 = ekf.trace_P
        ekf.predict()
        assert ekf.trace_P > t0

    def test_update_decreases_covariance(self, ekf):
        ekf.predict()
        t0 = ekf.trace_P
        ekf.update("gps", np.zeros(2))
        assert ekf.trace_P < t0

    def test_predict_covariance_correct_shape(self, ekf):
        assert ekf.predict_covariance(3).shape == (4, 4)

    def test_predict_covariance_increases_trace(self, ekf):
        P_pred = ekf.predict_covariance(3)
        assert np.trace(P_pred) > ekf.trace_P
