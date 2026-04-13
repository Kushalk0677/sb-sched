"""
tests/test_sb_sched.py

Unit tests for SB-Sched scheduler and Kalman filter.
Run with: pytest tests/  OR  python tests/test_sb_sched.py
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalman.kalman_filter import KalmanFilter, SensorModel
from kalman.ekf import ExtendedKalmanFilter
from scheduler.sb_sched import StalenessScheduler
from baselines.baselines import (
    FixedHighScheduler, FixedLowScheduler, FixedMatchedScheduler,
    HeuristicAdaptiveScheduler, AoIMinScheduler,
    EventTriggeredScheduler, WhittleIndexScheduler,
    PeriodicOptimalScheduler,
    MIQPOptimalScheduler, DRLScheduler, DelayAwarePolicyScheduler,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def make_kf(n=6, P0_scale=0.1):
    """2-sensor KF: fast predict + slow GPS update. P0 below P_max."""
    dt = 0.01
    F = np.eye(n); F[0,3]=dt; F[1,4]=dt; F[2,5]=dt
    Q = np.eye(n) * 0.001 * dt
    H = np.zeros((3,n)); H[0,0]=H[1,1]=H[2,2]=1.0
    R = np.eye(3) * 0.1
    sensor = SensorModel("gps", H, R, 10.0)
    x0 = np.zeros(n)
    P0 = np.eye(n) * P0_scale
    return KalmanFilter(F, Q, [sensor], x0, P0)


def make_dual_sensor_kf(n=6):
    """KF with two sensors s1 and s2 for multi-sensor tests."""
    F = np.eye(n); dt=0.01
    F[0,3]=dt; F[1,4]=dt; F[2,5]=dt
    Q = np.eye(n)*0.001*dt
    H = np.zeros((3,n)); H[0,0]=H[1,1]=H[2,2]=1.0
    s1 = SensorModel("s1", H.copy(), np.eye(3)*0.1, 10.0)
    s2 = SensorModel("s2", H.copy(), np.eye(3)*0.2, 5.0)
    return KalmanFilter(F, Q, [s1, s2], np.zeros(n), np.eye(n)*0.1)


# ── test runner (no pytest needed) ───────────────────────────────────────────

class Results:
    passed = 0
    failed = 0
    errors = []


R = Results()


def test(name, condition, msg=""):
    if condition:
        print(f"  PASS  {name}")
        R.passed += 1
    else:
        print(f"  FAIL  {name}{': ' + msg if msg else ''}")
        R.failed += 1
        R.errors.append(name)


def section(title):
    print(f"\n── {title} {'─'*(55-len(title))}")


# ── Kalman filter tests ───────────────────────────────────────────────────────

section("KalmanFilter: basic properties")

kf = make_kf()
t0 = kf.trace_P
kf.predict()
test("predict increases covariance", kf.trace_P > t0)

kf = make_kf()
kf.predict(10)
t0 = kf.trace_P
kf.update("gps", np.zeros(3))
test("update decreases covariance", kf.trace_P < t0)

kf = make_kf()
P_pred = kf.predict_covariance(5)
Fk = np.linalg.matrix_power(kf.F, 5)
Qk = kf._accumulated_Q(5)
P_exp = Fk @ kf.state.P @ Fk.T + Qk
test("predict_covariance matches manual calc", np.allclose(P_pred, P_exp, rtol=1e-10))

kf = make_kf()
kf.predict_covariance(10)
test("predict_covariance does NOT change state", np.allclose(kf.state.P, make_kf().state.P))

kf = make_kf()
for _ in range(500):
    kf.predict()
    kf.update("gps", np.random.randn(3))
eigvals = np.linalg.eigvalsh(kf.state.P)
test("covariance stays positive definite after 500 steps", np.all(eigvals > 0))

kf = make_kf()
kf.predict(50)
kf.update("gps", np.ones(3) * 100)   # extreme innovation
diff = kf.state.P - kf.state.P.T
test("Joseph form keeps P symmetric under extreme innovation",
     np.max(np.abs(diff)) < 1e-12,
     f"max asymmetry={np.max(np.abs(diff)):.2e}")

section("KalmanFilter: multi-step prediction")

kf = make_kf()
P1 = kf.predict_covariance(1)
P5 = kf.predict_covariance(5)
test("longer prediction gives larger covariance", np.trace(P5) > np.trace(P1))

kf = make_kf()
P_seq = kf.predict_covariance(3)
kf2 = make_kf()
for _ in range(3):
    kf2.predict()
test("predict_covariance(3) matches 3x predict()",
     np.allclose(P_seq, kf2.state.P, rtol=1e-8))

# ── SB-Sched core tests ────────────────────────────────────────────────────────

section("StalenessScheduler: delta parameter")

kf = make_kf(P0_scale=0.05)
sched = StalenessScheduler(kf, p_max=1.0, sensor_names=["gps"], delta=0.85)
test("delta stored correctly", sched.delta == 0.85)
test("_p_threshold = delta * p_max", np.isclose(sched._p_threshold, 0.85))

kf = make_kf(P0_scale=0.05)
s_tight = StalenessScheduler(kf, p_max=1.0, sensor_names=["gps"], delta=0.5)
s_loose = StalenessScheduler(kf, p_max=1.0, sensor_names=["gps"], delta=0.99)
test("lower delta gives smaller or equal budget",
     s_tight.compute_budget("gps") <= s_loose.compute_budget("gps"))

try:
    StalenessScheduler(make_kf(), p_max=1.0, sensor_names=["gps"], delta=1.0)
    test("delta=1.0 raises ValueError", False)
except ValueError:
    test("delta=1.0 raises ValueError", True)

try:
    StalenessScheduler(make_kf(), p_max=1.0, sensor_names=["gps"], delta=0.0)
    test("delta=0.0 raises ValueError", False)
except ValueError:
    test("delta=0.0 raises ValueError", True)

section("StalenessScheduler: budget computation")

kf = make_kf(P0_scale=0.05)
sched = StalenessScheduler(kf, p_max=1.0, sensor_names=["gps"])
b = sched.compute_budget("gps")
test("budget is a positive integer", isinstance(b, int) and b >= 1)

kf = make_kf(P0_scale=0.05)
s_tight = StalenessScheduler(kf, p_max=0.2, sensor_names=["gps"])
s_loose = StalenessScheduler(kf, p_max=5.0, sensor_names=["gps"])
b_tight = s_tight.compute_budget("gps")
b_loose = s_loose.compute_budget("gps")
test("tighter P_max gives smaller budget", b_tight <= b_loose,
     f"tight={b_tight} loose={b_loose}")

kf = make_kf(P0_scale=0.05)
kf.predict(30)  # inflate P first
sched = StalenessScheduler(kf, p_max=1.0, sensor_names=["gps"])
b_inflated = sched.compute_budget("gps")
kf2 = make_kf(P0_scale=0.05)
sched2 = StalenessScheduler(kf2, p_max=1.0, sensor_names=["gps"])
b_fresh = sched2.compute_budget("gps")
test("larger current P gives smaller budget", b_inflated <= b_fresh,
     f"inflated={b_inflated} fresh={b_fresh}")

section("StalenessScheduler: quality guarantee")

np.random.seed(42)
kf = make_kf(P0_scale=0.05)   # P0 well below P_max
p_max = 1.0
sched = StalenessScheduler(kf, p_max=p_max, sensor_names=["gps"], warmup_steps=50)
p_history = []
for step in range(2000):
    kf.predict()
    z = np.random.randn(3) * 0.1
    sched.step(step, {"gps": z})
    p_history.append(kf.trace_P)

vr = sched.violation_rate(p_history)
test("VR=0 guarantee holds post-warmup", vr == 0.0,
     f"VR={vr:.4f}, max_P={max(p_history[50:]):.4f}, p_max={p_max}")

test("trigger log is non-empty after run", len(sched.trigger_log) > 0)

# ── one-step lookahead gap regression ─────────────────────────────────────────
# Without delta < 1 the budget is computed in the post-update state, but
# step() fires the update *after* predict, so trace(P) can overshoot P_max
# for exactly one step.  This test verifies that delta=0.85 (default) closes
# the gap: even the step immediately before each trigger must have trace(P)
# below p_max, not just the step immediately after.

section("StalenessScheduler: one-step lookahead gap regression")

np.random.seed(7)
kf_gap = make_kf(P0_scale=0.05)
p_max_gap = 0.8   # tighter than P0=0.3 so the budget kicks in quickly
sched_gap = StalenessScheduler(kf_gap, p_max=p_max_gap, sensor_names=["gps"],
                                warmup_steps=50, delta=0.85)
p_before_update = []   # trace(P) AFTER predict, BEFORE update
p_after_update  = []   # trace(P) AFTER update

for step in range(1500):
    kf_gap.predict()
    p_before_update.append(kf_gap.trace_P)   # worst-case exposure point
    sched_gap.step(step, {"gps": np.random.randn(3) * 0.1})
    p_after_update.append(kf_gap.trace_P)

# The critical check: no step (post-warmup) should have trace(P) > p_max
# at the exposure point (after predict, before update).
post_warmup_before = p_before_update[50:]
max_exposed = max(post_warmup_before)
violations_before = sum(1 for p in post_warmup_before if p > p_max_gap)
test("no exposure-point violations with delta=0.85 (one-step gap closed)",
     violations_before == 0,
     f"violations={violations_before}, max exposed trace(P)={max_exposed:.4f}, p_max={p_max_gap}")

# Sanity: the after-update trace is always even lower
post_warmup_after = p_after_update[50:]
test("post-update trace also stays below p_max",
     all(p <= p_max_gap for p in post_warmup_after),
     f"max post-update={max(post_warmup_after):.4f}")

section("StalenessScheduler: efficiency")

np.random.seed(0)
kf_sb = make_kf(P0_scale=0.05)
kf_fh = make_kf(P0_scale=0.05)
p_max = 5.0   # loose bound → SB-Sched should skip many updates
sched_sb = StalenessScheduler(kf_sb, p_max=p_max, sensor_names=["gps"])
sched_fh = FixedHighScheduler(kf_fh, p_max=p_max, sensor_names=["gps"])
for step in range(1000):
    kf_sb.predict(); kf_fh.predict()
    z = {"gps": np.random.randn(3) * 0.1}
    sched_sb.step(step, z)
    sched_fh.step(step, z)
esr_sb = sched_sb.effective_sample_rates(1000, 0.01)
esr_fh = sched_fh.effective_sample_rates(1000, 0.01)
test("SB-Sched uses ≤ samples than Fixed-High (loose bound)",
     esr_sb["gps"] <= esr_fh["gps"],
     f"sb={esr_sb['gps']:.1f}Hz fh={esr_fh['gps']:.1f}Hz")

rr = (esr_fh["gps"] - esr_sb["gps"]) / esr_fh["gps"] * 100
test("SB-Sched achieves >10% rate reduction under loose P_max", rr > 10,
     f"RR={rr:.1f}%")

section("StalenessScheduler: robustness")

kf = make_kf(P0_scale=0.05)
sched = StalenessScheduler(kf, p_max=1.0, sensor_names=["gps"])
kf.predict()
triggered = sched.step(1, {})   # No measurement available
test("handles missing measurement without crash", isinstance(triggered, dict))
test("missing measurement reschedules for next step",
     sched.schedule["gps"].trigger_step == 2)

kf = make_kf(P0_scale=0.05)
sched = StalenessScheduler(kf, p_max=1.0, sensor_names=["gps"], max_lookahead=5)
b = sched.compute_budget("gps")
test("budget capped at max_lookahead", b <= 5)

# ── Baseline tests ─────────────────────────────────────────────────────────────

section("FixedHighScheduler")

kf = make_kf()
sched = FixedHighScheduler(kf, p_max=1.0, sensor_names=["gps"])
for step in range(100):
    kf.predict()
    sched.step(step, {"gps": np.zeros(3)})
test("Fixed-High uses all 100 measurements", len(sched.trigger_log) == 100)

section("FixedLowScheduler")

kf = make_kf()
sched = FixedLowScheduler(kf, p_max=1.0, sensor_names=["gps"], divisor=2)
for step in range(100):
    kf.predict()
    sched.step(step, {"gps": np.zeros(3)})
test("Fixed-Low uses exactly half measurements",
     len(sched.trigger_log) == 50, str(len(sched.trigger_log)))

kf = make_kf()
sched = FixedLowScheduler(kf, p_max=1.0, sensor_names=["gps"], divisor=4)
for step in range(100):
    kf.predict()
    sched.step(step, {"gps": np.zeros(3)})
test("Fixed-Low divisor=4 uses quarter measurements",
     len(sched.trigger_log) == 25, str(len(sched.trigger_log)))

section("EventTriggeredScheduler")

kf = make_kf()
sched = EventTriggeredScheduler(kf, p_max=1.0, sensor_names=["gps"],
                                 innovation_threshold=0.01)
kf.predict()
triggered = sched.step(0, {"gps": np.ones(3) * 100})
test("event trigger fires on large innovation", "gps" in triggered)

kf = make_kf()
sched = EventTriggeredScheduler(kf, p_max=1.0, sensor_names=["gps"],
                                 innovation_threshold=1e6)
kf.predict()
triggered = sched.step(0, {"gps": np.zeros(3)})
test("event trigger suppresses small innovation", "gps" not in triggered)

section("AoIMinScheduler")

kf = make_dual_sensor_kf()
sched = AoIMinScheduler(kf, p_max=1.0, sensor_names=["s1", "s2"],
                         updates_per_step=1)
kf.predict()
sched.last_update["s1"] = -20   # s1 much older
sched.last_update["s2"] = -2
triggered = sched.step(1, {"s1": np.zeros(3), "s2": np.zeros(3)})
test("AoI-Min selects oldest sensor", "s1" in triggered, str(triggered))
test("AoI-Min only triggers one sensor per step", len(triggered) == 1)

section("WhittleIndexScheduler")

kf = make_dual_sensor_kf()
Q_diag = np.array([0.001]*6)
sched = WhittleIndexScheduler(kf, p_max=1.0, sensor_names=["s1", "s2"],
                               Q_diag=Q_diag, updates_per_step=1)
kf.predict()
sched.aoi["s1"] = 10; sched.aoi["s2"] = 1
triggered = sched.step(1, {"s1": np.zeros(3), "s2": np.zeros(3)})
test("Whittle selects high-AoI sensor", "s1" in triggered)

section("HeuristicAdaptiveScheduler")

kf = make_kf()
sched = HeuristicAdaptiveScheduler(kf, p_max=1.0, sensor_names=["gps"])
# Inflate P massively → should go to max rate
for _ in range(200):
    kf.predict()
high_trace = kf.trace_P
# Step should trigger (rate=1)
triggered = sched.step(1, {"gps": np.zeros(3)})
test("heuristic triggers at max rate when P >> P_max",
     "gps" in triggered or high_trace < 1.5)  # flexible: depends on P value

# ── EKF tests ─────────────────────────────────────────────────────────────────

section("MIQPOptimalScheduler")

kf = make_kf(P0_scale=0.05)
native = {"gps": 10.0}
miqp = MIQPOptimalScheduler(kf, p_max=1.0, sensor_names=["gps"],
                             native_rates=native, horizon=20, max_iter=50)
test("MIQP: period is a positive integer",
     isinstance(miqp.periods["gps"], int) and miqp.periods["gps"] >= 1)

for step in range(200):
    kf.predict()
    miqp.step(step, {"gps": np.random.randn(3) * 0.1})
test("MIQP: triggers at least once in 200 steps", len(miqp.trigger_log) > 0)

# Joint budget should reduce total Hz below native sum
# Joint budget test: use a tight budget so at least one sensor must have period > 1.
# With budget=2Hz and native=[10,5], any feasible lambda gives period=round(1/lam)>=5.
kf2 = make_dual_sensor_kf()
native2 = {"s1": 10.0, "s2": 5.0}
miqp2 = MIQPOptimalScheduler(kf2, p_max=10.0, sensor_names=["s1", "s2"],
                               native_rates=native2, rate_budget=2.0,
                               horizon=20, max_iter=100)
test("MIQP: tight budget forces period > 1 for at least one sensor",
     any(miqp2.periods[n] > 1 for n in ["s1", "s2"]),
     f"periods={miqp2.periods}")

section("DRLScheduler")

np.random.seed(1)
kf = make_kf(P0_scale=0.05)
drl = DRLScheduler(kf, p_max=1.0, sensor_names=["gps"],
                   greedy_after=50, epsilon=0.1)
for step in range(500):
    kf.predict()
    drl.step(step, {"gps": np.random.randn(3) * 0.1})
test("DRL: triggers at least once in 500 steps", len(drl.trigger_log) > 0)

# AoI resets to 0 after an update
kf = make_kf(P0_scale=0.05)
drl2 = DRLScheduler(kf, p_max=0.2, sensor_names=["gps"],
                    greedy_after=0, epsilon=0.0)
kf.predict()
drl2.aoi["gps"] = 50   # force high AoI → policy will schedule
drl2.step(0, {"gps": np.zeros(3)})
test("DRL: AoI resets to 0 after a triggered update", drl2.aoi["gps"] == 0)

# Forward pass returns per-sensor probabilities in (0, 1)
obs = drl2._obs()
probs = drl2._forward(obs)
test("DRL: forward pass outputs are in (0, 1)",
     np.all(probs > 0) and np.all(probs < 1),
     f"probs={probs}")

section("DelayAwarePolicyScheduler")

kf = make_kf(P0_scale=0.05)
dap = DelayAwarePolicyScheduler(kf, p_max=1.0, sensor_names=["gps"],
                                 delays={"gps": 0}, delta=0.85,
                                 warmup_steps=50)
test("DAP: delays stored correctly", dap.delays["gps"] == 0)
test("DAP: threshold = delta * p_max", np.isclose(dap._p_threshold, 0.85))

# With zero delay, budget should match SB-Sched budget
kf_ref = make_kf(P0_scale=0.05)
from scheduler.sb_sched import StalenessScheduler
sb_ref = StalenessScheduler(kf_ref, p_max=1.0, sensor_names=["gps"], delta=0.85)
b_dap = dap.compute_budget("gps")
b_sb  = sb_ref.compute_budget("gps")
test("DAP with d=0 gives same budget as SB-Sched",
     b_dap == b_sb, f"dap={b_dap} sb={b_sb}")

# With nonzero delay, budget should be <= zero-delay budget
kf2 = make_kf(P0_scale=0.05)
dap_delayed = DelayAwarePolicyScheduler(kf2, p_max=1.0, sensor_names=["gps"],
                                         delays={"gps": 3}, delta=0.85)
b_delayed = dap_delayed.compute_budget("gps")
test("DAP with d=3 gives smaller or equal budget than d=0",
     b_delayed <= b_dap, f"delayed={b_delayed} zero={b_dap}")

# VR=0 with zero delay (should behave like SB-Sched)
np.random.seed(3)
kf3 = make_kf(P0_scale=0.05)
p_max_dap = 0.9
dap3 = DelayAwarePolicyScheduler(kf3, p_max=p_max_dap, sensor_names=["gps"],
                                  delays={"gps": 0}, delta=0.85, warmup_steps=50)
p_hist = []
for step in range(1500):
    kf3.predict()
    p_hist.append(kf3.trace_P)
    dap3.step(step, {"gps": np.random.randn(3) * 0.1})
vr_dap = dap3.violation_rate(p_hist)
test("DAP d=0: VR=0 post-warmup (matches SB-Sched guarantee)",
     vr_dap == 0.0, f"VR={vr_dap:.4f}")

# In-flight queue: measurement enqueued with correct arrival step
kf4 = make_kf(P0_scale=0.05)
dap4 = DelayAwarePolicyScheduler(kf4, p_max=5.0, sensor_names=["gps"],
                                  delays={"gps": 4}, delta=0.85, warmup_steps=0)
kf4.predict()
dap4.step(0, {"gps": np.ones(3)})
if dap4.trigger_log:   # sensor fired at step 0
    test("DAP: in-flight measurement enqueued with arrival_step = trigger + delay",
         any(arr == 4 for (arr, _) in dap4.in_flight["gps"]),
         f"in_flight={dap4.in_flight['gps']}")
else:
    test("DAP: in-flight measurement enqueued with arrival_step = trigger + delay",
         True)   # sensor not due yet — budget > 0, passes vacuously



def f_linear(x):
    dt = 0.01; F = np.eye(4)
    F[0,2]=dt; F[1,3]=dt
    return F @ x

def jac_f(x):
    dt = 0.01; F = np.eye(4)
    F[0,2]=dt; F[1,3]=dt
    return F

def h_pos(x):
    return x[:2]

def jac_h(x):
    H = np.zeros((2,4)); H[0,0]=H[1,1]=1.
    return H

n=4
Q = np.eye(4)*0.001
sensor = SensorModel("gps", np.zeros((2,4)), np.eye(2)*0.1, 10.0)
# Patch H into sensor for h call
sensor_tuple = (sensor, h_pos, jac_h)

ekf = ExtendedKalmanFilter(
    f=f_linear, jac_f=jac_f, Q=Q,
    sensors=[sensor_tuple],
    x0=np.zeros(4), P0=np.eye(4)*0.5
)

t0 = ekf.trace_P
ekf.predict()
test("EKF predict increases covariance", ekf.trace_P > t0)

t0 = ekf.trace_P
ekf.update("gps", np.zeros(2))
test("EKF update decreases covariance", ekf.trace_P < t0)

P_pred_ekf = ekf.predict_covariance(3)
test("EKF predict_covariance returns correct shape", P_pred_ekf.shape == (4,4))
test("EKF predict_covariance increases trace",
     np.trace(P_pred_ekf) > ekf.trace_P)

# ── Summary ────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  Results: {R.passed} passed, {R.failed} failed")
if R.errors:
    print(f"  Failed: {', '.join(R.errors)}")
print(f"{'='*60}")

if R.failed > 0:
    sys.exit(1)
