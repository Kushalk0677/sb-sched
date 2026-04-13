# EKF Extension Notes

## How the Staleness Budget Generalises to EKF

In the linear KF, the covariance prediction is exact:

```
P_{k+τ} = F^τ P_k (F^τ)^T + accumulated_Q(τ)
```

For the EKF, `F` is replaced by the Jacobian `F_k = ∂f/∂x |_{x̂_k}` at the
current state estimate. This linearisation is only locally valid, so the budget
becomes a **conservative bound** rather than an exact guarantee:

```
P_bound_{k+τ} = (F_k)^τ P_k ((F_k)^τ)^T + accumulated_Q(τ)
```

Because EKF linearisation underestimates covariance growth under high nonlinearity,
the actual covariance may grow faster than `P_bound`. The scheduler therefore
adds a **safety margin** `δ` to the budget computation:

```python
def compute_budget_ekf(self, sensor_name, delta=0.9):
    """
    δ < 1 means schedule the sensor before trace(P_bound) reaches P_max.
    Conservative choice: schedule when P_bound reaches δ * P_max.
    """
    for k in range(1, self.max_lookahead + 1):
        P_pred = self.kf.predict_covariance(k)   # EKF uses linearised F_k
        if np.trace(P_pred) > delta * self.p_max:
            return max(1, k - 1)
    return self.max_lookahead
```

## IMU Preintegration Model (EuRoC / TUM-VI)

For the full visual-inertial state `[p, v, q, b_g, b_a]`, the nonlinear
propagation function `f(x)` is the standard IMU preintegration:

```
p_{k+1} = p_k + v_k * dt + 0.5 * (R_k * a_k - g) * dt²
v_{k+1} = v_k + (R_k * a_k - g) * dt
q_{k+1} = q_k ⊗ Δq(ω_k * dt)
b_g_{k+1} = b_g_k  (random walk)
b_a_{k+1} = b_a_k  (random walk)
```

The Jacobian `F_k = ∂f/∂x` has a closed-form expression given in:
- Forster et al., "On-Manifold Preintegration for Real-Time Visual-Inertial
  Odometry", IEEE T-RO 2017

## Experiment 6 Setup

1. Use EuRoC MH_01–MH_05 sequences
2. Build EKF with IMU preintegration model
3. Run SB-Sched with `delta=0.9` safety margin
4. Compare VR and RMSE against:
   - Linear KF + SB-Sched (same sequences)
   - Fixed-High EKF (upper bound on accuracy)
   - Heuristic EKF baseline

Expected finding: VR stays near 0 for `delta ≤ 0.95`, with slight increase
in ESR compared to linear KF (conservative budget requires slightly more samples).
