# Formal Guarantee for SB-Sched in the Linear Kalman Filter Setting

## Purpose

This note provides a formal statement and proof of the quality guarantee used by **SB-Sched** in the **linear Kalman filter** setting.

The goal is to make precise the claim that, under the assumptions of the repository's core linear model, SB-Sched can maintain the estimator covariance below a prescribed budget after warmup.

This is intentionally written as a standalone technical note that can later be adapted into a paper appendix or theorem/proof subsection.

---

## 1. Setting

Consider the discrete-time linear Gaussian system

```math
x_{k+1} = A x_k + w_k,\qquad w_k \sim \mathcal{N}(0,Q), \quad Q \succeq 0
```

with a measurement from a given sensor

```math
y_k = H x_k + v_k,\qquad v_k \sim \mathcal{N}(0,R), \quad R \succ 0.
```

Let the Kalman filter prior covariance at time `k` be denoted by

```math
P_k^- \in \mathbb{S}_+^n
```

and the posterior covariance after a successful update by

```math
P_k^+ \in \mathbb{S}_+^n.
```

The Kalman covariance recursions are

```math
P_{k+1}^- = A P_k^+ A^\top + Q
```

and

```math
P_k^+ = P_k^- - P_k^- H^\top (H P_k^- H^\top + R)^{-1} H P_k^-.
```

We use the scalar quality metric

```math
J(P) := \operatorname{tr}(P).
```

Let `P_max > 0` be the prescribed covariance-trace budget.

---

## 2. Scheduler model

At any successful update time `t`, SB-Sched computes the exact open-loop future prior covariance sequence induced by the same Kalman prediction model used online:

```math
P_{t+h}^- = A^h P_t^+ (A^\top)^h + \sum_{i=0}^{h-1} A^i Q (A^\top)^i,\qquad h \ge 1.
```

Define the first predicted budget-crossing time

```math
h^\star := \min \{ h \ge 1 : \operatorname{tr}(P_{t+h}^-) > \delta P_{\max} \},
```

where `0 < \delta \le 1` is the safety margin used in SB-Sched.

The scheduler then places the next trigger at

```math
\tau := t + h^\star - 1.
```

Equivalently: trigger **one step before** the first predicted violation of the tightened threshold `\delta P_max`.

After a successful measurement arrives at `\tau`, the Kalman update is performed, and the procedure is repeated.

---

## 3. Assumptions

The guarantee below is exact under the following assumptions.

### A1. Exact linear covariance model
The online predictor and the scheduler use the same linear Kalman covariance recursion.

### A2. Successful scheduled measurement arrival
Whenever SB-Sched schedules an update at time `\tau`, the measurement is available and the update is successfully executed at `\tau`.

### A3. Budget feasibility at initialization / warmup
There exists at least one successful update after which the posterior covariance satisfies

```math
\operatorname{tr}(P_t^+) \le \delta P_{\max}.
```

This is the start of the post-warmup regime.

### A4. Scalar budget is defined on trace
The quality constraint is the scalar inequality

```math
\operatorname{tr}(P_k) \le P_{\max}.
```

---

## 4. Preliminary lemmas

### Lemma 1. Kalman update never increases covariance

For any prior covariance `P^- \succeq 0`, the posterior covariance after measurement update satisfies

```math
P^+ \preceq P^-.
```

Hence

```math
\operatorname{tr}(P^+) \le \operatorname{tr}(P^-).
```

#### Proof

Using the standard covariance update,

```math
P^+ = P^- - P^- H^\top (H P^- H^\top + R)^{-1} H P^-.
```

Let

```math
S := H P^- H^\top + R.
```

Since `R \succ 0` and `P^- \succeq 0`, we have `S \succ 0`. Therefore

```math
P^- H^\top S^{-1} H P^- \succeq 0.
```

So

```math
P^+ = P^- - \underbrace{P^- H^\top S^{-1} H P^-}_{\succeq 0} \preceq P^-.
```

Taking traces preserves the inequality. ∎

---

### Lemma 2. No tightened-threshold violation before the scheduled update

Let `h^\star` be defined as

```math
h^\star := \min \{ h \ge 1 : \operatorname{tr}(P_{t+h}^-) > \delta P_{\max} \}.
```

Then for every integer `j` with `1 \le j \le h^\star - 1`,

```math
\operatorname{tr}(P_{t+j}^-) \le \delta P_{\max}.
```

#### Proof

This is immediate from the minimality of `h^\star`. If some `j < h^\star` satisfied

```math
\operatorname{tr}(P_{t+j}^-) > \delta P_{\max},
```

then `h^\star` would not be the first such index. ∎

---

## 5. Main theorem

## Theorem 1. Post-warmup trace-budget guarantee for SB-Sched

Assume A1-A4. Suppose at some successful update time `t_0` we have

```math
\operatorname{tr}(P_{t_0}^+) \le \delta P_{\max}
```

with `0 < \delta \le 1`.

If, after every successful update, SB-Sched schedules the next update one step before the first predicted crossing of `\delta P_{\max}`, then for all subsequent times in the post-warmup regime,

```math
\operatorname{tr}(P_k) \le P_{\max}.
```

More strongly, for every predicted prior before the next scheduled update,

```math
\operatorname{tr}(P_k^-) \le \delta P_{\max} \le P_{\max},
```

and for every posterior immediately after a successful scheduled update,

```math
\operatorname{tr}(P_k^+) \le \delta P_{\max} \le P_{\max}.
```

### Proof

We prove the claim cycle by cycle.

Let `t` be any successful update time in the post-warmup regime such that

```math
\operatorname{tr}(P_t^+) \le \delta P_{\max}.
```

SB-Sched computes the first future predicted prior time at which the tightened threshold would be exceeded:

```math
h^\star := \min \{ h \ge 1 : \operatorname{tr}(P_{t+h}^-) > \delta P_{\max} \}.
```

It then schedules the next update at

```math
\tau = t + h^\star - 1.
```

By Lemma 2, for every intermediate prior time `k = t+1, t+2, ..., \tau`,

```math
\operatorname{tr}(P_k^-) \le \delta P_{\max}.
```

Since `\delta \le 1`, it follows that

```math
\operatorname{tr}(P_k^-) \le P_{\max}.
```

At the scheduled update time `\tau`, assumption A2 guarantees that the measurement arrives and the Kalman update is executed. By Lemma 1,

```math
P_\tau^+ \preceq P_\tau^-,
```

hence

```math
\operatorname{tr}(P_\tau^+) \le \operatorname{tr}(P_\tau^-) \le \delta P_{\max} \le P_{\max}.
```

Therefore, after the scheduled update, the posterior covariance is again within the tightened threshold, so the same argument may be repeated from time `\tau`.

By induction over successive scheduling cycles, the inequality

```math
\operatorname{tr}(P_k) \le P_{\max}
```

holds for all times after the first successful warmup update `t_0`. ∎

---

## 6. Interpretation

The theorem is stronger than a mere empirical observation:

- SB-Sched does **not** wait for the budget to be violated.
- It predicts the first future violation of the tightened threshold.
- It updates **one step earlier**.
- The Kalman update can only reduce covariance relative to the prior.
- Therefore the trace stays bounded for all post-warmup times, provided the scheduled measurements arrive successfully.

This is exactly why the implementation uses:
1. a future covariance prediction,
2. a first-crossing search,
3. a one-step-early trigger, and
4. a safety margin `\delta`.

---

## 7. Why the safety margin `\delta` appears

In exact arithmetic, under assumptions A1-A2, the theorem already holds with `\delta = 1` if the scheduler triggers one step before the first predicted crossing of `P_max`.

In practice, using `\delta < 1` is still useful because it provides robustness against:

- finite-precision numerical effects,
- mild model mismatch,
- implementation discretization details,
- off-by-one scheduling mistakes,
- approximate extensions such as EKF.

Thus:

- `\delta = 1` is the ideal exact-model boundary case;
- `\delta < 1` is the practical robust version used in the repository.

---

## 8. Corollary for the implementation claim

### Corollary 1. Zero post-warmup violation rate in the ideal linear setting

Under A1-A4, if SB-Sched is run on the exact linear covariance model with successful scheduled measurement arrivals, then after warmup the empirical violation rate

```math
\frac{\#\{k : \operatorname{tr}(P_k) > P_{\max}\}}{\#\{k \text{ in post-warmup window}\}}
```

is exactly zero.

### Proof

This follows immediately from Theorem 1, since no post-warmup time index can violate the trace budget. ∎

---

## 9. Important limitation of the proof

This proof is exact only for the repository's **linear KF setting** with exact future covariance prediction and successful scheduled arrivals.

It does **not** automatically extend unchanged to:

- EKF with repeated local linearization error,
- delayed updates,
- dropped packets,
- asynchronous arrival uncertainty,
- mismatched or learned dynamics models,
- budgets defined on other functionals not used in scheduling.

Those cases require either:
- a modified theorem, or
- a conservative approximation argument.

---

## 10. Suggested short theorem statement for the paper

If a shorter version is needed inside the paper, the following compact statement is suitable.

> **Theorem.** Consider a linear Gaussian system with Kalman covariance recursion and a scalar trace budget `P_max`. Suppose SB-Sched computes the exact future prior covariance from the current posterior covariance and schedules the next update one step before the first future time at which the predicted trace exceeds `\delta P_max`, with `0 < \delta \le 1`. If the scheduled measurement arrives successfully, then after warmup the covariance trace remains bounded by `P_max` at all subsequent times.

A short proof sketch can then be:

> **Proof sketch.** By construction, every predicted prior before the scheduled update has trace at most `\delta P_max`. The measurement update can only decrease covariance relative to the prior, so the posterior at the scheduled time also has trace at most `\delta P_max`. Induction over scheduling cycles yields the result.

---

## 11. Practical wording for the manuscript

A careful manuscript claim would be:

> “For the linear Kalman filter setting studied here, SB-Sched admits an exact post-warmup trace-budget guarantee under successful scheduled measurement arrivals. The guarantee follows because the scheduler triggers one step before the first predicted crossing of the tightened budget, and the Kalman update is covariance-reducing. The EKF case is treated as a conservative extension rather than an exact guarantee.”

That wording is strong, accurate, and reviewer-safe.

---
