# Experimental Pipeline: Staleness-Budget Sensor Fusion Scheduling

## Overview

This document defines the full experimental pipeline for the paper:
**"Quality-Guaranteed Sensor Fusion via Staleness-Budget Scheduling"**

The core claim to validate: a scheduler that dynamically adjusts sensor sampling rates based on a formally derived staleness budget maintains a covariance bound P(t) ≤ P_max at all times, while using fewer sensor reads than fixed-rate baselines.

---

## Datasets

### Dataset 1 — EuRoC MAV (IMU + Camera, Indoor UAV)
- **Sensors:** IMU at 200 Hz, stereo camera at 20 Hz
- **Ground truth:** 6-DoF pose from Vicon motion capture (~1 mm accuracy)
- **Sequences to use:** 11 sequences across 3 difficulty levels
  - Easy: MH_01, MH_02, V1_01, V2_01
  - Medium: MH_03, MH_04, V1_02, V2_02
  - Hard: MH_05, V1_03, V2_03
- **Why:** Large rate mismatch (10:1) between IMU and camera is the canonical test case for the staleness budget. Ground truth quality is excellent.
- **Download:** https://projects.asl.ethz.ch/datasets/euroc-mav/

### Dataset 2 — KITTI Raw (GPS/IMU + LiDAR, Autonomous Driving)
- **Sensors:** IMU/GPS at 100 Hz (OXTS RT3003), Velodyne LiDAR at 10 Hz
- **Ground truth:** RTK GPS trajectory (~10 cm accuracy)
- **Sequences to use:** 10 sequences across 4 categories
  - City: 2011_09_26_drive_0001, _0002, _0005
  - Residential: 2011_09_26_drive_0019, _0020
  - Road: 2011_09_26_drive_0027, _0028
  - Campus: 2011_09_28_drive_0001, _0002
- **Why:** Different sensor pair (GPS/IMU), slower dynamics, outdoor environment, real GPS dropout events provide natural stress tests.
- **Download:** https://www.cvlibs.net/datasets/kitti/raw_data.php

### Dataset 3 — TUM-VI (IMU + Camera, Handheld)
- **Sensors:** IMU at 200 Hz, camera at 20 Hz (photometrically calibrated)
- **Ground truth:** Motion capture at 120 Hz
- **Sequences to use:** room1–6 (indoor), corridor1–5 (dynamic motion)
- **Why:** Higher dynamic range than EuRoC, photometric calibration, and longer sequences stress the scheduler under sustained high-motion conditions where covariance grows fast.
- **Download:** https://vision.in.tum.de/data/datasets/visual-inertial-dataset

---

## Baselines

| Name | Description |
|------|-------------|
| **Fixed-High** | Both sensors run at maximum rate (IMU: 200 Hz, camera/GPS/LiDAR: native max) |
| **Fixed-Low** | Both sensors run at minimum viable rate (half native rate) |
| **Fixed-Matched** | Both sensors run at the slower sensor's rate |
| **Heuristic-Adaptive** | Rate doubles if estimated error exceeds threshold, halves if below (no formal bound) |
| **Ours (SB-Sched)** | Staleness-budget scheduler — triggers measurement when τ_i* is about to be exceeded |

---

## Metrics

| Metric | Symbol | Description |
|--------|--------|-------------|
| Covariance bound violation rate | VR | % of timesteps where tr(P(t)) > P_max |
| Root Mean Square Error | RMSE | Position error vs. ground truth (m) |
| Average Effective Sample Rate | ESR | Mean Hz at which each sensor is actually sampled |
| Rate Reduction vs Fixed-High | RR | (ESR_fixed_high − ESR_ours) / ESR_fixed_high × 100% |
| Staleness at violation | SV | Measured staleness τ when P exceeds bound (for baseline analysis) |

Primary metric is **VR = 0** (the guarantee). Secondary metric is **RR** (the benefit). RMSE confirms no accuracy regression.

---

## Experiment 1 — Baseline Comparison (Core Result)

**Hypothesis:** SB-Sched maintains VR = 0 while reducing sample rate vs Fixed-High.

**Protocol:**
1. For each dataset and sequence, set P_max = 3× the steady-state covariance of Fixed-High (a reasonable quality floor)
2. Run all 5 baselines on all sequences
3. Record VR, RMSE, ESR for each
4. Compute RR for SB-Sched vs Fixed-High

**Expected result table structure:**

| Dataset | Sequence | Method | VR (%) | RMSE (m) | ESR (Hz) | RR (%) |
|---------|----------|--------|--------|----------|----------|--------|
| EuRoC | MH_01 | Fixed-High | 0 | 0.05 | 200 | — |
| EuRoC | MH_01 | Heuristic | 3.2 | 0.08 | 140 | 30% |
| EuRoC | MH_01 | SB-Sched | 0 | 0.06 | 120 | 40% |
| ... | | | | | | |

**Key claim:** SB-Sched is the only method besides Fixed-High that achieves VR = 0, but does so at ~30–50% lower sample rate.

---

## Experiment 2 — P_max Sensitivity (Budget Tightness)

**Hypothesis:** As P_max tightens, SB-Sched gracefully increases sample rate. As P_max loosens, it aggressively reduces rate.

**Protocol:**
1. Fix dataset to EuRoC MH_03 (medium difficulty)
2. Sweep P_max over 5 values: 1.5×, 2×, 3×, 5×, 10× steady-state covariance
3. Run SB-Sched and Heuristic-Adaptive at each level
4. Record ESR and VR for each

**Plot:** X-axis = P_max multiplier, Y-axis (left) = ESR, Y-axis (right) = VR (%). SB-Sched should show monotonically decreasing ESR with VR = 0 always. Heuristic should show non-zero VR at tight budgets.

---

## Experiment 3 — Sensor Rate Asymmetry (Different Sensor Pairs)

**Hypothesis:** The staleness budget is derived per-sensor, so it handles asymmetric rate pairs naturally.

**Protocol:**
1. Use KITTI GPS/IMU dataset (10:1 rate ratio) and EuRoC IMU/camera (10:1 ratio)
2. Artificially introduce a 20:1 ratio by subsampling one sensor by 2×
3. Run all baselines
4. Compare SB-Sched performance relative to baselines at 10:1 vs 20:1 ratios

**Expected finding:** Heuristic degrades noticeably at 20:1. SB-Sched VR stays 0 because the budget adapts to the rate ratio automatically.

---

## Experiment 4 — Dynamic Motion Stress Test

**Hypothesis:** Under high-dynamics (fast motion), covariance grows faster, forcing the scheduler to increase rates. SB-Sched should respond faster than the heuristic.

**Protocol:**
1. Use TUM-VI corridor sequences (aggressive handheld motion) and EuRoC hard sequences (MH_05, V1_03, V2_03)
2. Compute motion intensity metric: mean absolute angular velocity from IMU
3. Bin sequences into low/medium/high motion
4. Plot VR and RMSE vs. motion intensity for each method

**Expected finding:** At high motion, covariance grows fast. SB-Sched detects this analytically (from the F, Q matrices) and pre-emptively raises rate. Heuristic reacts late, incurring violations.

---

## Experiment 5 — Computational Overhead

**Hypothesis:** SB-Sched adds negligible runtime overhead vs. the Kalman filter itself.

**Protocol:**
1. Measure wall-clock time of scheduler decision per timestep (averaged over 10,000 timesteps)
2. Compare against: (a) Kalman predict step time, (b) Kalman update step time, (c) Heuristic decision time
3. Hardware: standard laptop CPU (document exact spec)

**Expected result:** Scheduler decision is O(n) in state dimension, dominated by matrix operations already done in Kalman predict. Overhead < 1% of total filter time.

---

## Experiment 6 — Generalisation to EKF (Nonlinear Systems)

**Hypothesis:** The budget derivation extends to EKF by using the linearised covariance propagation, with a conservative bound that still outperforms baselines.

**Protocol:**
1. Use EuRoC with an EKF instead of linear KF (the IMU preintegration model is nonlinear)
2. Derive the staleness budget using the Jacobian F_k at each timestep (EKF linearisation)
3. Compare VR and RR against the linear KF version and baselines

**Note:** This is a secondary experiment — it validates the extension section of the paper, not the core claim. If EKF results are weaker, that is still a publishable honest finding.

---

## Implementation Notes

### Software Stack
- Python 3.11
- `numpy`, `scipy` for Kalman filter and matrix ops
- `rosbag` / `rosbag2` for EuRoC and TUM-VI data loading
- `pykitti` for KITTI data loading
- `matplotlib` for all plots
- All code to be released open-source on GitHub for reproducibility

### Kalman Filter Setup (EuRoC / TUM-VI)
- State: [position (3), velocity (3), orientation quaternion (4), IMU bias (6)] = 16-dim
- High-rate sensor: IMU (predict step at 200 Hz)
- Low-rate sensor: camera feature-based update (20 Hz, or as scheduled)
- P_max: set per-sequence based on 3× steady-state trace(P) of Fixed-High run

### Kalman Filter Setup (KITTI)
- State: [position (3), velocity (3), orientation (3)] = 9-dim
- High-rate sensor: IMU at 100 Hz
- Low-rate sensor: GPS at 10 Hz, or as scheduled
- P_max: 3× steady-state trace(P) of Fixed-High run

### Staleness Budget Computation
```
Given: F (state transition), Q (process noise), P_current, P_max, R_i (sensor noise)
1. Simulate: P_k+1 = F P_k F^T + Q (predict only, no update)
2. Find τ_i* = smallest k such that trace(P_k) > P_max
3. Schedule sensor i to trigger at t_now + τ_i*
4. Repeat every timestep
```
This is O(k_max × n²) where k_max is typically 5–20 steps and n is state dim.

---

## Expected Paper Figures

| Figure | Content |
|--------|---------|
| Fig 1 | System diagram: scheduler between sensors and Kalman filter |
| Fig 2 | Exp 1: Bar chart — VR and RR across all methods and datasets |
| Fig 3 | Exp 2: Line plot — ESR vs P_max multiplier (SB-Sched vs Heuristic) |
| Fig 4 | Exp 4: Scatter — VR vs motion intensity per method |
| Fig 5 | Exp 1: RMSE comparison table (shows no accuracy regression) |
| Fig 6 (optional) | Exp 6: EKF results vs linear KF |

---

## Reproducibility Checklist

- [ ] All datasets are publicly available and free to download
- [ ] Exact sequence names listed above
- [ ] P_max derivation is deterministic given dataset
- [ ] Random seed fixed for any stochastic initialisation
- [ ] Code to be released with paper submission
- [ ] All results logged to CSV for audit

---

## Timeline Estimate

| Task | Time |
|------|------|
| Data loading + KF implementation | 2 weeks |
| Exp 1 (core) | 1 week |
| Exp 2–3 | 1 week |
| Exp 4–5 | 1 week |
| Exp 6 (EKF) | 1 week |
| Writing + figures | 2 weeks |
| **Total** | **~8 weeks** |
