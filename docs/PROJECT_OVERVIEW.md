# SB-Sched: Covariance-Aware Sensor Scheduling for State Estimation

## 📌 Overview

This repository implements **SB-Sched (Staleness-Budget Scheduling)** — a principled framework for determining *when to trigger sensor updates* in state estimation systems.

Rather than focusing on full perception or sensor fusion pipelines, this project isolates the **scheduling problem within estimation systems**, enabling controlled and reproducible evaluation of update timing strategies.

The core idea is simple but powerful:

> Trigger sensor updates *only when necessary*, based on predicted growth in estimation uncertainty.

---

## 🧠 Problem Statement

Modern sensing systems combine heterogeneous sensors:
- Different update rates (e.g., IMU vs camera vs GPS)
- Different computational costs
- Different contributions to estimation accuracy

While significant work has focused on improving **sensor fusion algorithms**, the question of:

> **“When should a sensor be used?”**

remains underexplored.

Naïve approaches include:
- Fixed-rate sampling (wasteful)
- Heuristic adaptive policies (no guarantees)
- Event-triggered updates (reactive, often late)

---

## 🚀 Key Idea: Staleness Budget

SB-Sched introduces a **staleness budget**:

- Define a maximum allowable uncertainty: `P_max`
- Predict how covariance grows over time
- Trigger a sensor update *just before* the budget is violated

This results in:
- **Guaranteed bounded uncertainty** (post-warmup)
- **Reduced sensor usage**
- **Efficient computation**

---

## 🔬 Scope and Positioning

This project focuses specifically on:

> **Scheduling within state estimation systems**

It does **not** attempt to model:
- Full visual pipelines (e.g., feature extraction, SLAM)
- Raw sensor signal processing
- End-to-end perception systems

Instead, it isolates the scheduling layer to study:
- Update timing
- Uncertainty growth
- Trade-offs between accuracy and efficiency

---

## 📊 Datasets and Evaluation Strategy

We use standard robotics datasets:

- EuRoC MAV
- KITTI Odometry
- TUM-VI

### Important Note on Modeling

To isolate scheduling effects:

- **EuRoC and TUM-VI** use ground-truth–interpolated pseudo-measurements  
  → removes perception noise and isolates timing effects  

- **KITTI** uses real GPS measurements  
  → provides physically grounded validation  

This design ensures:
- Controlled experiments  
- Fair comparison of scheduling strategies  
- No confounding from front-end perception quality  

---

## ⚙️ Methodology

SB-Sched operates as follows:

1. Predict future covariance growth using system dynamics
2. Compute the earliest step where uncertainty exceeds `P_max`
3. Schedule the update just before violation (with safety margin `δ`)
4. Repeat for each sensor independently

---

## 📈 Experiments

The repository includes multiple experiments:

### Experiment 1 — Baseline Comparison
Compare SB-Sched against:
- Fixed-rate sampling
- Heuristic adaptive scheduling
- Event-triggered policies

### Experiment 2 — Sensitivity Analysis
Evaluate robustness to different uncertainty budgets (`P_max`)

### Experiment 3 — Rate Asymmetry
Test behavior under heterogeneous sensor rates

### Experiment 4 — Motion Stress
Evaluate under low / medium / high motion scenarios

### Experiment 5 — Computational Overhead
Measure runtime cost of scheduling decisions

### Experiment 6 — EKF Extension
Preliminary extension to nonlinear systems using EKF

---

## 🧩 Key Contributions

- A **covariance-aware scheduling framework** for sensor updates  
- A **staleness-budget formulation** that bounds estimation uncertainty  
- A **predictive triggering mechanism** (not reactive)  
- A **controlled evaluation methodology** isolating scheduling effects  
- Empirical validation showing improved efficiency over standard baselines  

---

## ⚠️ Limitations

- Does not model full perception pipelines  
- Simplified measurement models for controlled evaluation  
- EKF extension is preliminary (not full nonlinear fusion stack)  

Future work includes:
- Integration with real sensor pipelines (e.g., VIO, SLAM)
- Hardware deployment
- Multi-agent / distributed sensing systems

---

## 🏁 Summary

This project answers a fundamental question:

> **How can we use sensors only when needed, while guaranteeing estimation quality?**

SB-Sched provides a principled and efficient solution, bridging:
- estimation theory  
- scheduling  
- resource-aware sensing  

---

## 📂 Repository Structure

```
src/
  kalman/        → KF and EKF implementations
  scheduler/     → SB-Sched core logic
  baselines/     → comparison methods
  datasets/      → dataset loaders and models
  experiments/   → experiment pipelines
  utils/         → metrics and plotting

scripts/
  run_experiment.py
  run_all.py
  demo.py

results/
  experiment outputs and plots

tests/
  unit and regression tests
```

---

## ▶️ Quick Start

```bash
pip install -e .
pip install pyproj

python scripts/run_all.py
```

---

## 📌 Final Note

This repository is designed as a **research testbed** to study scheduling in isolation.

It is not a full production sensor system — and intentionally so.

This separation allows clearer insights into:
- when to sample
- how often to update
- how to balance accuracy vs efficiency
