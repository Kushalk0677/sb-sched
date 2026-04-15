# SB-Sched: Covariance-Bounded Adaptive Update Scheduling for State Estimators

SB-Sched is a covariance-aware scheduling framework for Kalman-filter-based state estimation. Instead of consuming every available low-rate measurement on a fixed timetable, it predicts covariance growth and triggers updates only when they are needed to stay within a user-defined uncertainty budget.

This repository contains the scheduler, baseline policies, dataset loaders, experiment runners, saved example results, and supporting documentation used to study **measurement-efficiency versus estimation-quality tradeoffs**.

## What this repository is for

This codebase is best understood as a **scheduling and systems testbed**, not a full visual-inertial or SLAM stack.

The main question it answers is:

> Given a high-rate prediction loop and lower-rate measurements, when should the estimator consume a measurement to maintain a covariance budget while minimising sensing cost?

That is the contribution the code supports.

## Important framing

This repository does **not** implement end-to-end sensor fusion in the strongest robotics sense.

- For **EuRoC** and **TUM-VI**, the low-rate `camera` updates are oracle-style pseudo-measurements derived from ground-truth interpolation. They are used to isolate scheduler behaviour from front-end perception noise.
- For **EuRoC** and **TUM-VI**, IMU values primarily drive the prediction cadence inside a simplified estimator testbed rather than a full visual-inertial front-end.
- For **KITTI Raw**, the measurement stream is physically grounded GPS/pose-derived data, but evaluation should still be read as estimator-consistency and scheduling behaviour rather than a claim to outperform production navigation pipelines.

So the correct reading of the results is:

- strong evidence about **covariance-aware adaptive scheduling**
- useful evidence about **resource-constrained estimation behaviour**
- not a claim to replace a full SLAM, VIO, or production fusion system

A fuller explanation is in `docs/FRAMING_AND_LIMITATIONS.md`.

## Repository contents

- SB-Sched scheduler implementation
- Fixed, heuristic, event-triggered, and delay-aware baselines
- Linear KF path and EKF extension path
- Dataset loaders for EuRoC, KITTI Raw, and TUM-VI
- Experiment runners for Exp 1–11
- Plotting utilities and saved result artifacts
- Unit tests for scheduler and core mechanics
- Supporting documentation under `docs/`

## Experiments included

### Exp 1–6
These are the original baseline, sensitivity, asymmetry, motion-stress, overhead, and EKF-extension experiments.

### Exp 7 — Outage robustness
Tests behaviour under random drops and burst blackouts.

### Exp 8 — Latency and jitter robustness
Tests delayed measurements, jitter, and combined delay-plus-dropout stress cases.

### Exp 9 — Resource-budgeted scheduling
Tests quality preservation under hard measurement budgets.

### Exp 10 — Motion regime transitions
Segments the trajectory into low / medium / high motion regimes and measures trigger behaviour and transition lag.

### Exp 11 — Pareto analysis
Builds Pareto frontiers over sensing rate, estimation error, and constraint-violation severity for the newer experiment families.

## What the results support best

The strongest supported claims in this repository are:

1. **SB-Sched can reduce effective low-rate measurement usage substantially** relative to dense or fixed-rate sensing.
2. **Under tight but feasible sensing budgets, SB-Sched can allocate updates more effectively than several fixed and reactive baselines.**
3. **SB-Sched often lies on the Pareto frontier** of sensing rate versus quality-loss metrics, especially in budgeted settings.
4. The framework is most compelling when framed as **adaptive update scheduling under uncertainty and resource constraints**.

The repository does **not** support broad claims such as:

- “best end-to-end sensor fusion system”
- “best overall state-estimation accuracy”
- “full visual-inertial odometry benchmark”
- “provably fewer reads than any fixed-rate policy”

## Installation

```bash
pip install -e .
pip install -r requirements.txt
```

## Configuration

Dataset roots and experiment settings live in:

```text
configs/config.yaml
```

Example:

```yaml
datasets:
  euroc:
    root: "D:/data/euroc"
  kitti:
    root: "D:/data/kitti_raw"
  tumvi:
    root: "D:/data/tumvi"

experiments:
  p_max_multiplier: 3.0
  p_max_sweep: [2.0, 3.0, 4.0, 5.0]
```

## Running experiments

Single experiment:

```bash
python scripts/run_experiment.py --exp 1
python scripts/run_experiment.py --exp 7
python scripts/run_experiment.py --exp 11
```

Full pipeline:

```bash
python scripts/run_all.py
```

Generate plots from saved CSVs:

```bash
python scripts/plot_results.py --all
```

Run tests:

```bash
pytest tests/ -v
```

## Output locations

Results are written under:

```text
results/exp1/
results/exp2/
...
results/exp11/
```

## Suggested interpretation for readers and reviewers

If you are using this repository for a paper, project report, or portfolio, the safest and most accurate framing is:

> SB-Sched is a covariance-bounded adaptive update scheduler for Kalman-style estimators, evaluated in controlled multi-rate estimation testbeds to study sensing-efficiency and quality tradeoffs.

That wording matches what the code actually demonstrates.

## Documentation

See:

- `docs/PROJECT_OVERVIEW.md`
- `docs/SB_Sched_Formal_Guarantee.md`
- `docs/experimental_pipeline.md`
- `docs/ekf_extension.md`
- `docs/FRAMING_AND_LIMITATIONS.md`

## License

MIT License
