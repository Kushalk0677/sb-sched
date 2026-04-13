# SB-Sched: Staleness-Budget Sensor Fusion Scheduling

Official implementation for the paper:
**"Quality-Guaranteed Sensor Fusion via Staleness-Budget Scheduling"**

## Overview

SB-Sched is a dynamic sensor sampling rate scheduler that maintains a formal
covariance bound `P(t) ≤ P_max` at all times, using provably fewer sensor reads
than any fixed-rate baseline. For each sensor, it analytically derives the maximum
time that can elapse before the Kalman filter error covariance exceeds a
user-specified quality threshold — the *staleness budget* `τ*_i`. It then
schedules each sensor to trigger exactly when its budget expires.

A safety margin `δ` (default `0.85`) is applied internally so the scheduler
triggers at `δ·P_max` rather than `P_max`, closing the one-step lookahead gap
that arises because the update fires after a predict step has already run.

```
Traditional approach:  fixed rates  →  unknown covariance quality
SB-Sched:             quality bound  →  minimum required sampling rates
```

## Installation

```bash
git clone https://github.com/yourname/sb_sched
cd sb_sched
pip install -e .
```

**Requirements:** Python 3.11+, numpy, scipy, matplotlib, pykitti, rosbags

```bash
pip install numpy scipy matplotlib pykitti rosbags tqdm pandas
```

## Datasets

| Dataset | Sensors | Platform | Download |
|---------|---------|----------|----------|
| EuRoC MAV | IMU (200 Hz) + Camera (20 Hz) | Indoor UAV | https://projects.asl.ethz.ch/datasets/euroc-mav/ |
| KITTI Raw | IMU/GPS (100 Hz) + LiDAR (10 Hz) | Autonomous car | https://www.cvlibs.net/datasets/kitti/raw_data.php |
| TUM-VI | IMU (200 Hz) + Camera (20 Hz) | Handheld | https://vision.in.tum.de/data/datasets/visual-inertial-dataset |

After downloading, set paths in `configs/datasets.yaml`.

## Quick Start

```bash
# Run core comparison experiment (Experiment 1)
python scripts/run_experiment.py --exp 1 --dataset euroc --sequence MH_01_easy

# Run all experiments on all datasets
python scripts/run_all.py

# Plot results
python scripts/plot_results.py --exp 1
```

## Baselines

| ID | Name | Description |
|----|------|-------------|
| `fixed_high` | Fixed-High | All sensors at maximum native rate |
| `fixed_low` | Fixed-Low | All sensors at half native rate |
| `fixed_matched` | Fixed-Matched | All sensors at slowest sensor rate |
| `heuristic` | Heuristic-Adaptive | Double rate if error > threshold, halve if below |
| `aoi_min` | AoI-Minimiser | Minimises Age-of-Information (no covariance bound) |
| `whittle` | Whittle Index | Whittle index policy for correlated sources |
| `periodic_opt` | Periodic-Optimal | Best fixed periodic schedule found by grid search |
| `event_trigger` | Event-Triggered | Triggers on innovation exceeding threshold |
| `cd_kf` | CD-KF Gradient | Continuous-discrete KF with gradient-optimised rates |
| `miqp_opt` | MIQP-Optimal | MIQP with joint rate budget (Dutta 2023) |
| `drl_sched` | DRL Scheduler | Deep RL policy with online TD fine-tuning (Alali 2024) |
| `delay_aware` | Delay-Aware Policy | Delay-aware staleness budget; d=0 reduces to SB-Sched (arXiv Jan 2026) |
| `sb_sched` | **SB-Sched (ours)** | Staleness-budget scheduler — our method |

## Repository Structure

```
sb_sched/
├── src/
│   ├── scheduler/          # SB-Sched core algorithm
│   ├── kalman/             # Kalman filter (linear + EKF)
│   ├── baselines/          # All 12 baseline implementations
│   ├── datasets/           # Data loaders for EuRoC, KITTI, TUM-VI
│   ├── experiments/        # Experiment runners (Exp 1–6)
│   └── utils/              # Metrics, plotting, logging
├── tests/                  # Unit tests
├── configs/                # Dataset paths and experiment configs
├── results/                # Output CSVs and figures
├── scripts/                # CLI entry points
└── docs/                   # Additional documentation
```

## Experiments

| # | Name | Key result |
|---|------|-----------|
| 1 | Baseline comparison | SB-Sched is only method with VR=0 and RR>0 |
| 2 | P_max sensitivity | ESR scales monotonically with budget tightness |
| 3 | Rate asymmetry | Handles 20:1 rate mismatch; baselines fail |
| 4 | Dynamic motion stress | SB-Sched pre-empts; event-triggered reacts late |
| 5 | Computational overhead | <1% runtime overhead vs Kalman filter |
| 6 | EKF generalisation | Conservative bound still outperforms baselines |

## Citation

```bibtex
@article{yourname2025sbsched,
  title={Quality-Guaranteed Sensor Fusion via Staleness-Budget Scheduling},
  author={Your Name},
  journal={IEEE Signal Processing Letters},
  year={2025}
}
```

## License

MIT License
