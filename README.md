# SB-Sched: Staleness-Budget Sensor Fusion Scheduling

SB-Sched is a covariance-aware sensor scheduling framework for Kalman-filter-based state estimation. Instead of sampling low-rate sensors on a fixed timetable, it predicts when the estimation covariance will approach a user-defined quality budget and triggers measurements only when they are needed.

This repository contains the full codebase, dataset loaders, experiment runners, tests, generated result files, and supporting documentation for the final project version.

## What the repository includes

- SB-Sched core scheduler and baseline implementations
- Linear KF and EKF experiment paths
- Dataset loaders for EuRoC, KITTI Raw, and TUM-VI
- Reproducible experiment scripts for Exp 1–6
- Saved CSV results and figures under `results/`
- Supporting notes and proof writeups under `docs/`

## Main result in one paragraph

Across the final experimental pipeline, SB-Sched consistently keeps the covariance violation rate at or near zero while reducing measurement usage sharply relative to dense fixed-rate sensing. In the main baseline comparison, SB-Sched achieves an average measurement-rate reduction of about **91.2%** versus Fixed-High while maintaining **0.000 mean violation rate** across the aggregated Exp 1 table. The tradeoff is accuracy: SB-Sched is strongest as an efficiency-and-safety scheduler, not as the lowest-RMSE method. Methods such as Fixed-High, Whittle, and AoI-Min can be more accurate, but they do not deliver the same sensing-efficiency profile in the final comparisons.

## Final results snapshot

### Experiment 1 — Baseline comparison

Source: `results/exp1/exp1_results.csv`

Key takeaways from the final aggregated table:

- **SB-Sched**: mean VR `0.0000`, mean ESR `1.6395`, mean rate reduction `91.1842%`
- **Fixed-High**: mean VR `0.0000`, mean ESR `16.0249`, best overall accuracy among the simple baselines
- **Whittle**: mean VR `0.0000`, mean ESR `16.0249`, accuracy essentially matching Fixed-High in this testbed
- **AoI-Min**: mean VR `0.0000`, mean ESR `12.4274`, mean rate reduction `35.3022%`, strong accuracy-efficiency balance
- **Event-Trigger**: mean VR `0.0000`, mean ESR `2.9592`, mean rate reduction `82.8927%`
- **Periodic-Opt**: mean VR `0.0076`, the main remaining nonzero-VR classical baseline in Exp 1
- **DRL-Sched**: mean VR `0.3721`, clearly not reliable in the final setup

Dataset-level SB-Sched summary:

- **EuRoC**: mean VR `0.0000`, mean RMSE `0.3849`, mean ESR `2.4923`, rate reduction `87.2861%`
- **KITTI**: mean VR `0.0000`, mean RMSE `2.7005`, mean ESR `0.2698`, rate reduction `97.3017%`
- **TUM-VI**: mean VR `0.0000`, mean RMSE `0.3584`, mean ESR `2.5027`, rate reduction `87.5252%`

Interpretation: SB-Sched is best understood as a **safe sensing-reduction method**. It is not the lowest-RMSE method in this repository, but it is the strongest result if the objective is aggressive measurement reduction without covariance-budget violations.

### Experiment 2 — `P_max` sensitivity

Source: `results/exp2/exp2_results.csv`

SB-Sched remains at `VR = 0.0` for all tested budget multipliers while smoothly reducing sensor usage as the budget loosens:

- `1.5x`: ESR `10.0045`, RMSE `0.3423`
- `2.0x`: ESR `4.9948`, RMSE `0.4334`
- `3.0x`: ESR `2.4900`, RMSE `0.5612`
- `5.0x`: ESR `1.3305`, RMSE `0.7202`
- `10.0x`: ESR `0.9960`, RMSE `0.8407`

This is the cleanest illustration of the intended SB-Sched tradeoff: looser quality budgets produce fewer updates and higher estimation error, but the violation rate stays controlled.

### Experiment 3 — Rate asymmetry

Source: `results/exp3/exp3_results.csv`

SB-Sched stays effectively safe even under stronger sensor-rate asymmetry:

- `10:1`: VR `0.0000`, RMSE `0.5612`, ESR `2.4900`
- `20:1`: VR `0.0000`, RMSE `0.5416`, ESR `2.4900`

In the same experiment, Event-Trigger shows a small nonzero VR at `20:1` (`0.0006`), while SB-Sched remains at zero.

### Experiment 4 — Motion stress

Source: `results/exp4/exp4_results.csv`

All listed methods in Exp 4 finish with zero mean VR, and SB-Sched continues to use the fewest measurements among the compared schedulers:

- **SB-Sched**: mean VR `0.0000`, mean RMSE `0.4019`, mean ESR `2.4963`
- **Event-Trigger**: mean VR `0.0000`, mean RMSE `0.2259`, mean ESR `4.3871`
- **Heuristic**: mean VR `0.0000`, mean RMSE `0.2895`, mean ESR `6.6568`

The result again supports the same conclusion: SB-Sched trades accuracy for lower sensing frequency while keeping the covariance budget intact.

### Experiment 5 — Computational overhead

Source: `results/exp5/exp5_results.csv`

Measured timings:

- KF predict: `13.84 µs`
- KF update: `24.61 µs`
- SB-Sched idle step: `0.19 µs`
- SB-Sched budget computation: `15.91 µs`
- Heuristic idle step: `2.65 µs`

The idle scheduling overhead is tiny. The budget computation itself is meaningful work, but still in the same microsecond regime as the filter operations.

### Experiment 6 — EKF generalisation

Source: `results/exp6/exp6_results.csv`

All EKF runs in the saved final table have zero mean VR:

- **Fixed-High**: mean RMSE `0.1724`, mean ESR `19.9986`
- **Heuristic**: mean RMSE `0.2842`, mean ESR `9.9993`
- **SB-Sched**: mean RMSE `0.3237`, mean ESR `8.1347`

This shows the EKF path preserves the main scheduling story: SB-Sched reduces measurement rate substantially while staying within the covariance budget.

## Repository structure

```text
sb_sched/
├── configs/                 # Dataset and experiment configuration
├── docs/                    # Overview, formal guarantee, EKF notes, experimental pipeline
├── results/                 # Final CSV outputs and generated figures
├── scripts/                 # CLI entry points for experiments and plotting
├── src/
│   ├── baselines/           # SB-Sched and comparison baselines
│   ├── datasets/            # EuRoC, KITTI, TUM-VI loaders
│   ├── experiments/         # Experiment runners (Exp 1–6)
│   ├── kalman/              # Linear KF and EKF code
│   ├── scheduler/           # Scheduling logic
│   └── utils/               # Metrics and plotting helpers
├── tests/                   # Unit tests
├── requirements.txt
└── setup.py
```

## Documentation

Supporting project writeups are in `docs/`:

- `docs/PROJECT_OVERVIEW.md`
- `docs/SB_Sched_Formal_Guarantee.md`
- `docs/experimental_pipeline.md`
- `docs/ekf_extension.md`

## Installation

```bash
pip install -e .
pip install -r requirements.txt
```

## Dataset paths

Dataset roots are configured in `configs/config.yaml`.

Current local setup in this repo:

```yaml
datasets:
  euroc:
    root: "D:/data/euroc"
  kitti:
    root: "D:/data/kitti_raw"
  tumvi:
    root: "D:/data/tumvi"
```

## Running experiments

```bash
# Single experiment
python scripts/run_experiment.py --exp 1

# Full pipeline
python scripts/run_all.py

# Tests
pytest tests/ -v
```

## Notes on interpretation

This repository evaluates the scheduling layer inside a simplified estimation testbed. EuRoC and TUM-VI use interpolated pseudo-measurements to isolate scheduler behavior, while KITTI uses a physically grounded GPS path. The saved results should therefore be read as evidence about **covariance-aware scheduling and measurement-efficiency tradeoffs**, not as a claim to outperform full SLAM or VIO systems.

## License

MIT License
