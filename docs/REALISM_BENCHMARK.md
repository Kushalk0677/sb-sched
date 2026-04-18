# Realism benchmark

This benchmark now supports four dataset families:

- `euroc`
- `kitti`
- `uci_gas`
- `onemonth_gas`

## Why the gas datasets were added

After reviewing the repository and the paper scope, the strongest realism gap
was domain concentration: robotics + GPS only. The two gas datasets add:

- long-horizon sensor drift
- real deployment noise
- missing samples / delayed measurements under the same scheduler logic
- a non-robotics validation axis for the paper

## Commands

### EuRoC

```bash
python scripts/run_realism_benchmark.py --dataset euroc --sequence MH_03_medium
```

### KITTI

```bash
python scripts/run_realism_benchmark.py --dataset kitti --sequence 2011_09_26_drive_0005_sync
```

### UCI gas drift

```bash
python scripts/run_realism_benchmark.py --dataset uci_gas --sequence all --uci-feature-index 2
```

### One-month gas dataset

```bash
python scripts/run_realism_benchmark.py --dataset onemonth_gas --sequence sensor_0 --sensor-id 0 --feature rs_r0_ratio
```

## Outputs

For each run the benchmark writes:

- `*_realism_full.csv` — all methods and settings
- `*_realism_summary.csv` — SB-Sched summary table for the paper

## Notes on interpretation

- The gas datasets use a 1-D local-linear-trend KF to estimate the selected gas
  sensor response over time.
- The one-month dataset is the stronger realism source because it is a genuine
  long-horizon deployment-like stream with natural drift and environmental
  variation.
- The UCI dataset is still useful because it adds a known external benchmark and
  batch-to-batch drift.
- These gas experiments test scheduling robustness under noisy drifting sensing;
  they do not claim chemistry-specific state estimation or calibrated gas
  concentration reconstruction.
