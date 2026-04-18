# SB-Sched: Predictive Uncertainty-Bounded Sensor Activation for Rate-Constrained Robotic and Chemical Sensing

**SB-Sched** is a predictive sensor activation policy that anticipates covariance-bound crossings and schedules acquisitions before violations occur, using only the current covariance trace and a user-defined budget *P*_max. It is evaluated on robotic (EuRoC, KITTI) and chemical sensing datasets, including **CGD-30**, a new 31-day continuous gas-sensing field dataset.

> **Paper:** Khemani, K., Jain, D., Raines, J., Khan, R., & Rizvi, S. H. (2025). *Predictive Uncertainty-Bounded Sensor Activation for Rate-Constrained Robotic and Chemical Sensing.*
>
> **Code & Data:** [github.com/Kushalk0677/sb-sched](https://github.com/Kushalk0677/sb-sched)

---

## Authors

| Name | Role | Email | ORCID |
|---|---|---|---|
| **Kushal Khemani** | Framework, methodology, experiments, supervision | kushal.khemani@gmail.com | [0009-0005-5988-6656](https://orcid.org/0009-0005-5988-6656) |
| **Daksh Jain** | Related work, limitations | primary.jaindaksh@gmail.com | [0009-0005-8620-0701](https://orcid.org/0009-0005-8620-0701) |
| **Jackson Raines** | Theoretical components | raines.jackson.w@gmail.com | [0009-0002-4574-4102](https://orcid.org/0009-0002-4574-4102) |
| **Rakin Khan** | Mathematical formulation | khanr5@kgv.hk | [0009-0003-5732-5404](https://orcid.org/0009-0003-5732-5404) |
| **Syed Hamzah Rizvi** | Figure design | syedhamzah08@gmail.com | [0009-0004-9864-6607](https://orcid.org/0009-0004-9864-6607) |

*All authors are independent researchers.*

---

## Abstract

Many sensor systems operate under strict acquisition-rate, energy, or bandwidth constraints, where reactive threshold-based activation often responds too late to prevent reliability violations. SB-Sched anticipates covariance-budget crossings and schedules acquisitions one step early, absorbing delivery delay. Under matched sensing budgets, SB-Sched reduces violation rate by **2.2×** over the strongest reactive baseline and, under fixed delivery delay, is the **only tested policy to maintain zero violations** across all datasets, including long-horizon field sensing. The policy incurs minimal overhead (≈1.15× a Kalman predict step), making it practical for resource-constrained sensor systems.

---

## What this repository is for

This codebase is a **scheduling and systems testbed**, not a full visual-inertial or SLAM stack.

The central question it answers is:

> Given a high-rate prediction loop and lower-rate measurements, when should the estimator consume a measurement to maintain a covariance budget while minimising sensing cost?

### Important framing

This repository does **not** implement end-to-end sensor fusion in the strongest robotics sense.

- For **EuRoC** and **TUM-VI**, the low-rate `camera` updates are oracle-style pseudo-measurements derived from ground-truth interpolation. They are used to isolate scheduler behaviour from front-end perception noise.
- For **KITTI Raw**, the measurement stream is physically grounded GPS/pose-derived data, but evaluation should still be read as estimator-consistency and scheduling behaviour rather than a claim to outperform production navigation pipelines.

The correct reading of the results is:

- Strong evidence about **covariance-aware adaptive scheduling**
- Useful evidence about **resource-constrained estimation behaviour**
- Not a claim to replace a full SLAM, VIO, or production fusion system

A fuller explanation is in `docs/FRAMING_AND_LIMITATIONS.md`.

---

## Datasets

| Dataset | Description | N | Source |
|---|---|---|---|
| **EuRoC MAV** | IMU at 200 Hz, stereo camera at 20 Hz, Vicon ground truth | — | [Burri et al., 2016](https://doi.org/10.1177/0278364916652421) |
| **KITTI Raw** | GPS/IMU at 100 Hz with RTK ground truth | — | [Geiger et al., 2013](https://doi.org/10.1177/0278364913491297) |
| **UCI Gas Sensor Array Drift** | 16 metal-oxide sensors over 36 months | 13,910 | [Fonollosa et al., 2015](https://doi.org/10.1016/j.snb.2015.03.028) |
| **CGD-30** *(new)* | 31-day continuous field gas-sensing stream, 1-min resolution | 44,640 | This work |

---

## CGD-30: Continuous Gas Deployment Dataset

CGD-30 is a new field dataset introduced alongside this paper, designed as a **stress-test benchmark for long-horizon sensor scheduling** under real-world deployment conditions.

### Overview

| Property | Value |
|---|---|
| Duration | 31 days (April 1 – May 1, 2025) |
| Total samples | 446,400 (44,640 per channel) |
| Resolution | 1 sample per minute |
| Sensor array | 10 MQ-series metal-oxide sensors (MQ-2, MQ-3, MQ-4, MQ-5, MQ-6, MQ-7, MQ-8, MQ-9, MQ-135, MQ-136) |
| Hardware | 10 Arduino nodes (ARD_01 – ARD_10) |
| Signals recorded | ADC value, voltage, resistance, normalised *r*_s/*r*_0 ratio, temperature, humidity |
| Gas species | LPG, butane |
| Concentration range | 350 – 900 ppm |
| Temperature range | 18 °C – 43 °C |
| Humidity range | 24.9 % – 90.0 % RH |

Data collected by K. Khemani

### Controlled Events

The dataset includes **40 logged gas exposure events** with precise start/end timestamps, baseline windows, and metadata. Events span five enclosure conditions and seven perturbation profiles:

**Enclosure conditions:** sealed chamber, open ventilation, temperature variation, humidity controlled, low temperature

**Perturbation profiles:** none (baseline), gradual release, temperature sweeps (10–30 °C, 20–35 °C), humidity levels (45 % RH, 75 % RH), cold test (5 °C)

Event metadata is in `data/onemonth_gas/event_log.csv`. Raw sensor readings are in `data/onemonth_gas/gas_sensor_dataset.csv`.

### Why CGD-30 matters

Most sensing literature evaluates scheduling policies on short-horizon sequences or controlled lab conditions where drift is stationary and gaps are synthetic. CGD-30 exposes the failure modes these settings hide:

- **Nonstationary diurnal cycles** — temperature-coupled drift repeating daily with varying amplitude
- **Poisoning–recovery transients** — sensor response degrades and recovers across multi-hour windows
- **Real acquisition gaps** — uncontrolled missing-data intervals reflecting genuine deployment irregularities
- **Long-horizon drift** — slow baseline shift accumulating over weeks, not minutes

Under these conditions, Event-Trigger reaches VR = 0.333 even in the nominal case — a structural failure, not a tuning problem. SB-Sched maintains VR = 0 under both nominal and delayed conditions across the full 31-day stream.

> A policy that cannot handle 31 days of uncontrolled variation should not be deployed in long-horizon IoT applications.

### File Structure

```
data/onemonth_gas/
├── gas_sensor_dataset.csv   # 446,400 rows — per-minute readings from all 10 sensors
└── event_log.csv            # 40 annotated gas exposure events with metadata
```

### Schema

`gas_sensor_dataset.csv`:

| Column | Description |
|---|---|
| `timestamp` | UTC datetime, 1-minute resolution |
| `arduino_id` | Node identifier (ARD_01 – ARD_10) |
| `sensor_type` | MQ sensor model |
| `sensor_id` | Integer index (0–9) |
| `adc_value` | Raw 10-bit ADC reading |
| `voltage` | Computed voltage (V) |
| `resistance` | Sensor resistance (Ω) |
| `rs_r0_ratio` | Normalised resistance ratio used for scheduling |
| `temperature` | Ambient temperature (°C) |
| `humidity` | Relative humidity (%) |

`event_log.csv`:

| Column | Description |
|---|---|
| `event_id` | Unique event identifier |
| `start_time` / `end_time` | Exposure window |
| `gas_type` | LPG or butane |
| `baseline_before_*` / `baseline_after_*` | Pre/post baseline windows |
| `enclosure_condition` | Deployment environment |
| `gas_concentration` | Target concentration (ppm) |
| `perturbation` | Applied environmental stress |
| `notes` | Free-text annotation |

---

## Key Results

### Budget-Constrained Activation (KITTI Raw @ 2 Hz)

| Method | VR | RMSE (m) | AOT (×10⁻³) |
|---|---|---|---|
| Fixed-Low | 0.306 | 0.708 | 4.98 |
| Fixed-Matched | 0.275 | 0.694 | 4.00 |
| Heuristic | 0.306 | 0.714 | 4.98 |
| Event-Trigger | 0.196 | 0.707 | 2.61 |
| Delay-Aware | 0.210 | 0.683 | 2.29 |
| **SB-Sched** | **0.140** | **0.662** | **0.99** |

*VR = Violation Rate; AOT = Area Over Threshold.*

### Cross-Domain Results Under Delay (SB-Sched vs. Event-Trigger)

| Dataset | SB-Sched VR | Event-Trigger VR |
|---|---|---|
| EuRoC (+Delay) | **0.000** | 0.001 |
| KITTI (+Delay) | **0.000** | 0.004 |
| UCI Gas (+Delay) | **0.000** | 0.135 |
| CGD-30 (+Delay) | **0.000** | 0.333 |

SB-Sched is the **only tested policy** to achieve zero violations under fixed delivery delay on every dataset.

---

## Repository Contents

```
main_repo/
├── configs/            # Dataset roots and experiment settings
├── data/               # Raw datasets (EuRoC, KITTI, UCI Gas, CGD-30)
├── docs/               # Extended documentation
│   ├── PROJECT_OVERVIEW.md
│   ├── SB_Sched_Formal_Guarantee.md
│   ├── experimental_pipeline.md
│   ├── ekf_extension.md
│   └── FRAMING_AND_LIMITATIONS.md
├── results/            # Saved CSVs and figures (exp1–exp11, realism)
├── scripts/            # Entry points: demo, run experiments, plot
├── src/
│   ├── baselines/      # Fixed, heuristic, event-triggered, delay-aware
│   ├── datasets/       # Loaders for EuRoC, KITTI, UCI Gas, CGD-30
│   ├── experiments/    # Exp 1–11 runners
│   ├── kalman/         # Linear KF and EKF paths
│   ├── scheduler/      # SB-Sched core implementation
│   └── utils/          # Metrics and plotting utilities
└── tests/              # Unit tests for scheduler and gas loaders
```

---

## Experiments

| Exp | Description |
|---|---|
| 1–6 | Baseline comparison, sensitivity, asymmetry, motion-stress, overhead, EKF extension |
| 7 | Outage robustness (random drops, burst blackouts) |
| 8 | Latency and jitter robustness (delay, jitter, combined stress) |
| 9 | Resource-budgeted scheduling (hard measurement budgets) |
| 10 | Motion regime transitions (low / medium / high) |
| 11 | Pareto analysis (sensing rate vs. error vs. violation severity) |

---

## Installation

```bash
pip install -e .
pip install -r requirements.txt
```

## Configuration

Edit `configs/config.yaml` to point to your local dataset roots:

```yaml
datasets:
  euroc:
    root: "path/to/euroc"
  kitti:
    root: "path/to/kitti_raw"
  tumvi:
    root: "path/to/tumvi"

experiments:
  p_max_multiplier: 3.0
  p_max_sweep: [2.0, 3.0, 4.0, 5.0]
```

## Running

```bash
# Single experiment
python scripts/run_experiment.py --exp 1

# Full pipeline
python scripts/run_all.py

# Generate plots from saved CSVs
python scripts/plot_results.py --all

# Tests
pytest tests/ -v

# Demo
python scripts/demo.py
```

## Output Locations

```
results/exp1/   through   results/exp11/
results/realism/
results/demo/
```

---

## What the Results Support

**Supported claims:**

1. SB-Sched can reduce effective low-rate measurement usage substantially relative to dense or fixed-rate sensing.
2. Under tight sensing budgets, SB-Sched allocates updates more effectively than fixed and reactive baselines.
3. SB-Sched lies on the Pareto frontier of sensing rate versus quality-loss metrics in budgeted settings.
4. SB-Sched is the only tested policy to achieve zero violations under delivery delay across all four datasets.

**Not supported:**

- Best end-to-end sensor fusion system
- Best overall state-estimation accuracy
- Full visual-inertial odometry benchmark
- Provably fewer reads than any fixed-rate policy in all settings

---

## Citation

```bibtex
@article{khemani2025sbsched,
  title   = {Predictive Uncertainty-Bounded Sensor Activation for Rate-Constrained Robotic and Chemical Sensing},
  author  = {Khemani, Kushal and Jain, Daksh and Raines, Jackson and Khan, Rakin and Rizvi, Syed Hamzah},
  year    = {2026},
}
```

---

## License

MIT License
