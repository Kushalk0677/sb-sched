"""
src/utils/metrics.py

Metric computation for all experiments.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class RunResult:
    """Stores all metrics from a single scheduler run."""
    method: str
    dataset: str
    sequence: str
    violation_rate: float       # Fraction of steps with trace(P) > P_max
    rmse_m: float               # Position RMSE in metres
    esr: dict                   # Effective sample rate per sensor (Hz)
    rate_reduction: dict        # Rate reduction vs Fixed-High per sensor (%)
    p_history: list             # trace(P) at every step
    pos_errors: list            # Position error at every step (m)
    total_steps: int
    p_max: float


def compute_rmse(estimates: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute position RMSE between estimated and GT trajectories.
    Args:
        estimates:    [T x 3] array of estimated positions
        ground_truth: [T x 3] array of GT positions (aligned)
    Returns:
        RMSE in metres
    """
    assert estimates.shape == ground_truth.shape, \
        f"Shape mismatch: {estimates.shape} vs {ground_truth.shape}"
    errors = np.linalg.norm(estimates - ground_truth, axis=1)
    return float(np.sqrt(np.mean(errors**2)))


def compute_violation_rate(p_history: list, p_max: float, warmup_steps: int = 100) -> float:
    """Fraction of post-warmup timesteps where trace(P) exceeds p_max."""
    if not p_history:
        return 0.0
    post = p_history[warmup_steps:]
    if not post:
        return 0.0
    return sum(1 for p in post if p > p_max) / len(post)

def compute_rate_reduction(esr_method: dict, esr_fixed_high: dict) -> dict:
    """
    Percentage rate reduction vs Fixed-High for each sensor.
    Positive = fewer samples (more efficient).
    """
    reductions = {}
    for name in esr_method:
        if name in esr_fixed_high and esr_fixed_high[name] > 0:
            r = (esr_fixed_high[name] - esr_method[name]) / esr_fixed_high[name]
            reductions[name] = float(r * 100)
        else:
            reductions[name] = 0.0
    return reductions


def summarise_results(results: list[RunResult]) -> dict:
    """
    Aggregate results across sequences for a method/dataset pair.
    Returns mean ± std for each metric.
    """
    summary = {}
    methods = list({r.method for r in results})
    datasets = list({r.dataset for r in results})

    for method in methods:
        for dataset in datasets:
            runs = [r for r in results
                    if r.method == method and r.dataset == dataset]
            if not runs:
                continue
            key = f"{method}/{dataset}"
            summary[key] = {
                "vr_mean": np.mean([r.violation_rate for r in runs]),
                "vr_std":  np.std([r.violation_rate for r in runs]),
                "rmse_mean": np.mean([r.rmse_m for r in runs]),
                "rmse_std":  np.std([r.rmse_m for r in runs]),
                "esr_mean": {
                    sensor: np.mean([r.esr.get(sensor, 0) for r in runs])
                    for sensor in runs[0].esr
                },
                "rr_mean": {
                    sensor: np.mean([r.rate_reduction.get(sensor, 0) for r in runs])
                    for sensor in runs[0].rate_reduction
                },
                "n_sequences": len(runs),
            }
    return summary


def compute_area_over_threshold(p_history: list, p_max: float, warmup_steps: int = 100) -> float:
    if not p_history:
        return 0.0
    post = np.array(p_history[warmup_steps:], dtype=float)
    if post.size == 0:
        return 0.0
    return float(np.mean(np.maximum(post - p_max, 0.0)))


def compute_max_overshoot(p_history: list, p_max: float, warmup_steps: int = 100) -> float:
    if not p_history:
        return 0.0
    post = np.array(p_history[warmup_steps:], dtype=float)
    if post.size == 0:
        return 0.0
    return float(np.max(np.maximum(post - p_max, 0.0)))


def compute_recovery_durations(p_history: list, p_max: float, warmup_steps: int = 100, blackout_windows: list | None = None) -> list[int]:
    blackout_windows = blackout_windows or []
    p = np.array(p_history, dtype=float)
    durations = []
    for _start, end in blackout_windows:
        if end < warmup_steps or end >= len(p):
            continue
        for idx in range(end, len(p)):
            if p[idx] <= p_max:
                durations.append(int(idx - end))
                break
        else:
            durations.append(int(len(p) - end))
    return durations


def pareto_frontier_mask(points: np.ndarray) -> np.ndarray:
    """Return mask for 2D minimisation Pareto frontier."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                mask[i] = False
                break
    return mask
