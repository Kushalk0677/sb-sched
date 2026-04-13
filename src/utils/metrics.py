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


def compute_violation_rate(p_history: list, p_max: float) -> float:
    """Fraction of timesteps where trace(P) exceeds p_max."""
    if not p_history:
        return 0.0
    return sum(1 for p in p_history if p > p_max) / len(p_history)


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
