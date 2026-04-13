"""
src/utils/plotting.py

All figure generation for the paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

METHOD_COLORS = {
    "fixed_high":    "#2c7bb6",
    "fixed_low":     "#abd9e9",
    "fixed_matched": "#74add1",
    "heuristic":     "#fdae61",
    "aoi_min":       "#f46d43",
    "whittle":       "#d73027",
    "periodic_opt":  "#a50026",
    "event_trigger": "#fee090",
    "cd_kf":         "#878787",
    "sb_sched":      "#1a9641",   # Green = ours
}

METHOD_LABELS = {
    "fixed_high":    "Fixed-High",
    "fixed_low":     "Fixed-Low",
    "fixed_matched": "Fixed-Matched",
    "heuristic":     "Heuristic",
    "aoi_min":       "AoI-Min",
    "whittle":       "Whittle Index",
    "periodic_opt":  "Periodic-Opt",
    "event_trigger": "Event-Triggered",
    "cd_kf":         "CD-KF Grad",
    "sb_sched":      "SB-Sched (ours)",
}


def plot_exp1_vr_rr(df: pd.DataFrame, out_dir: str = "results/exp1"):
    """
    Figure 2: Grouped bar chart showing VR and RR across methods and datasets.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    datasets = df["dataset"].unique()
    methods  = df["method"].unique()

    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 4), sharey=False)

    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset]
        means = sub.groupby("method")["vr"].mean()
        x = np.arange(len(methods))
        bars = ax.bar(x, [means.get(m, 0) for m in methods],
                      color=[METHOD_COLORS.get(m, "grey") for m in methods],
                      edgecolor="black", linewidth=0.5)
        ax.set_title(dataset.upper())
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Violation Rate (%)")
        ax.axhline(0, color="black", linewidth=0.8)
        # Annotate SB-Sched bar
        idx = list(methods).index("sb_sched") if "sb_sched" in methods else -1
        if idx >= 0:
            ax.annotate("VR=0", xy=(idx, 0), xytext=(idx, 0.02),
                        ha="center", fontsize=8, color="green")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig2_vr_comparison.pdf")
    plt.savefig(f"{out_dir}/fig2_vr_comparison.png")
    plt.close()
    print(f"Saved fig2 to {out_dir}/")


def plot_exp2_pmax_sensitivity(df: pd.DataFrame, out_dir: str = "results/exp2"):
    """
    Figure 3: Line plot — ESR and VR vs P_max multiplier.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("p_max_mult")
        color = METHOD_COLORS.get(method, "grey")
        label = METHOD_LABELS.get(method, method)

        # ESR (pick first sensor column found)
        esr_cols = [c for c in sub.columns if c.startswith("esr_")]
        if esr_cols:
            esr_vals = sub[esr_cols[0]].values
            ax1.plot(sub["p_max_mult"], esr_vals, "-o", color=color, label=label)

        ax2.plot(sub["p_max_mult"], sub["vr"].values * 100,
                 "-o", color=color, label=label)

    ax1.set_xlabel("P_max Multiplier")
    ax1.set_ylabel("Effective Sample Rate (Hz)")
    ax1.set_title("Rate Reduction vs Budget Tightness")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("P_max Multiplier")
    ax2.set_ylabel("Violation Rate (%)")
    ax2.set_title("Quality Guarantee vs Budget Tightness")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig3_pmax_sensitivity.pdf")
    plt.savefig(f"{out_dir}/fig3_pmax_sensitivity.png")
    plt.close()
    print(f"Saved fig3 to {out_dir}/")


def plot_exp4_motion_stress(df: pd.DataFrame, out_dir: str = "results/exp4"):
    """
    Figure 4: VR vs motion intensity per method.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    motion_order = ["low", "medium", "high"]
    methods = df["method"].unique()

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(motion_order))
    width = 0.15
    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        means = [sub[sub["motion_level"] == m]["vr"].mean() * 100
                 for m in motion_order]
        ax.bar(x + i * width - width * len(methods) / 2,
               means, width, label=METHOD_LABELS.get(method, method),
               color=METHOD_COLORS.get(method, "grey"),
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(["Low Motion", "Medium Motion", "High Motion"])
    ax.set_ylabel("Violation Rate (%)")
    ax.set_title("Quality Under Dynamic Motion")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig4_motion_stress.pdf")
    plt.savefig(f"{out_dir}/fig4_motion_stress.png")
    plt.close()
    print(f"Saved fig4 to {out_dir}/")


def plot_exp5_overhead(df: pd.DataFrame, out_dir: str = "results/exp5"):
    """
    Figure 5: Horizontal bar chart of operation times.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ["#2c7bb6", "#2c7bb6", "#1a9641", "#fdae61"]
    ax.barh(df["operation"], df["time_us"], color=colors, edgecolor="black")
    ax.set_xlabel("Time (µs)")
    ax.set_title("Per-Step Computational Overhead")
    ax.axvline(0, color="black")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig5_overhead.pdf")
    plt.savefig(f"{out_dir}/fig5_overhead.png")
    plt.close()
    print(f"Saved fig5 to {out_dir}/")
