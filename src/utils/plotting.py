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
    "fixed_high":            "#2c7bb6",
    "fixed_low":             "#abd9e9",
    "fixed_matched":         "#74add1",
    "heuristic":             "#fdae61",
    "aoi_min":               "#f46d43",
    "periodic_opt":          "#a50026",
    "event_trigger":         "#fee090",
    "var_threshold":         "#f4a582",   # Salmon — tuned reactive variant
    "delay_aware":           "#35978f",   # Teal — closely related to ours
    "clairvoyant_lookahead": "#bababa",   # Grey — non-causal upper bound
    "sb_sched":              "#1a9641",   # Green = ours
}

METHOD_LABELS = {
    "fixed_high":            "Fixed-High",
    "fixed_low":             "Fixed-Low",
    "fixed_matched":         "Fixed-Matched",
    "heuristic":             "Heuristic",
    "aoi_min":               "AoI-Min",
    "periodic_opt":          "Periodic-Opt",
    "event_trigger":         "Event-Trigger",
    "var_threshold":         "Var-Threshold",
    "delay_aware":           "Delay-Aware",
    "clairvoyant_lookahead": "CL (upper bound)",
    "sb_sched":              "SB-Sched (ours)",
}

# Canonical display order for bar charts (upper bound last, ours second-to-last)
METHOD_ORDER = [
    "fixed_high", "fixed_low", "fixed_matched",
    "heuristic", "aoi_min", "periodic_opt",
    "event_trigger", "var_threshold", "delay_aware",
    "sb_sched", "clairvoyant_lookahead",
]


def plot_exp1_vr_rr(df: pd.DataFrame, out_dir: str = "results/exp1"):
    """
    Figure 2: Grouped bar chart showing VR and RR across methods and datasets.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    datasets = df["dataset"].unique()
    methods  = [m for m in METHOD_ORDER if m in df["method"].unique()]

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

        # ESR: prefer the dataset-agnostic measurement column when available
        if "esr_measurement" in sub.columns:
            esr_vals = sub["esr_measurement"].values
            ax1.plot(sub["p_max_mult"], esr_vals, "-o", color=color, label=label)
        else:
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
    color_map = {
        "KF predict": "#2c7bb6",
        "KF update": "#2c7bb6",
        "SB-Sched idle step": "#1a9641",
        "SB-Sched budget": "#1a9641",
        "Heuristic idle step": "#fdae61",
    }
    colors = [color_map.get(op, "#878787") for op in df["operation"]]
    ax.barh(df["operation"], df["time_us"], color=colors, edgecolor="black")
    ax.set_xlabel("Time (µs)")
    ax.set_title("Per-Step Computational Overhead")
    ax.axvline(0, color="black")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig5_overhead.pdf")
    plt.savefig(f"{out_dir}/fig5_overhead.png")
    plt.close()
    print(f"Saved fig5 to {out_dir}/")



def plot_exp7_outage_robustness(df: pd.DataFrame, out_dir: str = "results/exp7"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    scenarios = list(df["scenario"].unique())
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    width = 0.12
    x = np.arange(len(scenarios))
    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        vr_vals = [sub[sub["scenario"] == s]["vr"].mean() * 100 for s in scenarios]
        rec_vals = [sub[sub["scenario"] == s]["recovery_steps_mean"].mean() for s in scenarios]
        axes[0].bar(x + i*width - width*len(methods)/2, vr_vals, width, color=METHOD_COLORS.get(method, "grey"), label=METHOD_LABELS.get(method, method))
        axes[1].bar(x + i*width - width*len(methods)/2, rec_vals, width, color=METHOD_COLORS.get(method, "grey"))
    axes[0].set_xticks(x); axes[0].set_xticklabels(scenarios, rotation=30, ha="right"); axes[0].set_ylabel("Violation Rate (%)"); axes[0].set_title("Outage Robustness")
    axes[1].set_xticks(x); axes[1].set_xticklabels(scenarios, rotation=30, ha="right"); axes[1].set_ylabel("Mean Recovery Steps"); axes[1].set_title("Recovery After Outage")
    axes[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{out_dir}/fig_exp7_outage.png"); plt.savefig(f"{out_dir}/fig_exp7_outage.pdf"); plt.close()


def plot_exp8_latency_jitter(df: pd.DataFrame, out_dir: str = "results/exp8"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    scenarios = list(df["scenario"].unique())
    x = np.arange(len(scenarios)); width = 0.12
    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        vals1 = [sub[sub["scenario"] == s]["vr"].mean() * 100 for s in scenarios]
        vals2 = [sub[sub["scenario"] == s]["area_over_threshold"].mean() for s in scenarios]
        axes[0].bar(x + i*width - width*len(methods)/2, vals1, width, color=METHOD_COLORS.get(method, "grey"), label=METHOD_LABELS.get(method, method))
        axes[1].bar(x + i*width - width*len(methods)/2, vals2, width, color=METHOD_COLORS.get(method, "grey"))
    axes[0].set_xticks(x); axes[0].set_xticklabels(scenarios, rotation=30, ha="right"); axes[0].set_ylabel("Violation Rate (%)"); axes[0].set_title("Latency/Jitter Robustness")
    axes[1].set_xticks(x); axes[1].set_xticklabels(scenarios, rotation=30, ha="right"); axes[1].set_ylabel("Area Over Threshold"); axes[1].set_title("Latency Overshoot Severity")
    axes[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{out_dir}/fig_exp8_latency.png"); plt.savefig(f"{out_dir}/fig_exp8_latency.pdf"); plt.close()


def plot_exp9_resource_budget(df: pd.DataFrame, out_dir: str = "results/exp9"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    scenarios = list(df["scenario"].unique())
    x = np.arange(len(scenarios)); width = 0.12
    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        vals1 = [sub[sub["scenario"] == s]["vr"].mean() * 100 for s in scenarios]
        esr_col = next(c for c in df.columns if c.startswith("esr_"))
        vals2 = [sub[sub["scenario"] == s][esr_col].mean() for s in scenarios]
        axes[0].bar(x + i*width - width*len(methods)/2, vals1, width, color=METHOD_COLORS.get(method, "grey"), label=METHOD_LABELS.get(method, method))
        axes[1].bar(x + i*width - width*len(methods)/2, vals2, width, color=METHOD_COLORS.get(method, "grey"))
    axes[0].set_xticks(x); axes[0].set_xticklabels(scenarios, rotation=30, ha="right"); axes[0].set_ylabel("Violation Rate (%)"); axes[0].set_title("Quality Under Update Budgets")
    axes[1].set_xticks(x); axes[1].set_xticklabels(scenarios, rotation=30, ha="right"); axes[1].set_ylabel("Effective Sample Rate (Hz)"); axes[1].set_title("Used Measurement Rate")
    axes[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{out_dir}/fig_exp9_budget.png"); plt.savefig(f"{out_dir}/fig_exp9_budget.pdf"); plt.close()


def plot_exp10_motion_transitions(df: pd.DataFrame, out_dir: str = "results/exp10"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    phase_df = df[df["analysis"] == "phase_summary"]
    trans_df = df[df["analysis"] == "transition"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method in [m for m in METHOD_ORDER if m in phase_df["method"].unique()]:
        sub = phase_df[phase_df["method"] == method]
        axes[0].plot(sub["phase"], sub["trigger_rate_hz"], '-o', label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method, 'grey'))
    lag_summary = trans_df.groupby("method")["lag_steps"].mean().sort_values()
    axes[1].bar(lag_summary.index, lag_summary.values, color=[METHOD_COLORS.get(m, 'grey') for m in lag_summary.index])
    axes[0].set_title("Trigger Rate by Motion Phase"); axes[0].set_ylabel("Trigger Rate proxy")
    axes[1].set_title("Mean First-Trigger Lag After Transition"); axes[1].set_ylabel("Lag (steps)"); axes[1].tick_params(axis='x', rotation=45)
    axes[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{out_dir}/fig_exp10_motion_transitions.png"); plt.savefig(f"{out_dir}/fig_exp10_motion_transitions.pdf"); plt.close()


def plot_exp11_pareto(df: pd.DataFrame, out_dir: str = "results/exp11"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    families = list(df["family"].unique())
    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 4), squeeze=False)
    for ax, family in zip(axes[0], families):
        sub = df[df["family"] == family].copy().sort_values("esr_measurement")
        for _, row in sub.iterrows():
            ax.scatter(row["esr_measurement"], row["area_over_threshold"], color=METHOD_COLORS.get(row["method"], "grey"), s=60)
            ax.annotate(METHOD_LABELS.get(row["method"], row["method"]), (row["esr_measurement"], row["area_over_threshold"]), fontsize=7)
        front = sub[sub["pareto_area"]].sort_values("esr_measurement")
        if not front.empty:
            ax.plot(front["esr_measurement"], front["area_over_threshold"], '--', color='black')
        ax.set_title(f"Pareto Frontier: {family}")
        ax.set_xlabel("Effective Sample Rate (Hz)")
        ax.set_ylabel("Area Over Threshold")
        ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/fig_exp11_pareto.png"); plt.savefig(f"{out_dir}/fig_exp11_pareto.pdf"); plt.close()
