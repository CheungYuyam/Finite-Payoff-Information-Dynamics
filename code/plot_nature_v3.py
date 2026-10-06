"""Publication-style figures for the v3 Nature evidence suite."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

from figure_style import MODEL_COLORS, PALETTE, setup_style as setup_shared_style

ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    **{name: MODEL_COLORS[name] for name in ["classical", "variance_tail", "moment3_tail", "moment5_tail"]},
    "best_time_scaled_classical": PALETTE["sky"],
    "best_effective_classical": PALETTE["burgundy"],
}
LABELS = {
    "classical": "Classical",
    "variance_tail": "Zero-skew cubic",
    "moment3_tail": "Moment3",
    "moment5_tail": "Moment5",
    "best_time_scaled_classical": "Best time-scaled classical",
    "best_effective_classical": "Best fitted effective game",
}


def setup() -> None:
    setup_shared_style()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval used only for the plotted binomial error bars."""
    if total == 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1.0 + z**2 / total
    center = (p + z**2 / (2.0 * total)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / total + z**2 / (4.0 * total**2)) / denominator
    return float(center - half), float(center + half)


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_qualitative(frame: pd.DataFrame, figures: Path, profile: str) -> None:
    ensembles = ["unconditional_interior", "near_boundary", "constructive_neighborhood"]
    names = ["classical", "variance_tail", "moment3_tail", "moment5_tail"]
    x = np.arange(len(ensembles), dtype=float)
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    axis = axes[0]
    for offset, name in enumerate(names):
        rates, lower, upper = [], [], []
        for ensemble in ensembles:
            subset = frame[(frame["ensemble"] == ensemble) & (frame["exact_valid"] == 1) & (frame[f"{name}_valid"] == 1)]
            successes = int(subset[f"{name}_stability_mismatch"].sum())
            rate = successes / len(subset) if len(subset) else np.nan
            lo, hi = wilson(successes, len(subset))
            rates.append(rate)
            # Wilson intervals can differ from an exact zero rate by roundoff
            # (~1e-18). Matplotlib requires non-negative error magnitudes.
            lower.append(max(0.0, rate - lo))
            upper.append(max(0.0, hi - rate))
        positions = x + (offset - 1.5) * width
        axis.bar(positions, rates, width=width, color=COLORS[name], alpha=0.88, label=LABELS[name])
        axis.errorbar(positions, rates, yerr=np.asarray([lower, upper]), fmt="none", color=PALETTE["ink"], linewidth=0.7, capsize=2)
    axis.set_xticks(x, ["Unconditional\ninterior games", "Classical boundary\nneighbourhood", "Constructive-case\nneighbourhood"])
    axis.set_ylabel("stability-sign mismatch rate")
    axis.set_ylim(0.0, 0.5)
    axis.set_title("A  Stability-sign mismatch")
    for offset, name in enumerate(names):
        coverage = []
        for ensemble in ensembles:
            exact_valid = frame[(frame["ensemble"] == ensemble) & (frame["exact_valid"] == 1)]
            coverage.append(float((exact_valid[f"{name}_valid"] == 1).mean()) if len(exact_valid) else np.nan)
        positions = x + (offset - 1.5) * width
        axes[1].bar(positions, coverage, width=width, color=COLORS[name], alpha=0.88, label=LABELS[name])
    axes[1].set_xticks(x, ["Unconditional\ninterior games", "Classical boundary\nneighbourhood", "Constructive-case\nneighbourhood"])
    axes[1].set_ylabel("fraction with resolved interior equilibrium")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("B  Equilibrium-recovery coverage")
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Qualitative errors are localized; approximation failure is reported", y=1.13)
    save(fig, figures / f"{profile}_fig_e9_stability_mismatch")


def plot_paired(frame: pd.DataFrame, figures: Path, profile: str) -> None:
    frame = frame.copy()
    control = "psi" if "psi" in frame.columns else "chi"
    bin_name = f"{control}_bin"
    frame[bin_name] = pd.qcut(frame[control], 8, duplicates="drop")
    names = ["classical", "variance_tail", "moment3_tail", "moment5_tail"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for name in names:
        subset = frame[frame["model"] == name]
        grouped = subset.groupby(bin_name, observed=True)
        centers = np.asarray([float((interval.left + interval.right) / 2.0) for interval in grouped.groups])
        median = grouped["symmetric_relative_error"].median().to_numpy()
        q90 = grouped["symmetric_relative_error"].quantile(0.9).to_numpy()
        failure = grouped["symmetric_relative_error"].apply(lambda values: float((values > 0.1).mean())).to_numpy()
        axes[0].plot(centers, median, marker="o", markersize=3.2, color=COLORS[name], label=LABELS[name])
        axes[0].fill_between(centers, median, q90, color=COLORS[name], alpha=0.12, linewidth=0)
        axes[1].plot(centers, failure, marker="o", markersize=3.2, color=COLORS[name], label=LABELS[name])
    axes[0].set_yscale("log")
    axes[0].axhline(0.1, color=PALETTE["ink"], linestyle="--", linewidth=0.8)
    control_label = (
        r"comparison-signal control $\psi=\beta\sqrt{\mathbb{E}[(\Delta\pi)^2]}$"
        if control == "psi" else r"finite-feedback control $\chi=\beta\sigma_{\Delta\pi}$"
    )
    axes[0].set_xlabel(control_label)
    axes[0].set_ylabel("symmetric relative drift error")
    axes[0].set_title("Median and 90th percentile")
    axes[1].set_xlabel(control_label)
    axes[1].set_ylabel("fraction with error > 10%")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Predictable failure boundary")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Paired random-game test of the approximation hierarchy", y=1.02)
    save(fig, figures / f"{profile}_fig_e12_paired_control")


def plot_nonequivalence(frame: pd.DataFrame, figures: Path, profile: str) -> None:
    order = ["classical", "best_time_scaled_classical", "best_effective_classical", "moment3_tail", "moment5_tail"]
    data = [frame.loc[frame["model"] == name, "test_relative_rmse"].to_numpy() for name in order]
    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    boxes = axis.boxplot(data, patch_artist=True, showfliers=True, tick_labels=[LABELS[name] for name in order])
    for patch, name in zip(boxes["boxes"], order):
        patch.set_facecolor(COLORS[name]); patch.set_alpha(0.75)
    axis.set_yscale("log")
    axis.tick_params(axis="x", rotation=22)
    axis.set_ylabel("out-of-sample relative RMSE")
    axis.set_title("Can finite feedback be absorbed into a classical game?")
    save(fig, figures / f"{profile}_fig_e10_nonequivalence")


def plot_micro(frame: pd.DataFrame, figures: Path, profile: str) -> None:
    metric = "individual_rms_l2" if "individual_rms_l2" in frame.columns else "integrated_l1"
    games = list(frame["game"].drop_duplicates())
    conditions = list(frame["condition"].drop_duplicates())
    condition_colors = [PALETTE["navy"], PALETTE["blue"], PALETTE["sky"], PALETTE["green"], PALETTE["burgundy"], PALETTE["grey"]]
    palette = dict(zip(conditions, condition_colors[: len(conditions)]))
    fig, axes = plt.subplots(1, len(games), figsize=(4.0 * len(games), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, game in zip(axes, games):
        game_frame = frame[frame["game"] == game]
        for condition, subset in game_frame.groupby("condition"):
            subset = subset.sort_values("population")
            axis.plot(subset["population"], subset[metric], marker="o", color=palette[condition], label=condition)
        populations = np.sort(game_frame["population"].unique())
        anchor = float(game_frame.loc[game_frame["population"] == populations[0], metric].median())
        guide = anchor * np.sqrt(populations[0] / populations)
        axis.plot(populations, guide, color=PALETTE["grey"], linestyle="--", linewidth=0.9, label=r"$N^{-1/2}$ guide")
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.set_xticks(populations)
        axis.set_xticklabels([str(int(value)) for value in populations])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xlabel("population size N")
        axis.set_title(game.replace("_", " "))
    axes[0].set_ylabel(r"single-trajectory RMS $L^2$ error to $F_E$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=min(5, len(labels)), loc="lower center", bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Micro–macro closure across games and feedback mechanisms", y=1.02)
    save(fig, figures / f"{profile}_fig_e11_micro_macro")

    if "individual_rms_l2" in frame.columns:
        fig, axes = plt.subplots(1, len(games), figsize=(4.0 * len(games), 3.6), sharey=True)
        axes = np.atleast_1d(axes)
        for axis, game in zip(axes, games):
            game_frame = frame[frame["game"] == game]
            for condition, subset in game_frame.groupby("condition"):
                subset = subset.sort_values("population")
                collapsed = np.sqrt(subset["population"]) * subset["individual_rms_l2"]
                axis.plot(subset["population"], collapsed, marker="o", color=palette[condition], label=condition)
            axis.set_xscale("log")
            populations = np.sort(game_frame["population"].unique())
            axis.set_xticks(populations)
            axis.set_xticklabels([str(int(value)) for value in populations])
            axis.xaxis.set_minor_locator(NullLocator())
            axis.set_xlabel("population size N")
            axis.set_title(game.replace("_", " "))
        axes[0].set_ylabel(r"$\sqrt{N}$ × trajectory RMS error")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, ncol=min(4, len(labels)), loc="lower center", bbox_to_anchor=(0.5, -0.09))
        fig.suptitle(r"Fluctuation collapse expected from an $N^{-1/2}$ law", y=1.02)
        save(fig, figures / f"{profile}_fig_e11_fluctuation_collapse")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="nature_smoke")
    args = parser.parse_args()
    setup()
    results = ROOT / "results"; figures = ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_qualitative(pd.read_csv(results / f"{args.profile}_e9_qualitative_map.csv"), figures, args.profile)
    plot_nonequivalence(pd.read_csv(results / f"{args.profile}_e10_non_equivalence.csv"), figures, args.profile)
    plot_micro(pd.read_csv(results / f"{args.profile}_e11_micro_macro.csv"), figures, args.profile)
    plot_paired(pd.read_csv(results / f"{args.profile}_e12_paired_accuracy.csv"), figures, args.profile)
    count = 5 if "individual_rms_l2" in pd.read_csv(results / f"{args.profile}_e11_micro_macro.csv", nrows=1).columns else 4
    print(f"generated {count} Nature-evidence figure pairs in {figures}")


if __name__ == "__main__":
    main()
