"""Generate publication-style diagnostic figures for the extended v3 suite."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from figure_style import (
    MODEL_COLORS,
    PALETTE,
    diverging_cmap,
    error_cmap,
    failure_cmap,
    setup_style as setup_shared_style,
)


ROOT = Path(__file__).resolve().parents[1]
COLORS = {name: MODEL_COLORS[name] for name in ["classical", "variance_tail", "moment3_tail", "moment5_tail", "exact"]}
LABELS = {
    "classical": "Classical",
    "variance_tail": "Zero-skew cubic",
    "moment3_tail": "Moment3",
    "moment5_tail": "Moment5",
    "exact": r"Proposed $F_E$",
}


def setup_style() -> None:
    setup_shared_style()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def simplex_xy(states: np.ndarray) -> np.ndarray:
    vertices = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]])
    return states @ vertices


def plot_constructive(results: Path, figures: Path, profile: str) -> None:
    trajectories = pd.read_csv(results / f"{profile}_e3_trajectories.csv")
    equilibria = pd.read_csv(results / f"{profile}_e3_equilibria.csv").set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.65), gridspec_kw={"wspace": 0.30})
    colors = {**COLORS, "classical": "#68717D"}
    styles = {"classical": "--", "variance_tail": ":", "moment3_tail": "-.",
              "moment5_tail": (0, (5, 2)), "exact": "-"}
    labels = {**LABELS, "exact": r"Exact $F_E$", "moment3_tail": r"Cubic $F_3$",
              "moment5_tail": r"Quintic $F_5$"}
    states = []
    for name in ("classical", "variance_tail", "moment3_tail", "moment5_tail", "exact"):
        frame = trajectories[trajectories["model"] == name]
        wide = frame.pivot(index="time", columns="strategy", values="share").sort_index()
        values = wide.to_numpy()
        equilibrium = np.asarray(json.loads(equilibria.loc[name, "state"]))
        states.extend([values, equilibrium[None, :]])
        kwargs = dict(color=colors[name], linestyle=styles[name], linewidth=1.35,
                      label=labels[name], zorder=4 if name == "exact" else 2)
        axes[0].plot(values[:, 0], values[:, 2], **kwargs)
        axes[1].plot(wide.index, values[:, 0], **kwargs)
        axes[0].scatter(equilibrium[0], equilibrium[2], color=colors[name], s=32,
                        edgecolor="white", linewidth=0.65, zorder=6)
        start, end = int(len(values) * 0.10), int(len(values) * 0.115)
        axes[0].annotate("", xy=values[end, [0, 2]], xytext=values[start, [0, 2]],
                         arrowprops=dict(arrowstyle="->", color=colors[name], lw=1.1), zorder=5)
    all_states = np.vstack(states)
    for dim, setter in ((0, axes[0].set_xlim), (2, axes[0].set_ylim)):
        lo, hi = all_states[:, dim].min(), all_states[:, dim].max()
        setter(lo - 0.07 * (hi - lo), hi + 0.07 * (hi - lo))
    initial = trajectories[trajectories["time"] == trajectories["time"].min()]
    initial = initial[initial["model"] == "exact"].sort_values("strategy")["share"].to_numpy()
    axes[0].scatter(initial[0], initial[2], marker="D", s=26, color=PALETTE["ink"], zorder=7)
    axes[0].set(xlabel=r"strategy share $x_1$", ylabel=r"strategy share $x_3$")
    axes[0].set_title("(a) Enlarged trajectory region", loc="left")
    axes[1].set(xlabel=r"normalized time $\tau$", ylabel=r"strategy share $x_1$",
                xlim=(trajectories.time.min(), trajectories.time.max()))
    axes[1].set_title("(b) Share evolution", loc="left")
    for axis in axes:
        axis.grid(color="#E8EBEE", linewidth=0.6)
        axis.set_axisbelow(True)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.extend([Line2D([], [], color=PALETTE["ink"], marker="o", linestyle="none", markersize=4),
                    Line2D([], [], color=PALETTE["ink"], marker="D", linestyle="none", markersize=4)])
    legend_labels.extend(["Equilibrium", "Initial state"])
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, fontsize=7.6,
               columnspacing=1.6, handlelength=2.3, bbox_to_anchor=(0.53, 0.0))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.89, bottom=0.28)
    save(fig, figures / f"{profile}_fig_e3_simplex")


def ordered_m(values: pd.Series) -> list[str]:
    labels = {str(value) for value in values}
    finite = sorted((float(value), str(value)) for value in labels if str(value) != "inf")
    return [label for _, label in finite] + (["inf"] if "inf" in labels else [])


def plot_stability(results: Path, figures: Path, profile: str) -> None:
    frame = pd.read_csv(results / f"{profile}_e3_stability_grid.csv", dtype={"m": str})
    frame = frame[frame["model"] == "exact"].copy()
    m_order = ordered_m(frame["m"])
    beta_order = sorted(frame["beta"].unique())
    magnitude = float(np.nanmax(np.abs(frame["max_real_eigenvalue"])))
    norm = TwoSlopeNorm(vmin=-magnitude, vcenter=0.0, vmax=magnitude)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.1), sharex=True, sharey=True)
    image = None
    for axis, ((kernel, mode), subset) in zip(axes.flat, frame.groupby(["kernel", "sampling_mode"], sort=True)):
        grid = subset.pivot(index="m", columns="beta", values="max_real_eigenvalue").reindex(index=m_order, columns=beta_order)
        image = axis.imshow(grid.to_numpy(), origin="lower", aspect="auto", cmap=diverging_cmap(), norm=norm)
        axis.contour(grid.to_numpy(), levels=[0.0], colors=PALETTE["ink"], linewidths=1.1, origin="lower")
        axis.set_title(f"{kernel.capitalize()} / {mode}")
        axis.set_xticks(range(len(beta_order)), [f"{value:g}" for value in beta_order], rotation=45)
        axis.set_yticks(range(len(m_order)), m_order)
        axis.set_xlabel(r"selection intensity $\beta$")
        axis.set_ylabel("feedback samples m")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.88)
    colorbar.set_label("largest real part of local eigenvalues")
    fig.suptitle("Finite feedback moves the stability boundary", y=1.01)
    save(fig, figures / f"{profile}_fig_e3_stability_boundary")


def plot_phase_map(results: Path, figures: Path, profile: str) -> None:
    frame = pd.read_csv(results / f"{profile}_e5_phase_map.csv", dtype={"m": str})
    frame = frame[(frame["kernel"] == "fermi") & (frame["sampling_mode"] == "independent")]
    frame = frame[frame["exact_norm"] >= 1e-10]
    m_order = ordered_m(frame["m"])
    beta_order = sorted(frame["beta"].unique())
    names = ["classical", "variance_tail", "moment3_tail", "moment5_tail"]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.3), sharex=True, sharey=True)
    image = None
    for axis, name in zip(axes.flat, names):
        subset = frame[frame["model"] == name]
        grid = subset.pivot_table(index="m", columns="beta", values="relative_error", aggfunc="median").reindex(index=m_order, columns=beta_order)
        logged = np.log10(np.clip(grid.to_numpy(), 1e-8, 1e4))
        image = axis.imshow(logged, origin="lower", aspect="auto", cmap=error_cmap(), vmin=-6, vmax=4)
        axis.set_title(LABELS[name])
        axis.set_xticks(range(len(beta_order)), [f"{value:g}" for value in beta_order], rotation=45)
        axis.set_yticks(range(len(m_order)), m_order)
        axis.set_xlabel(r"selection intensity $\beta$")
        axis.set_ylabel("feedback samples m")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.88)
    colorbar.set_label(r"$\log_{10}$ median relative drift error")
    fig.suptitle("Accuracy domains of the nested macroscopic models", y=1.01)
    save(fig, figures / f"{profile}_fig_e5_accuracy_domains")


def plot_failure_domains(results: Path, figures: Path, profile: str) -> None:
    frame = pd.read_csv(results / f"{profile}_e5_phase_map.csv", dtype={"m": str})
    frame = frame[(frame["kernel"] == "fermi") & (frame["sampling_mode"] == "independent")]
    frame = frame[frame["exact_norm"] >= 1e-10]
    m_order = ordered_m(frame["m"])
    beta_order = sorted(frame["beta"].unique())
    names = ["classical", "variance_tail", "moment3_tail", "moment5_tail"]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.3), sharex=True, sharey=True)
    image = None
    for axis, name in zip(axes.flat, names):
        subset = frame[frame["model"] == name].copy()
        subset["failed"] = subset["relative_error"] > 0.10
        grid = subset.pivot_table(index="m", columns="beta", values="failed", aggfunc="mean").reindex(index=m_order, columns=beta_order)
        image = axis.imshow(grid.to_numpy(), origin="lower", aspect="auto", cmap=failure_cmap(), vmin=0.0, vmax=1.0)
        axis.set_title(LABELS[name])
        axis.set_xticks(range(len(beta_order)), [f"{value:g}" for value in beta_order], rotation=45)
        axis.set_yticks(range(len(m_order)), m_order)
        axis.set_xlabel(r"selection intensity $\beta$")
        axis.set_ylabel("feedback samples m")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.88)
    colorbar.set_label("fraction with relative drift error > 10%")
    fig.suptitle("Failure probability of finite-order macroscopic descriptions", y=1.01)
    save(fig, figures / f"{profile}_fig_e5_failure_probability")


def plot_stress(results: Path, figures: Path, profile: str) -> None:
    frame = pd.read_csv(results / f"{profile}_e6_random_stress.csv")
    names = ["classical", "variance_tail", "moment3_tail", "moment5_tail"]
    fig, axis = plt.subplots(figsize=(5.7, 4.2))
    for name in names:
        grouped = frame.groupby("beta")[f"relative_{name}"]
        beta = np.asarray(sorted(frame["beta"].unique()))
        median = grouped.median().reindex(beta).to_numpy()
        q90 = grouped.quantile(0.90).reindex(beta).to_numpy()
        axis.plot(beta, median, marker="o", markersize=3.5, color=COLORS[name], label=LABELS[name])
        axis.fill_between(beta, median, q90, color=COLORS[name], alpha=0.13, linewidth=0)
    axis.set_yscale("log")
    axis.set_xlabel(r"selection intensity $\beta$")
    axis.set_ylabel("relative drift error")
    axis.set_title("Random-game stress test: median and 90th percentile")
    axis.legend(frameon=False, ncol=2)
    save(fig, figures / f"{profile}_fig_e6_random_stress")


def plot_finite_n(results: Path, figures: Path, profile: str) -> None:
    frame = pd.read_csv(results / f"{profile}_e4_finite_n_runs.csv")
    fig, axis = plt.subplots(figsize=(5.4, 4.1))
    for replacement, subset in frame.groupby("replacement"):
        grouped = subset.groupby("population")["integrated_l1"]
        mean = grouped.mean()
        error = grouped.std() / np.sqrt(grouped.size())
        label = "with replacement" if replacement else "without replacement"
        marker = "o" if replacement else "s"
        color = PALETTE["green"] if replacement else PALETTE["blue"]
        axis.errorbar(mean.index, mean, yerr=1.96 * error, marker=marker, capsize=2, linewidth=1.2, color=color, label=label)
        slope, intercept = np.polyfit(np.log(mean.index.to_numpy()), np.log(mean.to_numpy()), 1)
        fitted = np.exp(intercept) * mean.index.to_numpy(dtype=float) ** slope
        axis.plot(mean.index, fitted, linestyle="--", linewidth=1.0, color=PALETTE["grey"], label=f"fit: N^{slope:.2f}")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("population size N")
    axis.set_ylabel("time-averaged L1 trajectory error")
    axis.set_title("Finite populations converge at the sampling-noise scale")
    axis.legend(frameon=False)
    save(fig, figures / f"{profile}_fig_e4_finite_n_scaling")


def plot_benchmark(results: Path, figures: Path, profile: str) -> None:
    frame = pd.read_csv(results / f"{profile}_e8_benchmark.csv")
    fig, axis = plt.subplots(figsize=(5.5, 4.1))
    markers = {2: "o", 3: "s", 4: "^", 5: "D"}
    measured = frame[frame["status"] == "measured"]
    for name in ("classical", "moment3_tail", "moment5_tail", "exact"):
        subset = measured[measured["model"] == name]
        for dimension, part in subset.groupby("dimension"):
            axis.scatter(part["support_bound"], part["median_seconds"], s=22, alpha=0.75, marker=markers[int(dimension)], color=COLORS[name], label=LABELS[name] if int(dimension) == 2 else None)
    skipped = frame[(frame["model"] == "exact") & (frame["status"] != "measured")]
    if not skipped.empty:
        ceiling = float(measured["median_seconds"].max()) * 1.6
        axis.scatter(skipped["support_bound"], np.full(len(skipped), ceiling), marker="x", s=30, color=COLORS["exact"], label=r"$F_E$ evaluation: work guard")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("upper bound on payoff-sample support")
    axis.set_ylabel("median RHS evaluation time (s)")
    axis.set_title(r"Direct $F_E$ evaluation cost versus moment approximations")
    axis.legend(frameon=False)
    save(fig, figures / f"{profile}_fig_e8_computational_cost")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="extended_full")
    args = parser.parse_args()
    setup_style()
    results = ROOT / "results"
    figures = ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_constructive(results, figures, args.profile)
    plot_stability(results, figures, args.profile)
    plot_phase_map(results, figures, args.profile)
    plot_failure_domains(results, figures, args.profile)
    plot_stress(results, figures, args.profile)
    plot_finite_n(results, figures, args.profile)
    plot_benchmark(results, figures, args.profile)
    print(f"generated seven figure pairs in {figures}")


if __name__ == "__main__":
    main()
