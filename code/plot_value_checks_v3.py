"""Figures for the high-value finite-N and continuation checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import PALETTE, diverging_cmap, stability_binary_cmap, setup_style


ROOT = Path(__file__).resolve().parents[1]


def save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_refined_boundaries(results: Path, figures: Path, profile: str) -> bool:
    scan_path = results / f"{profile}_e14_dense_scan.csv"
    root_path = results / f"{profile}_e14_critical_boundaries.csv"
    if not scan_path.exists() or not root_path.exists():
        return False
    scan = pd.read_csv(scan_path)
    roots = pd.read_csv(root_path)
    combinations = list(scan.groupby(["kernel", "sampling_mode"]))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.5), sharex=True, sharey=True)
    image = None
    for axis, ((kernel, sampling), frame) in zip(axes.ravel(), combinations):
        beta_values = np.sort(frame["beta"].unique())
        frame = frame.copy()
        frame["feedback_noise_scale"] = np.sqrt(frame["inverse_m"])
        inverse_values = np.sort(frame["feedback_noise_scale"].unique())
        grid = (
            frame.pivot_table(
                index="feedback_noise_scale", columns="beta", values="max_real_eigenvalue"
            )
            .reindex(index=inverse_values, columns=beta_values)
            .to_numpy()
        )
        beta_edges = np.r_[
            beta_values[0] - 0.5 * (beta_values[1] - beta_values[0]),
            0.5 * (beta_values[:-1] + beta_values[1:]),
            beta_values[-1] + 0.5 * (beta_values[-1] - beta_values[-2]),
        ]
        inverse_edges = np.empty(len(inverse_values) + 1)
        inverse_edges[1:-1] = 0.5 * (inverse_values[:-1] + inverse_values[1:])
        inverse_edges[0] = max(0.0, inverse_values[0] - 0.5 * (inverse_values[1] - inverse_values[0]))
        inverse_edges[-1] = inverse_values[-1] + 0.5 * (inverse_values[-1] - inverse_values[-2])
        image = axis.pcolormesh(
            beta_edges,
            inverse_edges,
            (grid > 0.0).astype(float),
            cmap=stability_binary_cmap(),
            vmin=0.0,
            vmax=1.0,
            shading="flat",
        )
        selected_roots = roots[
            (roots["kernel"] == kernel) & (roots["sampling_mode"] == sampling)
        ]
        entering = selected_roots[selected_roots["crossing_direction"] == "stable_to_unstable"]
        leaving = selected_roots[selected_roots["crossing_direction"] == "unstable_to_stable"]
        axis.scatter(
            entering["critical_beta"], np.sqrt(entering["inverse_m"]),
            marker="o", s=34, facecolor="white", edgecolor=PALETTE["ink"], linewidth=0.8,
            label="stable → unstable",
        )
        if not leaving.empty:
            axis.scatter(
                leaving["critical_beta"], np.sqrt(leaving["inverse_m"]),
                marker="X", s=42, color=PALETTE["navy"], linewidth=0.8,
                label="unstable → stable",
            )
        axis.set_title(f"{kernel}, {sampling}")
        axis.set_xlabel(r"selection intensity $\beta$")
        axis.set_xlim(beta_edges[0], beta_edges[-1])
        axis.set_ylim(inverse_edges[0], inverse_edges[-1])
        axis.set_yticks(np.sqrt([0.0, 0.05, 0.1, 0.2, 1.0 / 3.0, 0.5, 1.0]))
        axis.set_yticklabels([r"$\infty$", "20", "10", "5", "3", "2", "1"])
    axes[0, 0].set_ylabel("feedback sample size $m$")
    axes[1, 0].set_ylabel("feedback sample size $m$")
    handles, labels = [], []
    for axis in axes.ravel():
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle); labels.append(label)
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center")
    fig.text(0.79, 0.045, "pale blue: stable    burgundy: unstable", fontsize=8, ha="center")
    fig.suptitle("Finite feedback creates kernel- and coupling-dependent stability boundaries", y=0.99)
    fig.subplots_adjust(bottom=0.14, hspace=0.28, wspace=0.18)
    save(fig, figures / f"{profile}_fig_e14_stability_boundaries_refined")
    return True


def plot_continuation(results: Path, figures: Path, profile: str) -> None:
    scan_path = results / f"{profile}_e14_dense_scan.csv"
    source = scan_path if scan_path.exists() else results / f"{profile}_e14_stability_continuation.csv"
    data = pd.read_csv(source)
    roots_path = results / f"{profile}_e14_critical_boundaries.csv"
    roots = pd.read_csv(roots_path) if roots_path.exists() else pd.DataFrame()
    sizes = sorted(data["m"].unique())
    finite_sizes = [size for size in sizes if np.isfinite(size)]
    # Index every sample size explicitly; a zip with a shorter palette drops curves.
    color_list = [PALETTE["burgundy"], PALETTE["blue"], "#388466", "#B47B22", "#8665A3", PALETTE["sky"]]
    colors = {size: color_list[i % len(color_list)] for i, size in enumerate(finite_sizes)}
    styles = ["-", "-", "--", "-.", (0, (5, 2)), ":"]
    fig = plt.figure(figsize=(7.5, 7.05))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.85],
                           left=0.105, right=0.985, bottom=0.13, top=0.95,
                           hspace=0.66, wspace=0.35)
    axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(2)]
    combinations = [("fermi", "independent"), ("fermi", "common"),
                    ("arctan", "independent"), ("arctan", "common")]
    for index, (axis, (kernel, sampling)) in enumerate(zip(axes, combinations)):
        frame = data[(data.kernel == kernel) & (data.sampling_mode == sampling)]
        for i, size in enumerate(sizes):
            subset = frame[frame.m == size].sort_values("beta")
            if subset.empty:
                raise ValueError(f"Missing continuation branch: {kernel}, {sampling}, {size}")
            is_reference = np.isinf(size)
            axis.plot(subset.beta, 1e3 * subset.max_real_eigenvalue,
                      color="#68717D" if is_reference else colors[size],
                      linestyle=(0, (3, 2)) if is_reference else styles[i % len(styles)],
                      linewidth=1.2, label=r"$m=\infty$ / classical" if is_reference else rf"$m={size:g}$")
        axis.axhline(0, color=PALETTE["ink"], linewidth=0.8, zorder=1)
        lo = min(float(frame.max_real_eigenvalue.min()) * 1e3, 0)
        hi = max(float(frame.max_real_eigenvalue.max()) * 1e3, 0)
        pad = (hi - lo) * 0.10
        axis.set_ylim(lo - pad, hi + pad)
        axis.set_xlim(data.beta.min(), data.beta.max())
        axis.set_xlabel(r"response sensitivity $\beta$")
        axis.set_ylabel(r"$g(\beta)$  [$10^{-3}$]")
        kernel_name = "Fermi" if kernel == "fermi" else "Arctangent"
        axis.set_title(f"({chr(97 + index)}) {kernel_name}, {sampling}", loc="left")
        axis.grid(axis="y", color="#E8EBEE", linewidth=0.6)
        axis.set_axisbelow(True)

    focus = fig.add_subplot(grid[2, :])
    detail = data[(data.kernel == "fermi") & (data.sampling_mode == "independent") & (data.m == 2)]
    boundaries = roots[(roots.kernel == "fermi") & (roots.sampling_mode == "independent") & (roots.m == 2)].sort_values("critical_beta") if not roots.empty else pd.DataFrame()
    if len(boundaries) == 2:
        left, right = boundaries.critical_beta.to_numpy()
        window = (left - 0.015, right + 0.015)
        for row in boundaries.itertuples():
            detail = pd.concat([detail, pd.DataFrame({"beta": [row.critical_beta],
                                                     "max_real_eigenvalue": [row.root_eigenvalue]})])
        detail = detail[(detail.beta >= window[0]) & (detail.beta <= window[1])].sort_values("beta")
        focus.axvspan(left, right, color=PALETTE["burgundy"], alpha=0.09, zorder=0)
        for boundary in (left, right):
            focus.axvline(boundary, color=PALETTE["burgundy"], linestyle=":", linewidth=0.9)
        focus.set_xlim(*window)
        focus.set_xticks([window[0], left, right, window[1]],
                        [f"{window[0]:.2f}", f"{left:.4f}", f"{right:.4f}", f"{window[1]:.2f}"])
    focus.plot(detail.beta, 1e6 * detail.max_real_eigenvalue, color=colors[2.0],
               marker="o", markersize=3.2, linewidth=1.5)
    focus.axhline(0, color=PALETTE["ink"], linewidth=0.8)
    if len(boundaries) == 2:
        focus.scatter(boundaries.critical_beta, 1e6 * boundaries.root_eigenvalue,
                      color=PALETTE["burgundy"], edgecolor="white", linewidth=0.5, s=28, zorder=5)
        focus.text((left + right) / 2, 0.94, "unstable", color=PALETTE["burgundy"],
                   transform=focus.get_xaxis_transform(), ha="center", va="top", fontsize=8)
        focus.text(window[0] + 0.006, 0.94, "stable", color=PALETTE["ink"],
                   transform=focus.get_xaxis_transform(), ha="center", va="top", fontsize=8)
        focus.text(window[1] - 0.006, 0.94, "stable", color=PALETTE["ink"],
                   transform=focus.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    focus.margins(y=0.28)
    focus.set_title("(e) Fermi, independent: enlarged crossing for $m=2$", loc="left")
    focus.set_xlabel(r"response sensitivity $\beta$")
    focus.set_ylabel(r"$g(\beta)$  [$10^{-6}$]")
    focus.grid(axis="y", color="#E8EBEE", linewidth=0.6)
    focus.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.54, 0.005), columnspacing=2.0)
    save(fig, figures / f"{profile}_fig_e14_stability_continuation")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--profile", default="value_smoke")
    args = parser.parse_args(); results = ROOT / "results"; figures = ROOT / "figures"
    setup_style()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42})
    e13 = pd.read_csv(results / f"{args.profile}_e13_finite_generator.csv")
    e14 = pd.read_csv(results / f"{args.profile}_e14_stability_continuation.csv")

    finite = e13[e13["model"] == "finite_N_generator"].sort_values("population")
    classical = float(e13.loc[e13["model"] == "classical", "radial_growth_mean"].iloc[0])
    exact = float(e13.loc[e13["model"] == "exact", "radial_growth_mean"].iloc[0])
    fig, axis = plt.subplots(figsize=(5.4, 3.8))
    axis.errorbar(finite["population"], finite["max_real_eigenvalue"], yerr=finite["eigenvalue_step_spread"],
                  marker="o", capsize=3, color=PALETTE["blue"], label="finite-N exact generator")
    exact_eigen = float(e13.loc[e13["model"] == "exact", "max_real_eigenvalue"].iloc[0])
    classical_eigen = float(e13.loc[e13["model"] == "classical", "max_real_eigenvalue"].iloc[0])
    axis.axhline(exact_eigen, color=PALETTE["burgundy"], label="infinite-population Exact")
    axis.axhline(classical_eigen, color=PALETTE["grey"], linestyle="--", label="Classical")
    axis.axhline(0.0, color=PALETTE["ink"], linewidth=0.8)
    axis.set_xscale("log"); axis.set_xlabel("population size N")
    axis.set_ylabel("maximum real Jacobian eigenvalue")
    axis.set_title("Finite-population generator preserves the stability reversal")
    axis.legend(frameon=False, fontsize=8)
    save(fig, figures / f"{args.profile}_fig_e13_finite_generator")

    plot_continuation(results, figures, args.profile)
    plot_refined_boundaries(results, figures, args.profile)


if __name__ == "__main__":
    main()
