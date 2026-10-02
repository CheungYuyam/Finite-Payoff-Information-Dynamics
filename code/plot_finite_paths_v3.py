"""Plot E15 long-horizon finite-population evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import PALETTE, setup_style


ROOT = Path(__file__).resolve().parents[1]


def save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def m_text(value) -> str:
    return r"$m=\infty$" if str(value) == "inf" else f"$m={value}$"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="finite_path_smoke")
    args = parser.parse_args()
    setup_style()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42})
    results = ROOT / "results"
    ensemble = pd.read_csv(results / f"{args.profile}_e15_ensemble_response.csv", dtype={"m": str})
    summary = pd.read_csv(results / f"{args.profile}_e15_path_summary.csv", dtype={"m": str})
    betas = sorted(ensemble["beta"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    setup_style()
    colors = {"1": PALETTE["burgundy"], "inf": PALETTE["navy"]}
    for axis, beta in zip(axes[0], betas[-2:] if len(betas) >= 2 else betas * 2):
        frame = ensemble[ensemble["beta"] == beta]
        population = int(frame["population"].max())
        frame = frame[frame["population"] == population]
        for m_value, subset in frame.groupby("m"):
            subset = subset.sort_values("time")
            color = colors.get(str(m_value), "0.3")
            axis.plot(subset["time"], subset["response_ratio"], color=color,
                      label=f"finite N, {m_text(m_value)}")
            axis.fill_between(
                subset["time"], subset["response_pair_bootstrap_low"],
                subset["response_pair_bootstrap_high"], color=color, alpha=0.16,
            )
            axis.plot(subset["time"], subset["deterministic_response_ratio"],
                      color=color, linestyle="--", linewidth=1.2,
                      label=f"Exact ODE, {m_text(m_value)}")
        axis.axhline(1.0, color=PALETTE["grey"], linewidth=0.8)
        axis.set_yscale("log")
        axis.set_title(rf"$\beta={beta:g}$, $N={population}$")
        axis.set_xlabel("time")
        axis.set_ylabel("paired perturbation response")
    if len(betas) == 1:
        axes[0, 1].set_visible(False)

    markers = {0.5: "o", 0.9: "s"}
    for metric, lower, upper, axis, title in [
        ("exit_probability", "exit_pair_bootstrap_low", "exit_pair_bootstrap_high", axes[1, 0], "Exit from the local neighborhood"),
        ("extinction_probability", "extinction_pair_bootstrap_low", "extinction_pair_bootstrap_high", axes[1, 1], "At least one strategy reaches zero"),
    ]:
        for (beta, m_value), subset in summary.groupby(["beta", "m"]):
            subset = subset.sort_values("population")
            values = subset[metric].to_numpy()
            errors = np.vstack((values - subset[lower].to_numpy(), subset[upper].to_numpy() - values))
            axis.errorbar(
                subset["population"], values, yerr=errors,
                color=colors.get(str(m_value), "0.3"), marker=markers.get(float(beta), "o"),
                capsize=3, label=rf"$\beta={beta:g}$, {m_text(m_value)}",
            )
        axis.set_xscale("log")
        axis.set_ylim(-0.04, 1.04)
        axis.set_xlabel("population size N")
        axis.set_ylabel("probability (seed-pair bootstrap 95% CI)")
        axis.set_title(title)
    handles, labels = [], []
    for axis in axes.ravel():
        if not axis.get_visible():
            continue
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle); labels.append(label)
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center", fontsize=8)
    fig.suptitle("Finite-population path observability diagnostic", y=0.99)
    fig.subplots_adjust(bottom=0.19, hspace=0.34, wspace=0.25)
    save(fig, ROOT / "figures" / f"{args.profile}_fig_e15_finite_paths")


if __name__ == "__main__":
    main()
