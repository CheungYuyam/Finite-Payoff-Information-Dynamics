"""Plot the two-strategy finite-feedback convergence check."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import PALETTE, setup_style


ROOT = Path(__file__).resolve().parents[1]


def save_publication_figure(fig: mpl.figure.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })

    data_path = ROOT / "results" / "focused_finite_sample_completion" / "two_strategy_finite_sample_correction.csv"
    frame = pd.read_csv(data_path)
    frame = frame[np.isfinite(frame["m"].astype(float))].copy()
    frame = frame.sort_values("m")
    if frame.empty:
        raise ValueError("No finite-feedback rows were found")
    if not np.all(frame["exact_equilibrium_x1"].between(0.0, 1.0)):
        raise ValueError("Equilibrium values fall outside the simplex")
    if (frame[["moment3_root_error", "moment5_root_error"]] <= 0).any().any():
        raise ValueError("Log-scale error panel requires strictly positive errors")

    m = frame["m"].to_numpy(float)
    inverse_m = frame["inverse_m"].to_numpy(float)
    colors = {
        "finite": PALETTE["burgundy"],
        "moment3": PALETTE["blue"],
        "moment5": PALETTE["navy"],
        "reference": PALETTE["grey"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.45), gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    ax.plot(
        inverse_m,
        frame["exact_equilibrium_x1"],
        marker="o",
        color=colors["finite"],
        label=r"finite-feedback stationary point $F_E$",
    )
    ax.plot(
        inverse_m,
        frame["moment3_equilibrium_x1"],
        marker="s",
        color=colors["moment3"],
        label=r"Moment3 stationary point $F_3$",
    )
    ax.plot(
        inverse_m,
        frame["moment5_equilibrium_x1"],
        marker="^",
        color=colors["moment5"],
        label=r"Moment5 stationary point $F_5$",
    )
    reference = float(frame["full_feedback_equilibrium_x1"].iloc[0])
    ax.axhline(
        reference,
        color=colors["reference"],
        linestyle="--",
        linewidth=1.1,
        label=r"fully mixed Nash equilibrium $x_1^\star=0.6$",
    )
    ax.set_xlabel(r"inverse feedback size $1/m$")
    ax.set_ylabel(r"strategy-1 stationary composition $x_1$")
    ax.set_title("(a) Fully mixed Nash equilibrium and stationary points", loc="left")
    ax.set_xlim(-0.02, max(inverse_m) * 1.05)
    roots = frame[["exact_equilibrium_x1", "moment3_equilibrium_x1", "moment5_equilibrium_x1"]].to_numpy()
    lower, upper = min(roots.min(), reference), max(roots.max(), reference)
    padding = 0.09 * (upper - lower)
    ax.set_ylim(lower - padding, upper + padding)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.94,
              fontsize=7.5, loc="lower left")

    ax = axes[1]
    ax.loglog(
        m,
        np.abs(frame["exact_shift_from_full_feedback"]),
        marker="o",
        color=colors["finite"],
        label="finite-feedback displacement from Nash",
    )
    ax.loglog(
        m,
        frame["moment3_root_error"],
        marker="s",
        color=colors["moment3"],
        label="Moment3 root error",
    )
    ax.loglog(
        m,
        frame["moment5_root_error"],
        marker="^",
        color=colors["moment5"],
        label="Moment5 root error",
    )
    ax.set_xlabel(r"feedback size $m$")
    ax.set_ylabel("absolute composition error")
    ax.set_title("(b) Distance from fully mixed Nash equilibrium and root errors", loc="left")
    ax.set_xticks(m, [f"{value:g}" for value in m])
    ax.minorticks_off()
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.94,
              fontsize=7.5, loc="lower left")

    for ax in axes:
        ax.grid(axis="y", color="#E8EBEE", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.89)
    save_publication_figure(
        fig,
        ROOT / "figures" / "value_full_fig_two_strategy_convergence",
    )


if __name__ == "__main__":
    main()
