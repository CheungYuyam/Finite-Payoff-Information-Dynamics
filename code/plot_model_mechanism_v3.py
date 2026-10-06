"""Section-2 mechanism figure for the CNSNS manuscript.

The figure is deterministic and reproducible.  It uses the same normalized
response kernels and exact conditional field as the manuscript; no simulated
trajectory is introduced.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from automatica_model import (
    classical_rhs,
    exact_rhs,
    normalized_revision_flux,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
        }
    )

    # Palette adapted from the supplied Nature Materials reference.  The same
    # method keeps the same hue in every panel; grey is reserved for references.
    C = {
        "navy": "#16386A",
        "blue": "#1E69B0",
        "sky": "#64B0DF",
        "pale_blue": "#B6D3E5",
        "burgundy": "#770020",
        "green": "#69B78F",
        "mint": "#DBF5EB",
        "grey": "#D2D2D2",
        "ink": "#2F3B4A",
    }

    A = np.asarray([[0.0, 3.0], [2.0, 0.0]])
    x = np.asarray([0.6, 0.4])
    beta = 1.0
    m = 2

    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.24],
        width_ratios=[0.95, 1.25],
        hspace=0.22,
        wspace=0.42,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")

    # (a) The response kernels after the common slope calibration.
    z = np.linspace(-4.0, 4.0, 600)
    axes[0].plot(z, z, color=C["grey"], lw=1.4, ls="--", label=r"linear $z$")
    axes[0].plot(
        z,
        normalized_revision_flux(z, beta, kernel="fermi"),
        color=C["navy"],
        lw=2.0,
        label="Fermi",
    )
    axes[0].plot(
        z,
        normalized_revision_flux(z, beta, kernel="arctan"),
        color=C["burgundy"],
        lw=2.0,
        label="arctangent",
    )
    axes[0].set_title("(a) Calibrated response")
    axes[0].set_xlabel(r"observed payoff difference $z$")
    axes[0].set_ylabel(r"normalized flux $Q_\beta(z)$")
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-3.2, 3.2)
    axes[0].legend(loc="upper left", handlelength=2.5, borderaxespad=0.15)

    # (b) Exact conditional fields in the two-strategy control game.
    grid = np.linspace(0.001, 0.999, 300)
    classical = []
    full = []
    finite_ind = []
    finite_com = []
    for x1 in grid:
        state = np.asarray([x1, 1.0 - x1])
        classical.append(classical_rhs(state, A)[0])
        full.append(exact_rhs(state, A, beta, None, kernel="fermi")[0])
        finite_ind.append(exact_rhs(state, A, beta, m, kernel="fermi", sampling_mode="independent")[0])
        finite_com.append(exact_rhs(state, A, beta, m, kernel="fermi", sampling_mode="common")[0])
    c_classical, = axes[1].plot(grid, classical, color=C["grey"], lw=1.5, ls="--", label="classical")
    c_full, = axes[1].plot(grid, full, color=C["green"], lw=1.8, label="full feedback")
    c_ind, = axes[1].plot(grid, finite_ind, color=C["blue"], lw=1.9, label=r"finite, independent ($m=2$)")
    c_com, = axes[1].plot(grid, finite_com, color=C["burgundy"], lw=1.9, label=r"finite, common ($m=2$)")
    axes[1].axhline(0.0, color=C["ink"], lw=0.8)
    axes[1].axvline(0.6, color=C["ink"], lw=0.8, ls=":")
    axes[1].set_title(r"(b) Conditional population field")
    axes[1].set_xlabel(r"strategy share $x_1$")
    axes[1].set_ylabel(r"$\dot{x}_1$")
    axes[1].set_xlim(0, 1)
    legend_ax.legend(
        [c_classical, c_full, c_ind, c_com],
        ["classical", "full feedback", r"finite, independent ($m=2$)", r"finite, common ($m=2$)"],
        loc="center",
        ncol=4,
        handlelength=2.6,
        columnspacing=1.3,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.065, right=0.995, top=0.91, bottom=0.12)
    fig.savefig(FIGURES / "model_mechanism_section2.svg", bbox_inches="tight")
    fig.savefig(FIGURES / "model_mechanism_section2.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "model_mechanism_section2.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "model_mechanism_section2.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
