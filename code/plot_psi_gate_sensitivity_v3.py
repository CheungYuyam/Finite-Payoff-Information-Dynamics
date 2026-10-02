"""Submission-style plots for the psi applicability-gate sensitivity audit."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from figure_style import PALETTE, setup_style

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results" / "realistic_sensitivity" / "psi_gate_sensitivity_summary_v3.csv"
OUT = ROOT / "figures" / "realistic_sensitivity"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN)
    df = df[(df["rho_column"] == "fifth_relative_to_moment3") & (df["rho5_max"] == 0.1)]

    setup_style()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    axes[0].plot(df["psi_max"], df["safe_all_mean_delta"], "o-", color=PALETTE["blue"])
    axes[0].axhline(0, color=PALETTE["ink"], lw=0.8)
    axes[0].axvline(1.0, color=PALETTE["burgundy"], ls="--", lw=1.0)
    axes[0].set(xlabel=r"applicability boundary $\psi_{\max}$", ylabel="safe policy − Classical\nmean relative-error difference")

    axes[1].plot(df["psi_max"], df["safe_all_overshoot_rate"], "o-", color=PALETTE["burgundy"])
    axes[1].axvline(1.0, color=PALETTE["burgundy"], ls="--", lw=1.0)
    axes[1].set(xlabel=r"applicability boundary $\psi_{\max}$", ylabel="safe policy overshoot rate")

    axes[2].plot(df["psi_max"], df["validation_main_mean_delta"], "o-", color=PALETTE["green"], label="odd-case holdout")
    axes[2].axhline(0, color=PALETTE["ink"], lw=0.8)
    axes[2].axvline(1.0, color=PALETTE["burgundy"], ls="--", lw=1.0, label=r"reference $\psi_{\max}=1$")
    axes[2].set(xlabel=r"applicability boundary $\psi_{\max}$", ylabel="holdout main-region\nmean difference")
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "psi_gate_sensitivity_v3.svg", bbox_inches="tight")
    fig.savefig(OUT / "psi_gate_sensitivity_v3.pdf", bbox_inches="tight")
    fig.savefig(OUT / "psi_gate_sensitivity_v3.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "psi_gate_sensitivity_v3.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
