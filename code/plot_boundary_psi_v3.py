"""Plot the kernel-specific psi applicability boundary."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib

# Headless export only: use the already installed local matplotlib without
# requiring Tk/Tcl or opening a GUI window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import PALETTE, setup_style

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "boundary_psi" / "boundary_psi_full_rows.csv")
    args = parser.parse_args()
    d = pd.read_csv(args.csv)
    key = ["case_id", "beta", "m", "kernel", "sampling_mode", "psi", "dimension", "scale", "dirichlet_alpha", "state_class"]
    p = d[d.model.isin(["classical", "bounded_rho_gate"])].pivot_table(index=key, columns="model", values="relative_error", aggfunc="first").reset_index()
    p["delta"] = p["bounded_rho_gate"] - p["classical"]
    p["bin"] = pd.cut(p.psi, [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35], right=False)
    agg = p.groupby(["kernel", "bin"], observed=True).agg(psi=("psi", "mean"), delta=("delta", "mean")).reset_index()
    setup_style()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    for kernel, color, marker in [("fermi", PALETTE["navy"], "o"), ("arctan", PALETTE["burgundy"], "s")]:
        g = agg[agg.kernel == kernel]
        ax.plot(g.psi, g.delta, color=color, marker=marker, lw=2, ms=5, label=kernel)
    ax.axhline(0.0, color=PALETTE["ink"], lw=1)
    ax.axvline(0.75, color=PALETTE["burgundy"], ls="--", lw=1, alpha=0.8)
    ax.axvline(1.00, color=PALETTE["navy"], ls="--", lw=1, alpha=0.8)
    ax.set_xlabel(r"feedback index $\psi=\beta\sqrt{\mathbb{E}[\Delta\pi^2]}$")
    ax.set_ylabel("bounded-rho gate error − Classical error")
    ax.set_title("Kernel-specific applicability boundary near ψ≈1")
    ax.legend(frameon=False, title="revision kernel")
    out = ROOT / "figures" / "boundary_psi"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "boundary_psi_kernel_map.svg", bbox_inches="tight")
    fig.savefig(out / "boundary_psi_kernel_map.pdf", bbox_inches="tight")
    fig.savefig(out / "boundary_psi_kernel_map.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "boundary_psi_kernel_map.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(out / "boundary_psi_kernel_map.png")


if __name__ == "__main__":
    main()
