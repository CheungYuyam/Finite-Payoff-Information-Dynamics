"""Shared visual language for all manuscript figures."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


PALETTE = {
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

# Stable semantic aliases used across figures.
MODEL_COLORS = {
    "classical": PALETTE["grey"],
    "variance_tail": PALETTE["sky"],
    "moment3_tail": PALETTE["blue"],
    "moment5_tail": PALETTE["navy"],
    "exact": PALETTE["burgundy"],
    "finite": PALETTE["burgundy"],
    "moment3": PALETTE["blue"],
    "moment5": PALETTE["navy"],
    "reference": PALETTE["grey"],
}


def setup_style() -> None:
    """Apply the shared manuscript typography and line defaults."""
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
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def diverging_cmap() -> LinearSegmentedColormap:
    """Blue--neutral--burgundy map for signed stability quantities."""
    return LinearSegmentedColormap.from_list(
        "paper_diverging", [PALETTE["blue"], PALETTE["pale_blue"], "#FFFFFF", PALETTE["burgundy"]]
    )


def error_cmap() -> LinearSegmentedColormap:
    """Light-to-dark blue map for relative-error heatmaps."""
    return LinearSegmentedColormap.from_list(
        "paper_error", [PALETTE["mint"], PALETTE["sky"], PALETTE["blue"], PALETTE["navy"]]
    )


def failure_cmap() -> LinearSegmentedColormap:
    """Low-failure mint to high-failure burgundy map."""
    return LinearSegmentedColormap.from_list(
        "paper_failure", [PALETTE["mint"], PALETTE["green"], PALETTE["burgundy"]]
    )


def stability_binary_cmap() -> ListedColormap:
    """Two-state map for stable (light blue) and unstable (burgundy) cells."""
    return ListedColormap([PALETTE["pale_blue"], PALETTE["burgundy"]], name="paper_stability_binary")
