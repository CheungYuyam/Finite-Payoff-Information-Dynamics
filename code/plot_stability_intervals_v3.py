"""Display existing exact-field stability scans as feedback-specific intervals.

Uses all 3808 dense-scan observations and all 11 refined zero crossings.
No observations are omitted, and no new simulations are performed. The
original heatmap generator remains available separately. Colors and fonts
are inherited from the shared manuscript style; the chart itself is rebuilt.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figure_style import PALETTE, setup_style


ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "results" / "value_full_e14_dense_scan.csv"
BOUNDARIES = ROOT / "results" / "value_full_e14_critical_boundaries.csv"
OUT = ROOT / "figures"
STEM = "value_full_fig_e14_stability_region"
M_ORDER = [1, 2, 3, 5, 10, 20, np.inf]
PANELS = [
    ("fermi", "independent", "Fermi / independent"),
    ("fermi", "common", "Fermi / common"),
    ("arctan", "independent", "Arctangent / independent"),
    ("arctan", "common", "Arctangent / common"),
]


def select_branch(frame: pd.DataFrame, kernel: str, sampling: str, m: float) -> pd.DataFrame:
    return frame.loc[
        (frame["kernel"] == kernel)
        & (frame["sampling_mode"] == sampling)
        & (frame["m"] == m)
    ]


def branch_intervals(scan: pd.DataFrame, roots: pd.DataFrame) -> list[tuple[float, float, bool]]:
    """Recover the sign intervals from the scanned endpoint and refined roots.

    Validate the classification against every scan point. Intervals describe
    the numerically resolved branch within the scan range only; there is no
    interpolation in feedback size and no extrapolation beyond the range.
    """
    scan = scan.sort_values("beta")
    roots = roots.sort_values("critical_beta")
    lo, hi = float(scan["beta"].min()), float(scan["beta"].max())
    cuts = roots["critical_beta"].to_numpy(dtype=float)
    if np.any(cuts <= lo) or np.any(cuts >= hi):
        raise ValueError("A refined root lies outside the displayed scan range")
    unstable = bool(scan.iloc[0]["max_real_eigenvalue"] > 0.0)
    segments = []
    left = lo
    for _, root in roots.iterrows():
        expected = "unstable_to_stable" if unstable else "stable_to_unstable"
        if root["crossing_direction"] != expected:
            raise ValueError("Crossing direction disagrees with the branch sign")
        right = float(root["critical_beta"])
        segments.append((left, right, unstable))
        left, unstable = right, not unstable
    segments.append((left, hi, unstable))
    for _, row in scan.iterrows():
        beta, g = float(row["beta"]), float(row["max_real_eigenvalue"])
        matching = [state for a, b, state in segments if a <= beta <= b]
        if len(matching) != 1 or matching[0] != (g > 0.0):
            raise ValueError(f"Interval classification fails at beta={beta}")
    return segments


def draw_branch(ax, y: float, segments, cuts, linewidth: float = 6.0) -> None:
    for left, right, unstable in segments:
        ax.plot(
            [left, right], [y, y],
            color=PALETTE["burgundy"] if unstable else PALETTE["grey"],
            linewidth=linewidth, solid_capstyle="butt", zorder=2,
        )
    for cut in cuts:
        ax.plot(
            [cut, cut], [y - 0.17, y + 0.17],
            color=PALETTE["navy"], linewidth=1.0,
            solid_capstyle="butt", zorder=3,
        )


def main() -> None:
    scan = pd.read_csv(SCAN)
    roots = pd.read_csv(BOUNDARIES)
    if len(scan) != 3808 or len(roots) != 11:
        raise ValueError("The figure expects the complete stored dense scan and root table")
    if scan[["beta", "max_real_eigenvalue"]].isna().any().any():
        raise ValueError("Missing measurements must not be silently omitted")
    if not np.isfinite(scan[["beta", "max_real_eigenvalue"]].to_numpy()).all():
        raise ValueError("Nonfinite measurements in the dense scan")
    if scan.duplicated(["kernel", "sampling_mode", "m", "beta"]).any():
        raise ValueError("Duplicate scan locations")
    beta_grid = np.sort(scan["beta"].unique())
    lo, hi = float(beta_grid[0]), float(beta_grid[-1])
    intervals = {}
    summaries = []
    for kernel, sampling, _ in PANELS:
        for m in M_ORDER:
            branch = select_branch(scan, kernel, sampling, m)
            cuts = select_branch(roots, kernel, sampling, m)
            if not np.array_equal(np.sort(branch["beta"].to_numpy()), beta_grid):
                raise ValueError("Incomplete or incompatible branch scan")
            segments = branch_intervals(branch, cuts)
            intervals[kernel, sampling, m] = (segments, cuts["critical_beta"].to_numpy())
            for left, right, unstable in segments:
                summaries.append({
                    "kernel": kernel, "sampling": sampling,
                    "m": "inf" if np.isinf(m) else int(m),
                    "beta_left": left, "beta_right": right,
                    "classification": "unstable" if unstable else "stable",
                })

    setup_style()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.titlecolor": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "text.color": PALETTE["ink"],
        "xtick.color": PALETTE["ink"], "ytick.color": PALETTE["ink"],
    })
    fig = plt.figure(figsize=(7.2, 5.5), facecolor="white")
    grid = fig.add_gridspec(
        3, 2, height_ratios=[1.0, 1.0, 0.42],
        left=0.085, right=0.975, bottom=0.11, top=0.91,
        wspace=0.22, hspace=0.64,
    )
    for index, (kernel, sampling, title) in enumerate(PANELS):
        ax = fig.add_subplot(grid[index // 2, index % 2])
        for row, m in enumerate(M_ORDER):
            segments, cuts = intervals[kernel, sampling, m]
            draw_branch(ax, float(row), segments, cuts)
        ax.set_xlim(lo, hi)
        ax.set_ylim(6.65, -0.65)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
        ax.set_yticks(range(7), ["1", "2", "3", "5", "10", "20", r"$\infty$"])
        ax.set_title(title, loc="left", pad=9, fontsize=9.5)
        ax.text(-0.12, 1.075, chr(ord("a") + index), transform=ax.transAxes,
                fontsize=10, fontweight="bold", color=PALETTE["ink"])
        ax.set_ylabel(r"feedback size $m$" if index % 2 == 0 else "")
        ax.tick_params(axis="y", length=0, pad=5)
        ax.tick_params(axis="x", length=3, pad=3)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(PALETTE["grey"])
        ax.grid(False)
        if index >= 2:
            ax.set_xlabel(r"response sensitivity $\beta$", labelpad=4)

    detail = fig.add_subplot(grid[2, :])
    segments, cuts = intervals["fermi", "independent", 2]
    if len(cuts) != 2:
        raise ValueError("The detailed m=2 branch must have two refined boundaries")
    draw_branch(detail, 0.0, segments, cuts, linewidth=8.0)
    detail.set_xlim(0.70, 0.86)
    detail.set_ylim(-0.5, 0.85)
    detail.set_yticks([])
    detail.set_xticks([0.70, 0.74, 0.78, 0.82, 0.86])
    detail.set_xlabel(r"response sensitivity $\beta$", labelpad=3)
    detail.set_title(r"Fermi / independent, $m=2$: narrow instability window",
                     loc="left", fontsize=9.5, pad=7)
    detail.text(-0.053, 1.12, "e", transform=detail.transAxes,
                fontsize=10, fontweight="bold")
    detail.spines["left"].set_visible(False)
    detail.spines["bottom"].set_color(PALETTE["grey"])
    detail.tick_params(axis="x", length=3, pad=3)
    for cut in cuts:
        detail.text(cut, 0.32, f"{cut:.4f}", ha="center", va="bottom",
                    color=PALETTE["navy"], fontsize=8)

    legend = [
        Line2D([], [], color=PALETTE["grey"], linewidth=5, solid_capstyle="butt",
               label=r"Locally stable ($g<0$)"),
        Line2D([], [], color=PALETTE["burgundy"], linewidth=5, solid_capstyle="butt",
               label=r"Locally unstable ($g>0$)"),
        Line2D([], [], color=PALETTE["navy"], marker="|", markersize=9,
               linestyle="none", label=r"Refined boundary ($g=0$)"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.53, 1.0),
               ncol=3, handlelength=1.7, columnspacing=1.5, fontsize=8)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{STEM}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{STEM}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{STEM}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{STEM}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(ROOT / "results" / "e14_stability_intervals.csv", index=False)
    report = {
        "core_conclusion": "Instability intervals depend on the kernel, sampling design and feedback size; one Fermi m=2 window is narrow.",
        "archetype": "quantitative grid with a magnified interval panel",
        "reuse_level": "style-only inheritance; interval chart rebuilt",
        "backend": "Python/matplotlib",
        "input_rows": len(scan), "represented_rows": len(scan), "excluded_rows": 0,
        "feedback_branches": len(intervals), "refined_boundaries": len(roots),
        "scan_beta_range": [lo, hi], "no_feedback_interpolation": True,
        "source_sha256": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                          for file in [SCAN, BOUNDARIES]},
        "classification_check": "Every dense-scan sign agrees with its displayed interval",
        "color_roles": {"stable": PALETTE["grey"], "unstable": PALETTE["burgundy"],
                        "boundary": PALETTE["navy"]},
        "figure_width_inches": 7.2, "editable_svg_text": True, "tiff_dpi": 600,
    }
    (ROOT / "results" / "e14_stability_intervals_qa.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
