"""Audit, summarize, and plot the fixed-game cross-condition transfer test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from figure_style import PALETTE, error_cmap, setup_style

ROOT = Path(__file__).resolve().parents[1]
ORDER = [
    "original_classical",
    "anchor_time_scale",
    "anchor_effective_game",
    "oracle_time_scale",
    "oracle_effective_game",
    "moment3_tail",
    "moment5_tail",
]
LABELS = {
    "original_classical": "Original classical",
    "anchor_time_scale": "Fixed anchor time scale",
    "anchor_effective_game": "Fixed anchor effective game",
    "oracle_time_scale": "Condition-refit time scale",
    "oracle_effective_game": "Condition-refit effective game",
    "moment3_tail": "Moment3",
    "moment5_tail": "Moment5",
}


def load_config(path: Path) -> tuple[dict, str]:
    """Read the frozen experiment config without importing simulation code."""
    config = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return config, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "nature_cross_smoke.json")
    args = parser.parse_args()
    config, config_hash = load_config(args.config.resolve())
    profile = config["profile"]
    results = ROOT / "results"
    figures = ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(results / f"{profile}_e10_cross_condition.csv")
    problems = []
    expected = int(config["cases"]) * len(config["targets"]) * len(ORDER)
    if len(frame) != expected:
        problems.append(f"expected {expected} rows, found {len(frame)}")
    if set(frame["config_hash"].astype(str)) != {config_hash}:
        problems.append("config hash mismatch")
    if not np.all(np.isfinite(frame["test_relative_rmse"])):
        problems.append("non-finite RMSE")
    if (frame["test_relative_rmse"] < 0).any():
        problems.append("negative RMSE")

    summary = (
        frame.groupby(["target_condition", "is_anchor", "model"])["test_relative_rmse"]
        .agg(median="median", q90=lambda values: values.quantile(0.9), mean="mean")
        .reset_index()
    )
    summary.to_csv(results / f"{profile}_e10_cross_summary.csv", index=False)
    cross = frame[frame["is_anchor"] == 0]
    aggregate = {}
    for name, subset in cross.groupby("model"):
        aggregate[name] = {
            "median": float(subset["test_relative_rmse"].median()),
            "q90": float(subset["test_relative_rmse"].quantile(0.9)),
        }
    pivot = summary.pivot(index="model", columns="target_condition", values="median").reindex(ORDER)
    penalties = pivot.loc["anchor_effective_game"] / pivot.loc["oracle_effective_game"]
    report = {
        "status": "passed" if not problems else "failed",
        "profile": profile,
        "config_hash": config_hash,
        "problems": problems,
        "rows": int(len(frame)),
        "cross_condition_aggregate": aggregate,
        "fixed_to_oracle_effective_game_penalty": {
            key: float(value) for key, value in penalties.items()
        },
    }
    (results / f"{profile}_e10_cross_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    setup_style()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), gridspec_kw={"height_ratios": [2.3, 1.0]})
    values = pivot.to_numpy(dtype=float)
    image = axes[0].imshow(values, aspect="auto", cmap=error_cmap(), norm=LogNorm(vmin=max(values.min(), 1e-4), vmax=values.max()))
    axes[0].set_yticks(np.arange(len(ORDER)), [LABELS[name] for name in ORDER])
    axes[0].set_xticks(np.arange(len(pivot.columns)), [name.replace("_", "\n") for name in pivot.columns], rotation=0)
    axes[0].set_title("A  Median out-of-sample error under feedback-condition transfer")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if values[row, column] > np.sqrt(values.min() * values.max()) else PALETTE["ink"]
            axes[0].text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", fontsize=7, color=color)
    colorbar = fig.colorbar(image, ax=axes[0], pad=0.015)
    colorbar.set_label("relative RMSE")
    axes[1].bar(np.arange(len(penalties)), penalties.to_numpy(), color=PALETTE["burgundy"], alpha=0.88)
    axes[1].axhline(1.0, color=PALETTE["ink"], linestyle="--", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(penalties)), [name.replace("_", "\n") for name in penalties.index])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("fixed-game / refit-game error")
    axes[1].set_title("B  Transfer penalty of keeping one effective payoff matrix fixed")
    fig.suptitle("A condition-specific fitted game is not a universal replacement for finite feedback", y=1.01)
    fig.tight_layout()
    base = figures / f"{profile}_fig_e10_cross_condition"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
