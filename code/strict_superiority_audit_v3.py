"""Strict, pre-specified superiority audit for the v3 evidence suite.

This report separates three claims that are often conflated:
1. approximation improvement (paired error difference against Classical),
2. qualitative identification (stability-mismatch rates), and
3. transfer robustness (fixed-game cross-condition error).

All uncertainty intervals are case-level bootstrap intervals.  No result is
called a win unless the paired point estimate and its interval support that
direction.  Negative differences mean that the proposed approximation has
smaller error than its comparator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_nature_v3 import rank_auc, wilson

ROOT = Path(__file__).resolve().parents[1]


def bootstrap_mean(values: np.ndarray, seed: int, draws: int = 5000) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def bootstrap_rate(values: np.ndarray, seed: int, draws: int = 5000) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    rates = values[indices].mean(axis=1)
    return [float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))]


def bootstrap_median(values: np.ndarray, seed: int, draws: int = 5000) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    medians = np.median(values[indices], axis=1)
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def paired_error_record(frame: pd.DataFrame, proposed: str, comparator: str, seed: int) -> dict:
    difference = frame[f"relative_{proposed}"] - frame[f"relative_{comparator}"]
    improvement = (difference < 0).to_numpy(dtype=float)
    ci = bootstrap_mean(difference.to_numpy(), seed)
    rate_ci = bootstrap_rate(improvement, seed + 1)
    return {
        "n": int(len(frame)),
        "mean_difference_proposed_minus_comparator": float(difference.mean()),
        "median_difference_proposed_minus_comparator": float(difference.median()),
        "median_difference_bootstrap_95": bootstrap_median(difference.to_numpy(), seed + 2),
        "win_rate": float(improvement.mean()),
        "mean_difference_bootstrap_95": ci,
        "win_rate_bootstrap_95": rate_ci,
        "supported_mean_improvement": bool(ci[1] < 0.0),
        "supported_median_improvement": bool(bootstrap_median(difference.to_numpy(), seed + 3)[1] < 0.0),
        "supported_win_rate_above_50pct": bool(rate_ci[0] > 0.5),
    }


def main() -> None:
    results = ROOT / "results"
    stress = pd.read_csv(results / "extended_full_e6_random_stress.csv")
    e9 = pd.read_csv(results / "nature_full_e9_qualitative_map.csv")
    cross = pd.read_csv(results / "nature_cross_full_e10_cross_condition.csv")
    e12 = pd.read_csv(results / "nature_full_e12_paired_accuracy.csv")

    report: dict = {
        "schema_version": "v3-strict-audit-1",
        "purpose": "case-level uncertainty audit; exploratory strata are reported separately",
        "approximation_superiority": {},
        "stability_identification": {},
        "cross_condition_transfer": {},
        "control_variable": {},
        "guardrails": [
            "A point estimate alone is not called a supported improvement.",
            "High-order tails are not required to dominate in every regime.",
            "Exact remains the mechanism-level reference; comparisons against Exact are not treated as prediction wins.",
        ],
    }

    report["approximation_superiority"]["moment3_vs_classical_overall"] = paired_error_record(
        stress, "moment3_tail", "classical", 2026090301
    )
    report["approximation_superiority"]["moment5_vs_moment3_overall"] = paired_error_record(
        stress, "moment5_tail", "moment3_tail", 2026090302
    )
    strata = {}
    for keys, frame in stress.groupby(["dimension", "kernel", "sampling_mode"], dropna=False):
        key = "dimension=%s|kernel=%s|sampling=%s" % keys
        strata[key] = {
            "moment3_vs_classical": paired_error_record(frame, "moment3_tail", "classical", 20261001 + int(keys[0])),
            "moment5_vs_moment3": paired_error_record(frame, "moment5_tail", "moment3_tail", 20262001 + int(keys[0])),
        }
    report["approximation_superiority"]["predefined_strata"] = strata

    # Stability identification is a qualitative task: lower mismatch is better.
    constructive = e9[(e9["ensemble"] == "constructive_neighborhood") & (e9["exact_valid"] == 1)]
    for name in ("classical", "variance_tail", "moment3_tail", "moment5_tail"):
        valid = constructive[constructive[f"{name}_valid"] == 1]
        mismatch = valid[f"{name}_stability_mismatch"].to_numpy(dtype=float)
        low, high = wilson(int(mismatch.sum()), len(mismatch))
        report["stability_identification"][name] = {
            "n": int(len(mismatch)),
            "mismatch_rate": float(mismatch.mean()),
            "wilson_95": [low, high],
            "stable_to_unstable_count": int(((valid[f"{name}_label"] == -1) & (valid["exact_label"] == 1)).sum()),
            "unstable_to_stable_count": int(((valid[f"{name}_label"] == 1) & (valid["exact_label"] == -1)).sum()),
        }
    # Classical's constructive-neighborhood error rate is explicitly compared
    # to the unconditional random-interior baseline, preventing prevalence
    # inflation from being mistaken for universality.
    random_interior = e9[(e9["ensemble"] == "unconditional_interior") & (e9["exact_valid"] == 1)]
    for label, frame in (("constructive_neighborhood", constructive), ("unconditional_interior", random_interior)):
        valid = frame[frame["classical_valid"] == 1]
        values = valid["classical_stability_mismatch"].to_numpy(dtype=float)
        report["stability_identification"][f"classical_{label}"] = {
            "n": int(len(values)),
            "mismatch_rate": float(values.mean()),
            "wilson_95": list(wilson(int(values.sum()), len(values))),
        }

    # Transfer: compare fixed anchor effective game to the refit oracle on
    # matched case-target rows; lower RMSE is better.
    non_anchor = cross[cross["is_anchor"] == 0].copy()
    pivot = non_anchor.pivot_table(index=["case_id", "target_condition"], columns="model", values="test_relative_rmse")
    transfer = {}
    for comparator in ("original_classical", "anchor_effective_game", "moment3_tail", "moment5_tail"):
        if comparator not in pivot or "oracle_effective_game" not in pivot:
            continue
        diff = (pivot[comparator] - pivot["oracle_effective_game"]).dropna().to_numpy()
        # Positive means the comparator is worse than the condition-refit oracle.
        transfer[comparator] = {
            "n": int(len(diff)),
            "mean_excess_error_vs_oracle": float(diff.mean()),
            "median_excess_error_vs_oracle": float(np.median(diff)),
            "bootstrap_95": bootstrap_mean(diff, 20263001 + len(diff)),
        }
    report["cross_condition_transfer"] = transfer

    # [psi] is evaluated as a model-selection diagnostic, not as a model
    # accuracy claim.  Bootstrap the AUC and its difference to chi.
    diagnostics = {}
    for name, frame in e12.groupby("model"):
        failure = (frame["symmetric_relative_error"] > 0.1).astype(int)
        auc_psi = rank_auc(frame["psi"], failure)
        auc_chi = rank_auc(frame["chi"], failure)
        rng = np.random.default_rng(20264000 + len(name))
        values = frame[["psi", "chi", "symmetric_relative_error"]].to_numpy()
        auc_diff = []
        for _ in range(3000):
            sample = values[rng.integers(0, len(values), len(values))]
            labels = (sample[:, 2] > 0.1).astype(int)
            auc_diff.append(rank_auc(sample[:, 0], labels) - rank_auc(sample[:, 1], labels))
        diagnostics[name] = {
            "auc_psi": float(auc_psi),
            "auc_chi": float(auc_chi),
            "auc_difference_psi_minus_chi": float(auc_psi - auc_chi),
            "auc_difference_bootstrap_95": [float(np.quantile(auc_diff, 0.025)), float(np.quantile(auc_diff, 0.975))],
        }
    report["control_variable"] = diagnostics

    path = results / "strict_superiority_audit_v3.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
