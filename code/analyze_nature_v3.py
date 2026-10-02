"""Audit and summarize the v3 Nature-level evidence suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_nature_v3 import load_config


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("classical", "variance_tail", "moment3_tail", "moment5_tail")


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1.0 + z**2 / total
    center = (p + z**2 / (2.0 * total)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / total + z**2 / (4.0 * total**2)) / denominator
    return float(center - half), float(center + half)


def rank_auc(scores: pd.Series, labels: pd.Series) -> float:
    """Tie-aware ROC AUC without an optional machine-learning dependency."""
    scores = pd.Series(scores, dtype=float).reset_index(drop=True)
    labels = pd.Series(labels, dtype=int).reset_index(drop=True)
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        return np.nan
    ranks = scores.rank(method="average").to_numpy()
    rank_sum = float(ranks[positive.to_numpy()].sum())
    return float((rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "nature_smoke.json")
    args = parser.parse_args()
    config, config_hash = load_config(args.config.resolve())
    profile = config["profile"]
    results = ROOT / "results"
    e9 = pd.read_csv(results / f"{profile}_e9_qualitative_map.csv")
    e10 = pd.read_csv(results / f"{profile}_e10_non_equivalence.csv")
    e11 = pd.read_csv(results / f"{profile}_e11_micro_macro.csv")
    e12 = pd.read_csv(results / f"{profile}_e12_paired_accuracy.csv")

    hashes = set()
    for frame in (e9, e10, e11, e12):
        hashes.update(frame["config_hash"].astype(str).unique())
    problems = []
    if hashes != {config_hash}:
        problems.append(f"config hash mismatch: {sorted(hashes)} versus {config_hash}")
    numeric_sets = {
        "e9": e9.loc[e9["exact_valid"] == 1, ["chi", "feedback_skewness", "exact_stability"]],
        "e10": e10[["test_relative_rmse"]],
        "e11": e11.select_dtypes(include=[np.number]),
        "e12": e12[["chi", "exact_norm", "absolute_error", "symmetric_relative_error"]],
    }
    for name, frame in numeric_sets.items():
        if not np.all(np.isfinite(frame.to_numpy(dtype=float))):
            problems.append(f"{name} contains non-finite required metrics")
    if float(e9["exact_valid"].mean()) < 0.95:
        problems.append("fewer than 95% of E9 exact interior equilibria were resolved")

    qualitative = {}
    for ensemble, frame in e9.groupby("ensemble"):
        qualitative[ensemble] = {}
        for name in MODELS:
            valid = frame[(frame["exact_valid"] == 1) & (frame[f"{name}_valid"] == 1)]
            successes = int(valid[f"{name}_stability_mismatch"].sum())
            low, high = wilson(successes, len(valid))
            opposite_to_unstable = int(((valid[f"{name}_label"] == -1) & (valid["exact_label"] == 1)).sum())
            opposite_to_stable = int(((valid[f"{name}_label"] == 1) & (valid["exact_label"] == -1)).sum())
            qualitative[ensemble][name] = {
                "ensemble_cases": int(len(frame)),
                "exact_valid_cases": int((frame["exact_valid"] == 1).sum()),
                "model_valid_cases": int((frame[f"{name}_valid"] == 1).sum()),
                "model_valid_rate": float((frame[f"{name}_valid"] == 1).mean()),
                "valid_cases": int(len(valid)),
                "mismatches": successes,
                "mismatch_rate": float(successes / len(valid)) if len(valid) else np.nan,
                "wilson_95": [low, high],
                "predicts_stable_exact_unstable": opposite_to_unstable,
                "predicts_unstable_exact_stable": opposite_to_stable,
                "median_equilibrium_shift": float(valid[f"{name}_equilibrium_shift"].median()),
            }

    non_equivalence = {}
    for name, frame in e10.groupby("model"):
        non_equivalence[name] = {
            "median_test_relative_rmse": float(frame["test_relative_rmse"].median()),
            "q90_test_relative_rmse": float(frame["test_relative_rmse"].quantile(0.9)),
        }

    paired = {}
    for beta, frame in e12.groupby("beta"):
        paired[str(beta)] = {}
        for name, subset in frame.groupby("model"):
            values = subset["symmetric_relative_error"]
            paired[str(beta)][name] = {
                "median": float(values.median()),
                "q90": float(values.quantile(0.9)),
                "failure_over_10pct": float((values > 0.1).mean()),
            }

    e12 = e12.copy()
    for control in [name for name in ("chi", "psi") if name in e12.columns]:
        bin_name = f"{control}_bin"
        e12[bin_name] = pd.qcut(e12[control], 8, duplicates="drop")
        control_rows = []
        for (interval, name), frame in e12.groupby([bin_name, "model"], observed=True):
            values = frame["symmetric_relative_error"]
            control_rows.append(
                {
                    f"{control}_left": float(interval.left),
                    f"{control}_right": float(interval.right),
                    "model": name,
                    "cases": int(len(frame)),
                    "median": float(values.median()),
                    "q90": float(values.quantile(0.9)),
                    "failure_over_10pct": float((values > 0.1).mean()),
                }
            )
        pd.DataFrame(control_rows).to_csv(
            results / f"{profile}_e12_{control}_summary.csv", index=False
        )

    diagnostics = {}
    for name, frame in e12.groupby("model"):
        failure = (frame["symmetric_relative_error"] > 0.1).astype(int)
        predictors = {
            "beta": frame["beta"],
            "beta_over_sqrt_m": frame["beta"] / np.sqrt(frame["m"]),
            "chi": frame["chi"],
        }
        if "psi" in frame.columns:
            predictors["psi"] = frame["psi"]
        log_error = np.log10(np.maximum(frame["symmetric_relative_error"], 1e-12))
        diagnostics[name] = {
            "failure_prevalence": float(failure.mean()),
            "predictor_auc": {key: rank_auc(value, failure) for key, value in predictors.items()},
            "spearman_with_log_error": {
                key: float(pd.Series(value).rank().corr(pd.Series(log_error).rank()))
                for key, value in predictors.items()
            },
        }

    micro = []
    for (game, condition), frame in e11.groupby(["game", "condition"]):
        if frame["population"].nunique() >= 2:
            slope = float(np.polyfit(np.log(frame["population"]), np.log(frame["integrated_l1"]), 1)[0])
        else:
            slope = np.nan
        entry = {
                "game": game,
                "condition": condition,
                "population_levels": int(frame["population"].nunique()),
                "integrated_l1_slope": slope,
                "scaled_fluctuation_cv": float(frame["sqrt_n_scaled_fluctuation"].std(ddof=1) / max(frame["sqrt_n_scaled_fluctuation"].mean(), 1e-15)),
            }
        if "individual_rms_l2" in frame.columns:
            entry["individual_rms_l2_slope"] = float(
                np.polyfit(np.log(frame["population"]), np.log(frame["individual_rms_l2"]), 1)[0]
            )
            scaled_rms = np.sqrt(frame["population"]) * frame["individual_rms_l2"]
            entry["sqrt_n_scaled_rms_cv"] = float(scaled_rms.std(ddof=1) / max(scaled_rms.mean(), 1e-15))
        micro.append(entry)

    report = {
        "status": "passed" if not problems else "failed",
        "profile": profile,
        "config_hash": config_hash,
        "problems": problems,
        "row_counts": {"e9": len(e9), "e10": len(e10), "e11": len(e11), "e12": len(e12)},
        "qualitative_stability": qualitative,
        "non_equivalence": non_equivalence,
        "paired_accuracy_by_beta": paired,
        "control_diagnostics": diagnostics,
        "micro_macro": micro,
    }
    path = results / f"{profile}_nature_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
