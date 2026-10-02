"""Post-process extended v3 results and compute stability boundaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_extended_v3 import (
    classical_interior_equilibrium,
    interior_equilibrium,
    named_rhs,
    simplex_jacobian,
)


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> tuple[dict, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return config, hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    temporary.replace(path)


def stability_grid(config: dict, config_hash: str) -> pd.DataFrame:
    case = config["constructive_case"]
    A = np.asarray(case["payoff"], dtype=float)
    guess = classical_interior_equilibrium(A)
    rows = []
    model_names = [
        "classical", "variance_tail", "moment3_tail", "moment5_tail", "exact"
    ]
    for kernel in config["kernels"]:
        for sampling_mode in config["sampling_modes"]:
            for sample_size in config["phase_sample_sizes"]:
                for beta_value in config["phase_betas"]:
                    beta = float(beta_value)
                    for name in model_names:
                        rhs = named_rhs(
                            name,
                            A,
                            beta,
                            sample_size,
                            kernel=kernel,
                            sampling_mode=sampling_mode,
                        )
                        equilibrium, valid, residual = interior_equilibrium(rhs, guess)
                        if valid:
                            eigenvalues = np.linalg.eigvals(
                                simplex_jacobian(rhs, equilibrium)
                            )
                            max_real = float(np.max(np.real(eigenvalues)))
                            max_imag = float(np.max(np.abs(np.imag(eigenvalues))))
                        else:
                            max_real = np.nan
                            max_imag = np.nan
                        rows.append(
                            {
                                "schema_version": config["schema_version"],
                                "config_hash": config_hash,
                                "kernel": kernel,
                                "sampling_mode": sampling_mode,
                                "m": "inf" if sample_size is None else sample_size,
                                "beta": beta,
                                "model": name,
                                "valid": int(valid),
                                "residual_l2": residual,
                                "max_real_eigenvalue": max_real,
                                "max_abs_imag_eigenvalue": max_imag,
                                "equilibrium": json.dumps(
                                    equilibrium.tolist(), separators=(",", ":")
                                ),
                                "stable": (
                                    int(max_real < 0.0) if np.isfinite(max_real) else ""
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def phase_summary(phase: pd.DataFrame) -> pd.DataFrame:
    grouping = ["kernel", "sampling_mode", "m", "beta", "model"]
    return (
        phase.groupby(grouping, dropna=False)["relative_error"]
        .agg(
            cases="size",
            median="median",
            mean="mean",
            q90=lambda values: values.quantile(0.90),
            q99=lambda values: values.quantile(0.99),
            pass_5pct=lambda values: float((values < 0.05).mean()),
            pass_10pct=lambda values: float((values < 0.10).mean()),
        )
        .reset_index()
    )


def stress_summary(stress: pd.DataFrame) -> pd.DataFrame:
    grouping = ["dimension", "kernel", "sampling_mode", "m", "beta"]
    rows = []
    for keys, frame in stress.groupby(grouping):
        row = dict(zip(grouping, keys))
        row["cases"] = len(frame)
        for name in ("classical", "variance_tail", "moment3_tail", "moment5_tail"):
            values = frame[f"relative_{name}"]
            row[f"{name}_median"] = float(values.median())
            row[f"{name}_q90"] = float(values.quantile(0.90))
            row[f"{name}_q99"] = float(values.quantile(0.99))
        row["moment3_win_vs_classical"] = float(
            (frame["relative_moment3_tail"] < frame["relative_classical"]).mean()
        )
        row["moment5_win_vs_moment3"] = float(
            (frame["relative_moment5_tail"] < frame["relative_moment3_tail"]).mean()
        )
        improvement = frame["relative_variance_tail"] - frame["relative_moment3_tail"]
        row["skew_signal_improvement_spearman"] = float(
            frame["skew_signal"].corr(improvement, method="spearman")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def finite_scaling(finite_runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for replacement, frame in finite_runs.groupby("replacement"):
        means = frame.groupby("population")[["integrated_l1", "endpoint_l1"]].mean()
        for metric in ("integrated_l1", "endpoint_l1"):
            log_n = np.log(means.index.to_numpy(dtype=float))
            log_error = np.log(means[metric].to_numpy(dtype=float))
            slope, intercept = np.polyfit(log_n, log_error, 1)
            fitted = slope * log_n + intercept
            residual = float(np.sum((log_error - fitted) ** 2))
            total = float(np.sum((log_error - float(log_error.mean())) ** 2))
            rows.append(
                {
                    "replacement": int(replacement),
                    "metric": metric,
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r_squared": 1.0 - residual / total if total > 0.0 else 1.0,
                    "population_min": int(means.index.min()),
                    "population_max": int(means.index.max()),
                    "levels": len(means),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "extended_full.json"
    )
    args = parser.parse_args()
    config, config_hash = load_config(args.config.resolve())
    profile = config["profile"]
    results = ROOT / "results"
    phase = pd.read_csv(results / f"{profile}_e5_phase_map.csv")
    stress = pd.read_csv(results / f"{profile}_e6_random_stress.csv")
    finite_runs = pd.read_csv(results / f"{profile}_e4_finite_n_runs.csv")

    stability = stability_grid(config, config_hash)
    phase_stats = phase_summary(phase)
    stress_stats = stress_summary(stress)
    scaling = finite_scaling(finite_runs)
    write_frame(results / f"{profile}_e3_stability_grid.csv", stability)
    write_frame(results / f"{profile}_e5_phase_summary.csv", phase_stats)
    write_frame(results / f"{profile}_e6_stress_summary.csv", stress_stats)
    write_frame(results / f"{profile}_e4_scaling.csv", scaling)

    overall = {}
    for name in ("classical", "variance_tail", "moment3_tail", "moment5_tail"):
        values = phase.loc[phase["model"] == name, "relative_error"]
        overall[name] = {
            "median_relative_error": float(values.median()),
            "q90_relative_error": float(values.quantile(0.90)),
            "q99_relative_error": float(values.quantile(0.99)),
            "pass_5pct": float((values < 0.05).mean()),
            "pass_10pct": float((values < 0.10).mean()),
        }
    exact_grid = stability[stability["model"] == "exact"]
    reversal = exact_grid[exact_grid["max_real_eigenvalue"] > 0.0]
    report = {
        "schema_version": config["schema_version"],
        "profile": profile,
        "config_hash": config_hash,
        "phase_overall": overall,
        "stress": {
            "cases": len(stress),
            "moment3_win_rate_vs_classical": float(
                (stress["relative_moment3_tail"] < stress["relative_classical"]).mean()
            ),
            "moment5_win_rate_vs_moment3": float(
                (stress["relative_moment5_tail"] < stress["relative_moment3_tail"]).mean()
            ),
            "skew_signal_improvement_spearman": float(
                stress["skew_signal"].corr(
                    stress["relative_variance_tail"]
                    - stress["relative_moment3_tail"],
                    method="spearman",
                )
            ),
        },
        "finite_n_scaling": scaling.to_dict(orient="records"),
        "stability_grid": {
            "exact_valid_points": int(exact_grid["valid"].sum()),
            "exact_unstable_points": len(reversal),
            "unstable_by_kernel_and_sampling": (
                reversal.groupby(["kernel", "sampling_mode"]).size().to_dict()
            ),
        },
    }
    # JSON cannot serialize tuple dictionary keys.
    report["stability_grid"]["unstable_by_kernel_and_sampling"] = {
        f"{key[0]}::{key[1]}": int(value)
        for key, value in reversal.groupby(["kernel", "sampling_mode"]).size().items()
    }
    output_path = results / f"{profile}_analysis_summary.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
