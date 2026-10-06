"""Integrity and scientific-sanity audit for extended v3 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> tuple[dict, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return config, hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def finite_columns(frame: pd.DataFrame, columns: list[str]) -> bool:
    return all(np.isfinite(frame[column].to_numpy(dtype=float)).all() for column in columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "extended_smoke.json"
    )
    args = parser.parse_args()
    config, config_hash = load_config(args.config.resolve())
    profile = config["profile"]
    results = ROOT / "results"
    manifest_path = results / f"{profile}_extended_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    warnings: list[str] = []
    require(manifest["config_hash"] == config_hash, "manifest config hash mismatch", failures)

    files = {
        "e3_equilibria": results / f"{profile}_e3_equilibria.csv",
        "e3_trajectories": results / f"{profile}_e3_trajectories.csv",
        "e5_phase_map": results / f"{profile}_e5_phase_map.csv",
        "e6_random_stress": results / f"{profile}_e6_random_stress.csv",
        "e7_heterogeneity": results / f"{profile}_e7_heterogeneity.csv",
        "e4_finite_n_runs": results / f"{profile}_e4_finite_n_runs.csv",
        "e4_finite_n_mean": results / f"{profile}_e4_finite_n_mean.csv",
        "e8_benchmark": results / f"{profile}_e8_benchmark.csv",
    }
    frames = {name: pd.read_csv(path) for name, path in files.items()}
    for name, frame in frames.items():
        require(len(frame) == manifest["rows"][name], f"row count mismatch: {name}", failures)
        require(set(frame["config_hash"].astype(str)) == {config_hash}, f"hash mismatch: {name}", failures)

    equilibria = frames["e3_equilibria"]
    require(len(equilibria) == 5, "E3 must contain five nested models", failures)
    require(bool((equilibria["valid"] == 1).all()), "invalid E3 interior equilibrium", failures)
    require(finite_columns(equilibria, ["residual_l2", "max_real_eigenvalue", "max_abs_imag_eigenvalue"]), "non-finite E3 metrics", failures)
    real_parts = equilibria.set_index("model")["max_real_eigenvalue"]
    require(float(real_parts["classical"]) < 0.0 < float(real_parts["exact"]), "constructive case does not reverse stability", failures)
    require(float(real_parts["moment3_tail"]) * float(real_parts["exact"]) > 0.0, "third-moment tail misses stability sign", failures)

    trajectories = frames["e3_trajectories"]
    mass = trajectories.groupby(["model", "time"])["share"].sum()
    require(float(np.max(np.abs(mass - 1.0))) < 1e-10, "E3 trajectory mass violation", failures)
    require(float(trajectories["share"].min()) >= -1e-12, "negative E3 trajectory share", failures)

    phase = frames["e5_phase_map"]
    expected_phase = (
        sum(len(game["states"]) for game in config["phase_games"])
        * len(config["kernels"])
        * len(config["sampling_modes"])
        * len(config["phase_sample_sizes"])
        * len(config["phase_betas"])
        * 4
    )
    require(len(phase) == expected_phase, "E5 factorial grid is incomplete", failures)
    require(finite_columns(phase, ["exact_norm", "error_l2", "relative_error", "skew_signal", "fifth_signal", "mass_error"]), "non-finite E5 values", failures)
    require(float(phase["mass_error"].max()) < 1e-10, "E5 mass conservation failure", failures)

    stress = frames["e6_random_stress"]
    require(len(stress) == int(config["random_stress"]["cases"]), "E6 random case count mismatch", failures)
    stress_metrics = [column for column in stress.columns if column.startswith("error_") or column.startswith("relative_")]
    require(finite_columns(stress, stress_metrics + ["skew_signal", "fifth_signal", "state_min"]), "non-finite E6 values", failures)
    require(float(stress["state_min"].min()) > 0.0, "E6 generated a boundary state", failures)
    moment3_win_rate = float((stress["relative_moment3_tail"] < stress["relative_classical"]).mean())
    moment5_win_rate = float((stress["relative_moment5_tail"] < stress["relative_moment3_tail"]).mean())
    if moment3_win_rate < 0.70:
        warnings.append(f"moment3 beats classical in only {moment3_win_rate:.1%} of stress cases")
    if moment5_win_rate < 0.60:
        warnings.append(f"moment5 beats moment3 in only {moment5_win_rate:.1%} of stress cases")

    heterogeneity = frames["e7_heterogeneity"]
    require(len(heterogeneity) == 4 * len(config["heterogeneous_samples"]), "E7 heterogeneity grid incomplete", failures)
    require(finite_columns(heterogeneity, ["error_l2"]), "non-finite E7 errors", failures)

    finite_runs = frames["e4_finite_n_runs"]
    expected_runs = (
        len(config["finite_population"]["populations"])
        * len(config["finite_population"]["replacement_modes"])
        * int(config["finite_population"]["seeds"])
    )
    require(len(finite_runs) == expected_runs, "E4 finite-N run count mismatch", failures)
    require(finite_columns(finite_runs, ["integrated_l1", "endpoint_l1"]), "non-finite E4 errors", failures)
    require(bool((finite_runs[["integrated_l1", "endpoint_l1"]] >= 0.0).all().all()), "negative E4 error", failures)

    finite_mean = frames["e4_finite_n_mean"]
    mean_mass = finite_mean.groupby(["population", "replacement", "time"])["mean_share"].sum()
    exact_mass = finite_mean.groupby(["population", "replacement", "time"])["exact_share"].sum()
    require(float(np.max(np.abs(mean_mass - 1.0))) < 1e-10, "E4 mean trajectory mass violation", failures)
    require(float(np.max(np.abs(exact_mass - 1.0))) < 1e-10, "E4 exact trajectory mass violation", failures)
    require(finite_columns(finite_mean, ["mean_share", "se_share", "exact_share"]), "non-finite E4 mean values", failures)

    benchmark = frames["e8_benchmark"]
    expected_benchmark = len(config["benchmark"]["dimensions"]) * len(config["benchmark"]["sample_sizes"]) * 4
    require(len(benchmark) == expected_benchmark, "E8 benchmark grid incomplete", failures)
    measured = benchmark[benchmark["status"] == "measured"]
    require(bool((measured["median_seconds"] > 0.0).all()), "non-positive E8 timing", failures)

    report = {
        "schema_version": config["schema_version"],
        "profile": profile,
        "config_hash": config_hash,
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "warnings": warnings,
        "metrics": {
            "phase_max_mass_error": float(phase["mass_error"].max()),
            "moment3_stress_win_rate_vs_classical": moment3_win_rate,
            "moment5_stress_win_rate_vs_moment3": moment5_win_rate,
            "constructive_classical_real_part": float(real_parts["classical"]),
            "constructive_exact_real_part": float(real_parts["exact"]),
        },
    }
    output_path = results / f"{profile}_extended_audit.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
