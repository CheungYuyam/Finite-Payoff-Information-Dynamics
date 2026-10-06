"""E15: long-horizon finite-population paths around the reversal equilibrium."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

import automatica_model as model
from run_extended_v3 import (
    classical_interior_equilibrium,
    interior_equilibrium,
    named_rhs,
    simplex_jacobian,
)


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> tuple[dict, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return config, hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def runtime_directory() -> Path:
    workspace = ROOT.parents[2]
    runtime = workspace / ".codex_tmp" / "finite_path_v3"
    runtime.mkdir(parents=True, exist_ok=True)
    return runtime


def compile_simulator(runtime: Path) -> Path:
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ is required for the long-horizon finite-path experiment")
    source = ROOT / "code" / "finite_path_simulator_v3.cpp"
    ascii_source = runtime / "finite_path_simulator_v3.cpp"
    executable = runtime / "finite_path_simulator_v3.exe"
    if not ascii_source.exists() or ascii_source.read_bytes() != source.read_bytes():
        shutil.copy2(source, ascii_source)
    if not executable.exists() or executable.stat().st_mtime < ascii_source.stat().st_mtime:
        subprocess.run(
            [compiler, "-O3", "-std=c++17", str(ascii_source), "-o", str(executable)],
            check=True,
        )
    return executable


def vector_text(state: np.ndarray) -> str:
    return ",".join(f"{value:.17g}" for value in state)


def tangent_direction(rhs, equilibrium: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    jacobian = simplex_jacobian(rhs, equilibrium)
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    index = int(np.argmax(np.imag(eigenvalues)))
    if abs(float(np.imag(eigenvalues[index]))) < 1e-10:
        index = int(np.argmax(np.real(eigenvalues)))
    reduced = np.real(eigenvectors[:, index])
    direction = np.r_[reduced, -float(reduced.sum())]
    direction /= float(np.linalg.norm(direction))
    eigenvector = eigenvectors[:, index]
    basis = np.column_stack((np.real(eigenvector), -np.imag(eigenvector)))
    if abs(float(np.linalg.det(basis))) < 1e-10:
        basis = np.eye(2)
    return direction, float(np.max(np.real(eigenvalues))), basis


def modal_norm(delta: np.ndarray, basis: np.ndarray) -> np.ndarray:
    reduced = np.asarray(delta)[..., :2]
    coordinates = np.linalg.solve(basis, reduced.reshape(-1, 2).T).T
    return np.linalg.norm(coordinates, axis=1).reshape(reduced.shape[:-1])


def cluster_bootstrap_mean(
    pair_values: np.ndarray, rng: np.random.Generator, draws: int,
) -> tuple[float, float]:
    pair_values = np.asarray(pair_values, dtype=float)
    if pair_values.size == 0:
        return math.nan, math.nan
    indices = rng.integers(0, pair_values.size, size=(draws, pair_values.size))
    estimates = pair_values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def slope(times: np.ndarray, values: np.ndarray, fit_horizon: float) -> float:
    mask = (times > 0.0) & (times <= fit_horizon) & np.isfinite(values) & (values > 0.0)
    if int(mask.sum()) < 3:
        return math.nan
    return float(np.polyfit(times[mask], np.log(values[mask]), 1)[0])


def paired_response_bootstrap(
    pair_deltas: np.ndarray,
    basis: np.ndarray,
    times: np.ndarray,
    fit_horizon: float,
    rng: np.random.Generator,
    draws: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    curves = np.empty((draws, len(times)), dtype=float)
    slopes = np.empty(draws, dtype=float)
    pair_count = pair_deltas.shape[0]
    for replicate in range(draws):
        indices = rng.integers(0, pair_count, size=pair_count)
        response = pair_deltas[indices].mean(axis=0)
        radii = modal_norm(response, basis)
        curves[replicate] = radii / radii[0]
        slopes[replicate] = slope(times, curves[replicate], fit_horizon)
    curve_low, curve_high = np.quantile(curves, [0.025, 0.975], axis=0)
    finite_slopes = slopes[np.isfinite(slopes)]
    if finite_slopes.size:
        slope_low, slope_high = np.quantile(finite_slopes, [0.025, 0.975])
    else:
        slope_low, slope_high = math.nan, math.nan
    return curve_low, curve_high, float(slope_low), float(slope_high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "finite_path_smoke.json")
    args = parser.parse_args()
    config, config_hash = load_config(args.config.resolve())
    runtime = runtime_directory()
    executable = compile_simulator(runtime)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    temporary = runtime / "tasks"
    temporary.mkdir(exist_ok=True)
    A = np.asarray(config["payoff"], dtype=float)
    classical_equilibrium = classical_interior_equilibrium(A)
    raw_frames: list[pd.DataFrame] = []
    metadata: dict[tuple[float, str], dict] = {}

    try:
        task_index = 0
        for beta in config["betas"]:
            beta = float(beta)
            for sample_size in config["sample_sizes"]:
                label = "inf" if sample_size is None else str(int(sample_size))
                rhs = named_rhs(
                    "exact", A, beta, sample_size,
                    kernel="fermi", sampling_mode="independent",
                )
                equilibrium, valid, residual = interior_equilibrium(rhs, classical_equilibrium)
                if not valid:
                    raise RuntimeError(f"equilibrium failed for beta={beta}, m={label}: {residual}")
                direction, eigenvalue, basis = tangent_direction(rhs, equilibrium)
                radius = float(config["perturbation_radius"])
                plus = equilibrium + radius * direction
                minus = equilibrium - radius * direction
                if np.min(plus) <= 0.0 or np.min(minus) <= 0.0:
                    raise RuntimeError("configured perturbation leaves the simplex interior")
                metadata[(beta, label)] = {
                    "rhs": rhs, "equilibrium": equilibrium, "direction": direction,
                    "eigenvalue": eigenvalue, "basis": basis, "plus": plus, "minus": minus,
                }
                for population in config["populations"]:
                    task_path = temporary / f"task_{task_index:03d}.csv"
                    command = [
                        str(executable), "--output", str(task_path),
                        "--beta", str(beta), "--m", label,
                        "--population", str(int(population)),
                        "--seeds", str(int(config["seeds_per_direction"])),
                        "--horizon", str(float(config["horizon"])),
                        "--record-dt", str(float(config["record_dt"])),
                        "--base-seed", str(int(config["base_seed"]) + 100000 * task_index),
                        "--x-plus", vector_text(plus), "--x-minus", vector_text(minus),
                    ]
                    subprocess.run(command, check=True)
                    frame = pd.read_csv(task_path)
                    frame.insert(0, "population", int(population))
                    frame.insert(0, "m", label)
                    frame.insert(0, "beta", beta)
                    frame.insert(0, "config_hash", config_hash)
                    frame.insert(0, "schema_version", config["schema_version"])
                    raw_frames.append(frame)
                    task_path.unlink()
                    task_index += 1
    finally:
        if temporary.exists() and not any(temporary.iterdir()):
            temporary.rmdir()

    raw = pd.concat(raw_frames, ignore_index=True)
    raw_path = results / f"{config['profile']}_e15_paths_raw.csv"
    raw.to_csv(raw_path.with_suffix(".csv.tmp"), index=False)
    raw_path.with_suffix(".csv.tmp").replace(raw_path)

    ensemble_rows: list[dict] = []
    summary_rows: list[dict] = []
    for (beta, m_label, population), group in raw.groupby(["beta", "m", "population"], sort=True):
        key = (float(beta), str(m_label))
        info = metadata[key]
        equilibrium = info["equilibrium"]
        basis = info["basis"]
        rhs = info["rhs"]
        times, deterministic_plus = model.rk4_integrate(
            rhs, info["plus"], float(config["horizon"]), float(config["record_dt"]),
            internal_dt=0.01,
        )
        _, deterministic_minus = model.rk4_integrate(
            rhs, info["minus"], float(config["horizon"]), float(config["record_dt"]),
            internal_dt=0.01,
        )
        deterministic_response = 0.5 * (deterministic_plus - deterministic_minus)
        deterministic_radius = modal_norm(deterministic_response, basis)
        deterministic_ratio = deterministic_radius / deterministic_radius[0]
        state_columns = ["x1", "x2", "x3"]
        plus_paths = group[group["direction"] == "plus"][["seed", "time", *state_columns]]
        minus_paths = group[group["direction"] == "minus"][["seed", "time", *state_columns]]
        paired = plus_paths.merge(
            minus_paths, on=["seed", "time"], validate="one_to_one", suffixes=("_plus", "_minus")
        ).sort_values(["seed", "time"])
        seed_ids = np.sort(paired["seed"].unique())
        observed_times = np.sort(paired["time"].unique())
        if len(observed_times) != len(times) or not np.allclose(observed_times, times):
            raise RuntimeError("finite-path recording times do not match deterministic times")
        pair_deltas = np.stack([
            0.5 * (
                paired.loc[paired["seed"] == seed, [f"{name}_plus" for name in state_columns]].to_numpy()
                - paired.loc[paired["seed"] == seed, [f"{name}_minus" for name in state_columns]].to_numpy()
            )
            for seed in seed_ids
        ])
        response = pair_deltas.mean(axis=0)
        response_radius = modal_norm(response, basis)
        response_ratio = response_radius / response_radius[0]
        bootstrap_rng = np.random.default_rng(
            int(config["base_seed"]) + 7919 * int(population) + 104729 * int(round(1000.0 * float(beta)))
            + (0 if str(m_label) == "inf" else int(m_label))
        )
        bootstrap_draws = int(config.get("bootstrap_draws", 2000))
        response_low, response_high, response_slope_low, response_slope_high = paired_response_bootstrap(
            pair_deltas, basis, times, float(config["fit_horizon"]), bootstrap_rng, bootstrap_draws
        )
        plus_mean = group[group["direction"] == "plus"].groupby("time")[state_columns].mean()
        minus_mean = group[group["direction"] == "minus"].groupby("time")[state_columns].mean()
        for index, time in enumerate(times):
            ensemble_rows.append({
                "schema_version": config["schema_version"], "config_hash": config_hash,
                "beta": beta, "m": m_label, "population": population, "time": time,
                "response_ratio": response_ratio[index],
                "response_pair_bootstrap_low": response_low[index],
                "response_pair_bootstrap_high": response_high[index],
                "deterministic_response_ratio": deterministic_ratio[index],
                "linear_response_ratio": math.exp(info["eigenvalue"] * time),
                "mean_plus_x1": plus_mean.iloc[index, 0], "mean_plus_x2": plus_mean.iloc[index, 1],
                "mean_plus_x3": plus_mean.iloc[index, 2], "mean_minus_x1": minus_mean.iloc[index, 0],
                "mean_minus_x2": minus_mean.iloc[index, 1], "mean_minus_x3": minus_mean.iloc[index, 2],
            })

        path_metrics = []
        for (direction, seed), path in group.groupby(["direction", "seed"]):
            path = path.sort_values("time")
            states = path[["x1", "x2", "x3"]].to_numpy()
            radii = np.linalg.norm(states - equilibrium, axis=1)
            exit_mask = radii >= float(config["exit_radius"])
            extinction_mask = path["min_count"].to_numpy() == 0
            path_metrics.append({
                "direction": direction,
                "seed": int(seed),
                "exited": bool(exit_mask.any()),
                "first_exit": float(path.loc[exit_mask, "time"].iloc[0]) if exit_mask.any() else math.nan,
                "extinct": bool(extinction_mask.any()),
                "first_extinction": float(path.loc[extinction_mask, "time"].iloc[0]) if extinction_mask.any() else math.nan,
                "max_radius": float(np.max(radii)),
            })
        metrics = pd.DataFrame(path_metrics)
        trials = len(metrics)
        seed_pairs = int(metrics["seed"].nunique())
        exits = int(metrics["exited"].sum())
        extinctions = int(metrics["extinct"].sum())
        exit_pair_values = metrics.groupby("seed")["exited"].mean().to_numpy()
        extinction_pair_values = metrics.groupby("seed")["extinct"].mean().to_numpy()
        exit_low, exit_high = cluster_bootstrap_mean(exit_pair_values, bootstrap_rng, bootstrap_draws)
        extinction_low, extinction_high = cluster_bootstrap_mean(
            extinction_pair_values, bootstrap_rng, bootstrap_draws
        )
        exit_times = [item["first_exit"] for item in path_metrics if item["exited"]]
        extinction_times = [item["first_extinction"] for item in path_metrics if item["extinct"]]
        summary_rows.append({
            "schema_version": config["schema_version"], "config_hash": config_hash,
            "beta": beta, "m": m_label, "population": population, "paths": trials,
            "seed_pairs": seed_pairs,
            "exact_max_real_eigenvalue": info["eigenvalue"],
            "ensemble_response_slope": slope(times, response_ratio, float(config["fit_horizon"])),
            "response_slope_pair_bootstrap_low": response_slope_low,
            "response_slope_pair_bootstrap_high": response_slope_high,
            "deterministic_response_slope": slope(times, deterministic_ratio, float(config["fit_horizon"])),
            "exit_probability": exits / trials, "exit_pair_bootstrap_low": exit_low,
            "exit_pair_bootstrap_high": exit_high,
            "median_first_exit_time": float(np.median(exit_times)) if exit_times else math.nan,
            "extinction_probability": extinctions / trials,
            "extinction_pair_bootstrap_low": extinction_low,
            "extinction_pair_bootstrap_high": extinction_high,
            "median_first_extinction_time": float(np.median(extinction_times)) if extinction_times else math.nan,
            "median_max_radius": float(np.median([item["max_radius"] for item in path_metrics])),
        })

    ensemble = pd.DataFrame(ensemble_rows)
    summary = pd.DataFrame(summary_rows)
    ensemble.to_csv(results / f"{config['profile']}_e15_ensemble_response.csv", index=False)
    summary.to_csv(results / f"{config['profile']}_e15_path_summary.csv", index=False)
    expected_rows = (
        len(config["betas"]) * len(config["sample_sizes"]) * len(config["populations"])
        * 2 * int(config["seeds_per_direction"])
        * (int(round(float(config["horizon"]) / float(config["record_dt"]))) + 1)
    )
    problems = []
    if len(raw) != expected_rows:
        problems.append(f"raw row count {len(raw)} != {expected_rows}")
    mass_error = float(np.max(np.abs(raw[["x1", "x2", "x3"]].sum(axis=1) - 1.0)))
    if mass_error > 1e-12:
        problems.append(f"mass error {mass_error}")
    if (raw[["x1", "x2", "x3"]] < -1e-12).any().any():
        problems.append("negative state")
    audit = {
        "status": "passed" if not problems else "failed",
        "profile": config["profile"], "config_hash": config_hash,
        "compiler": subprocess.check_output([shutil.which("g++"), "--version"], text=True).splitlines()[0],
        "rows": {"raw": len(raw), "ensemble": len(ensemble), "summary": len(summary)},
        "max_mass_error": mass_error, "problems": problems,
        "summary": summary.to_dict(orient="records"),
    }
    (results / f"{config['profile']}_e15_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    resolved_runtime = runtime.resolve()
    expected_parent = (ROOT.parents[2] / ".codex_tmp").resolve()
    if resolved_runtime.parent == expected_parent and resolved_runtime.name == "finite_path_v3":
        shutil.rmtree(resolved_runtime)
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
