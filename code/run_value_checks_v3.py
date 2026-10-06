"""High-value checks: finite-N generator reversal and stability continuation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

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


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def tangent_growth(rhs, center: np.ndarray, radii: list[float], angles: int) -> tuple[float, float, int]:
    basis_1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    basis_2 = np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0)
    baseline = rhs(center)
    values = []
    for radius in radii:
        for angle in np.linspace(0.0, 2.0 * np.pi, angles, endpoint=False):
            delta = radius * (np.cos(angle) * basis_1 + np.sin(angle) * basis_2)
            state = center + delta
            if np.min(state) <= 0.0:
                continue
            values.append(float(delta @ (rhs(state) - baseline) / (delta @ delta)))
    return float(np.mean(values)), float(np.std(values, ddof=1)), len(values)


def finite_generator_jacobian(counts: np.ndarray, rhs_counts, step_counts: int) -> np.ndarray:
    """Centered simplex Jacobian using count-preserving transfers."""
    jacobian = np.empty((2, 2), dtype=float)
    for column in range(2):
        plus = counts.copy(); minus = counts.copy()
        plus[column] += step_counts; plus[2] -= step_counts
        minus[column] -= step_counts; minus[2] += step_counts
        if np.min(plus) < 0 or np.min(minus) < 0:
            raise ValueError("finite-difference transfer left the count simplex")
        jacobian[:, column] = (
            rhs_counts(plus)[:2] - rhs_counts(minus)[:2]
        ) / (2.0 * step_counts / counts.sum())
    return jacobian


def generator_rows(config: dict, config_hash: str) -> list[dict]:
    settings = config["finite_generator"]
    A = np.asarray(config["payoff"], dtype=float)
    beta = float(settings["beta"]); sample_size = int(settings["m"])
    kernel = settings["kernel"]; sampling_mode = settings["sampling_mode"]
    classical_eq = classical_interior_equilibrium(A)
    exact_rhs = named_rhs("exact", A, beta, sample_size, kernel=kernel, sampling_mode=sampling_mode)
    exact_eq, valid, residual = interior_equilibrium(exact_rhs, classical_eq)
    if not valid:
        raise RuntimeError(f"exact equilibrium was not resolved: residual={residual}")
    classical_rhs = named_rhs("classical", A, beta, sample_size, kernel=kernel, sampling_mode=sampling_mode)
    rows = []
    for name, center, rhs in (("classical", classical_eq, classical_rhs), ("exact", exact_eq, exact_rhs)):
        mean, spread, points = tangent_growth(rhs, center, settings["radii"], int(settings["angles"]))
        eigen = np.linalg.eigvals(simplex_jacobian(rhs, center))
        rows.append({
            "schema_version": config["schema_version"], "config_hash": config_hash,
            "model": name, "population": "inf", "radial_growth_mean": mean,
            "radial_growth_sd": spread, "ring_points": points,
            "max_real_eigenvalue": float(np.max(np.real(eigen))),
            "eigenvalue_step_spread": 0.0,
            "center_residual": float(np.linalg.norm(rhs(center))),
        })
    for population in settings["populations"]:
        population = int(population)
        center_counts = model.initial_counts(exact_eq, population)
        center = center_counts / population
        def rhs_counts(counts: np.ndarray) -> np.ndarray:
            return model.finite_population_generator_rhs(
                counts, A, beta, sample_size, kernel=kernel, sampling_mode=sampling_mode
            )
        def finite_rhs(state: np.ndarray) -> np.ndarray:
            counts = model.initial_counts(state, population)
            return rhs_counts(counts)
        mean, spread, points = tangent_growth(finite_rhs, center, settings["radii"], int(settings["angles"]))
        step_counts = sorted(set(max(1, int(round(population * fraction))) for fraction in settings["jacobian_step_fractions"]))
        stability_values = [
            float(np.max(np.real(np.linalg.eigvals(finite_generator_jacobian(center_counts, rhs_counts, step)))))
            for step in step_counts
        ]
        rows.append({
            "schema_version": config["schema_version"], "config_hash": config_hash,
            "model": "finite_N_generator", "population": population,
            "radial_growth_mean": mean, "radial_growth_sd": spread,
            "ring_points": points, "max_real_eigenvalue": float(np.median(stability_values)),
            "eigenvalue_step_spread": float(np.ptp(stability_values)),
            "center_residual": float(np.linalg.norm(finite_rhs(center))),
        })
    return rows


def continuation_rows(config: dict, config_hash: str) -> list[dict]:
    settings = config["continuation"]
    A = np.asarray(config["payoff"], dtype=float)
    classical_eq = classical_interior_equilibrium(A)
    rows = []
    for kernel in settings["kernels"]:
        for sampling_mode in settings["sampling_modes"]:
            for sample_size in settings["sample_sizes"]:
                guess = classical_eq.copy()
                for beta_value in settings["betas"]:
                    beta = float(beta_value)
                    rhs = named_rhs("exact", A, beta, sample_size, kernel=kernel, sampling_mode=sampling_mode)
                    candidates = []
                    for candidate_guess in (guess, classical_eq, np.full(3, 1.0 / 3.0)):
                        equilibrium, valid, residual = interior_equilibrium(rhs, candidate_guess)
                        if valid:
                            candidates.append((float(np.linalg.norm(equilibrium - guess)), equilibrium, residual))
                    if candidates:
                        _, equilibrium, residual = min(candidates, key=lambda item: item[0])
                        guess = equilibrium
                        stability_values = []
                        for step in settings["jacobian_steps"]:
                            eigen = np.linalg.eigvals(simplex_jacobian(rhs, equilibrium, h=float(step)))
                            stability_values.append(float(np.max(np.real(eigen))))
                        stability = float(np.median(stability_values))
                        sensitivity = float(np.ptp(stability_values))
                        valid = True
                    else:
                        equilibrium = guess; residual = float(np.linalg.norm(rhs(guess)))
                        stability = np.nan; sensitivity = np.nan; valid = False
                    rows.append({
                        "schema_version": config["schema_version"], "config_hash": config_hash,
                        "kernel": kernel, "sampling_mode": sampling_mode,
                        "m": "inf" if sample_size is None else sample_size,
                        "inverse_m": 0.0 if sample_size is None else 1.0 / float(sample_size),
                        "beta": beta, "valid": int(valid), "residual_l2": residual,
                        "max_real_eigenvalue": stability,
                        "jacobian_step_spread": sensitivity,
                        "equilibrium": json.dumps(equilibrium.tolist(), separators=(",", ":")),
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "value_smoke.json")
    args = parser.parse_args()
    config, config_hash = load_config(args.config.resolve())
    results = ROOT / "results"; results.mkdir(exist_ok=True)
    profile = config["profile"]
    generator = generator_rows(config, config_hash)
    continuation = continuation_rows(config, config_hash)
    write_csv(results / f"{profile}_e13_finite_generator.csv", generator)
    write_csv(results / f"{profile}_e14_stability_continuation.csv", continuation)
    valid = [row for row in continuation if row["valid"] == 1]
    audit = {
        "status": "passed" if len(valid) == len(continuation) else "failed",
        "profile": profile, "config_hash": config_hash,
        "rows": {"e13": len(generator), "e14": len(continuation)},
        "finite_generator": generator,
        "continuation_valid_rate": len(valid) / len(continuation),
        "max_jacobian_step_spread": max(float(row["jacobian_step_spread"]) for row in valid),
    }
    path = results / f"{profile}_value_audit.json"
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
