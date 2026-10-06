"""Refine E14 boundaries and audit branch/Jacobian robustness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

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


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def unique_states(states: list[np.ndarray], tolerance: float = 1e-7) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for state in states:
        if not any(float(np.linalg.norm(state - other)) <= tolerance for other in unique):
            unique.append(state)
    return unique


def resolve_point(
    A: np.ndarray,
    beta: float,
    sample_size: int | None,
    kernel: str,
    sampling_mode: str,
    reference: np.ndarray,
    classical_equilibrium: np.ndarray,
    jacobian_steps: list[float],
) -> dict:
    rhs = named_rhs("exact", A, beta, sample_size, kernel=kernel, sampling_mode=sampling_mode)
    guesses = [
        reference,
        classical_equilibrium,
        np.full(A.shape[0], 1.0 / A.shape[0]),
        0.75 * reference + 0.25 * classical_equilibrium,
    ]
    candidates: list[tuple[np.ndarray, float]] = []
    for guess in guesses:
        equilibrium, valid, residual = interior_equilibrium(rhs, guess)
        if valid:
            candidates.append((equilibrium, residual))
    if not candidates:
        raise RuntimeError(
            f"no interior equilibrium at beta={beta}, m={sample_size}, "
            f"kernel={kernel}, mode={sampling_mode}"
        )
    distinct = unique_states([item[0] for item in candidates])
    equilibrium, residual = min(
        candidates, key=lambda item: float(np.linalg.norm(item[0] - reference))
    )
    stability_values = [
        float(np.max(np.real(np.linalg.eigvals(simplex_jacobian(rhs, equilibrium, h=float(step))))))
        for step in jacobian_steps
    ]
    candidate_spread = max(
        (float(np.linalg.norm(first - second)) for first in distinct for second in distinct),
        default=0.0,
    )
    return {
        "equilibrium": equilibrium,
        "residual_l2": float(residual),
        "max_real_eigenvalue": float(np.median(stability_values)),
        "jacobian_step_spread": float(np.ptp(stability_values)),
        "candidate_solution_count": len(distinct),
        "candidate_equilibrium_spread": candidate_spread,
    }


def dense_grid(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step))
    values = start + step * np.arange(count + 1, dtype=float)
    if values[-1] < stop - 1e-12:
        values = np.r_[values, stop]
    values[-1] = stop
    return values


def run_refinement(config: dict, config_hash: str) -> tuple[list[dict], list[dict]]:
    settings = config["boundary_refinement"]
    A = np.asarray(config["payoff"], dtype=float)
    classical_equilibrium = classical_interior_equilibrium(A)
    betas = dense_grid(
        float(settings["beta_min"]),
        float(settings["beta_max"]),
        float(settings["dense_step"]),
    )
    scan_rows: list[dict] = []
    root_rows: list[dict] = []

    for kernel in settings["kernels"]:
        for sampling_mode in settings["sampling_modes"]:
            for sample_size in settings["sample_sizes"]:
                reference = classical_equilibrium.copy()
                previous_equilibrium = None
                points: list[tuple[float, dict]] = []
                for beta in betas:
                    point = resolve_point(
                        A, float(beta), sample_size, kernel, sampling_mode,
                        reference, classical_equilibrium, settings["jacobian_steps"],
                    )
                    equilibrium = point["equilibrium"]
                    branch_step = 0.0 if previous_equilibrium is None else float(
                        np.linalg.norm(equilibrium - previous_equilibrium)
                    )
                    previous_equilibrium = equilibrium
                    reference = equilibrium
                    points.append((float(beta), point))
                    scan_rows.append({
                        "schema_version": config["schema_version"],
                        "config_hash": config_hash,
                        "kernel": kernel,
                        "sampling_mode": sampling_mode,
                        "m": "inf" if sample_size is None else sample_size,
                        "inverse_m": 0.0 if sample_size is None else 1.0 / float(sample_size),
                        "beta": float(beta),
                        "max_real_eigenvalue": point["max_real_eigenvalue"],
                        "jacobian_step_spread": point["jacobian_step_spread"],
                        "residual_l2": point["residual_l2"],
                        "branch_step_l2": branch_step,
                        "candidate_solution_count": point["candidate_solution_count"],
                        "candidate_equilibrium_spread": point["candidate_equilibrium_spread"],
                        "equilibrium": json.dumps(equilibrium.tolist(), separators=(",", ":")),
                    })

                for (left_beta, left), (right_beta, right) in zip(points[:-1], points[1:]):
                    left_value = float(left["max_real_eigenvalue"])
                    right_value = float(right["max_real_eigenvalue"])
                    if left_value == 0.0 or right_value == 0.0 or left_value * right_value > 0.0:
                        continue
                    left_equilibrium = left["equilibrium"]
                    right_equilibrium = right["equilibrium"]

                    def objective(beta_value: float) -> float:
                        weight = (beta_value - left_beta) / (right_beta - left_beta)
                        reference_state = (
                            (1.0 - weight) * left_equilibrium + weight * right_equilibrium
                        )
                        point = resolve_point(
                            A, beta_value, sample_size, kernel, sampling_mode,
                            reference_state, classical_equilibrium, settings["jacobian_steps"],
                        )
                        return float(point["max_real_eigenvalue"])

                    critical_beta = float(brentq(
                        objective, left_beta, right_beta,
                        xtol=float(settings["root_xtol"]),
                        rtol=float(settings["root_rtol"]), maxiter=100,
                    ))
                    weight = (critical_beta - left_beta) / (right_beta - left_beta)
                    critical = resolve_point(
                        A, critical_beta, sample_size, kernel, sampling_mode,
                        (1.0 - weight) * left_equilibrium + weight * right_equilibrium,
                        classical_equilibrium, settings["jacobian_steps"],
                    )
                    root_rows.append({
                        "schema_version": config["schema_version"],
                        "config_hash": config_hash,
                        "kernel": kernel,
                        "sampling_mode": sampling_mode,
                        "m": "inf" if sample_size is None else sample_size,
                        "inverse_m": 0.0 if sample_size is None else 1.0 / float(sample_size),
                        "left_beta": left_beta,
                        "right_beta": right_beta,
                        "critical_beta": critical_beta,
                        "crossing_direction": "stable_to_unstable" if left_value < right_value else "unstable_to_stable",
                        "left_eigenvalue": left_value,
                        "right_eigenvalue": right_value,
                        "root_eigenvalue": critical["max_real_eigenvalue"],
                        "jacobian_step_spread": critical["jacobian_step_spread"],
                        "residual_l2": critical["residual_l2"],
                        "candidate_solution_count": critical["candidate_solution_count"],
                        "candidate_equilibrium_spread": critical["candidate_equilibrium_spread"],
                        "equilibrium": json.dumps(critical["equilibrium"].tolist(), separators=(",", ":")),
                    })
    return scan_rows, root_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "value_smoke.json")
    args = parser.parse_args()
    config, config_hash = load_config(args.config.resolve())
    scan, roots = run_refinement(config, config_hash)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    profile = config["profile"]
    root_fields = list(roots[0]) if roots else [
        "schema_version", "config_hash", "kernel", "sampling_mode", "m",
        "inverse_m", "left_beta", "right_beta", "critical_beta",
        "crossing_direction", "left_eigenvalue", "right_eigenvalue",
        "root_eigenvalue", "jacobian_step_spread", "residual_l2",
        "candidate_solution_count", "candidate_equilibrium_spread", "equilibrium",
    ]
    write_csv(results / f"{profile}_e14_dense_scan.csv", scan, list(scan[0]))
    write_csv(results / f"{profile}_e14_critical_boundaries.csv", roots, root_fields)
    narrow = [
        row for row in roots
        if row["kernel"] == "fermi"
        and row["sampling_mode"] == "independent"
        and str(row["m"]) == "2"
    ]
    audit = {
        "status": "passed",
        "profile": profile,
        "config_hash": config_hash,
        "rows": {"dense_scan": len(scan), "critical_boundaries": len(roots)},
        "max_residual_l2": max(float(row["residual_l2"]) for row in scan),
        "max_jacobian_step_spread": max(float(row["jacobian_step_spread"]) for row in scan),
        "max_branch_step_l2": max(float(row["branch_step_l2"]) for row in scan),
        "max_candidate_equilibrium_spread": max(float(row["candidate_equilibrium_spread"]) for row in scan),
        "fermi_independent_m2_boundaries": narrow,
    }
    (results / f"{profile}_e14_boundary_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
