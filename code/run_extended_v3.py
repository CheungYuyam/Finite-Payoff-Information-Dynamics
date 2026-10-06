"""Extended v3 evidence suite for Nature/Automatica/TAC-level evaluation.

The runner is deliberately streaming and separates deterministic mean-field
evidence from finite-population Monte Carlo evidence.  It never reads v1/v2
results.  Every output carries a configuration hash and is replaced atomically.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import root

import automatica_model as model


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> tuple[dict, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return config, digest


def atomic_csv(path: Path, fields: list[str], rows: Iterable[dict]) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
            if count % 250 == 0:
                handle.flush()
    temporary.replace(path)
    return count


def m_label(value: int | None | list[int | None]) -> str:
    if isinstance(value, list):
        return "[" + ",".join("inf" if q is None else str(q) for q in value) + "]"
    return "inf" if value is None else str(value)


def softmax_state(z: np.ndarray) -> np.ndarray:
    logits = np.r_[np.asarray(z, dtype=float), 0.0]
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    return weights / float(weights.sum())


def classical_interior_equilibrium(A: np.ndarray) -> np.ndarray:
    k = A.shape[0]
    augmented = np.block(
        [[A, -np.ones((k, 1))], [np.ones((1, k)), np.zeros((1, 1))]]
    )
    solution = np.linalg.solve(augmented, np.r_[np.zeros(k), 1.0])[:k]
    if np.min(solution) <= 0.0:
        raise ValueError("classical equilibrium is not interior")
    return solution


def interior_equilibrium(
    rhs: Callable[[np.ndarray], np.ndarray], guess: np.ndarray
) -> tuple[np.ndarray, bool, float]:
    z0 = np.log(guess[:-1] / guess[-1])
    solved = root(lambda z: rhs(softmax_state(z))[:-1], z0)
    state = softmax_state(solved.x)
    residual = float(np.linalg.norm(rhs(state)))
    valid = bool(solved.success and residual < 1e-8 and np.min(state) > 1e-8)
    return state, valid, residual


def simplex_jacobian(
    rhs: Callable[[np.ndarray], np.ndarray], state: np.ndarray, h: float = 1e-6
) -> np.ndarray:
    k = len(state)
    y = state[:-1]
    jacobian = np.empty((k - 1, k - 1))
    for column in range(k - 1):
        direction = np.zeros(k - 1)
        direction[column] = h
        plus_y = y + direction
        minus_y = y - direction
        plus = np.r_[plus_y, 1.0 - float(plus_y.sum())]
        minus = np.r_[minus_y, 1.0 - float(minus_y.sum())]
        jacobian[:, column] = (rhs(plus)[:-1] - rhs(minus)[:-1]) / (2.0 * h)
    return jacobian


def named_rhs(
    name: str,
    A: np.ndarray,
    beta: float,
    sample_size: int | None | list[int | None],
    *,
    kernel: str = "fermi",
    sampling_mode: str = "independent",
) -> Callable[[np.ndarray], np.ndarray]:
    if name == "classical":
        return lambda x: model.classical_rhs(x, A)
    if name == "variance_tail":
        return lambda x: model.tail_rhs(
            x,
            A,
            beta,
            sample_size,
            include_skewness=False,
            kernel=kernel,
            sampling_mode=sampling_mode,
        )
    if name == "moment3_tail":
        return lambda x: model.moment_tail_rhs(
            x,
            A,
            beta,
            sample_size,
            order=3,
            kernel=kernel,
            sampling_mode=sampling_mode,
        )
    if name == "moment5_tail":
        return lambda x: model.moment_tail_rhs(
            x,
            A,
            beta,
            sample_size,
            order=5,
            kernel=kernel,
            sampling_mode=sampling_mode,
        )
    if name == "exact":
        return lambda x: model.exact_rhs(
            x,
            A,
            beta,
            sample_size,
            kernel=kernel,
            sampling_mode=sampling_mode,
        )
    raise ValueError(name)


def constructive_outputs(config: dict, config_hash: str, output: Path) -> dict:
    case = config["constructive_case"]
    A = np.asarray(case["payoff"], dtype=float)
    beta = float(case["beta"])
    sample_size = case["m"]
    x0 = np.asarray(case["initial_state"], dtype=float)
    equilibrium_guess = classical_interior_equilibrium(A)
    model_names = [
        "classical", "variance_tail", "moment3_tail", "moment5_tail", "exact"
    ]
    equilibrium_rows = []
    trajectory_rows = []
    for name in model_names:
        rhs = named_rhs(name, A, beta, sample_size)
        equilibrium, valid, residual = interior_equilibrium(rhs, equilibrium_guess)
        eigenvalues = np.linalg.eigvals(simplex_jacobian(rhs, equilibrium)) if valid else []
        equilibrium_rows.append(
            {
                "schema_version": config["schema_version"],
                "config_hash": config_hash,
                "case": case["name"],
                "model": name,
                "valid": int(valid),
                "residual_l2": f"{residual:.17g}",
                "state": json.dumps(equilibrium.tolist(), separators=(",", ":")),
                "max_real_eigenvalue": (
                    f"{float(np.max(np.real(eigenvalues))):.17g}" if valid else ""
                ),
                "max_abs_imag_eigenvalue": (
                    f"{float(np.max(np.abs(np.imag(eigenvalues)))):.17g}" if valid else ""
                ),
            }
        )
        times, trajectory = model.rk4_integrate(
            rhs,
            x0,
            horizon=float(case["horizon"]),
            record_dt=float(case["record_dt"]),
            internal_dt=float(case["internal_dt"]),
        )
        for row, instant in enumerate(times):
            for strategy, value in enumerate(trajectory[row]):
                trajectory_rows.append(
                    {
                        "schema_version": config["schema_version"],
                        "config_hash": config_hash,
                        "case": case["name"],
                        "model": name,
                        "time": f"{float(instant):.12g}",
                        "strategy": strategy,
                        "share": f"{float(value):.17g}",
                    }
                )
        del trajectory
        gc.collect()
    equilibrium_count = atomic_csv(
        output / f"{config['profile']}_e3_equilibria.csv",
        [
            "schema_version", "config_hash", "case", "model", "valid",
            "residual_l2", "state", "max_real_eigenvalue",
            "max_abs_imag_eigenvalue",
        ],
        equilibrium_rows,
    )
    trajectory_count = atomic_csv(
        output / f"{config['profile']}_e3_trajectories.csv",
        ["schema_version", "config_hash", "case", "model", "time", "strategy", "share"],
        trajectory_rows,
    )
    return {"e3_equilibria": equilibrium_count, "e3_trajectories": trajectory_count}


def phase_rows(config: dict, config_hash: str):
    models = ["classical", "variance_tail", "moment3_tail", "moment5_tail"]
    for game in config["phase_games"]:
        A = np.asarray(game["payoff"], dtype=float)
        for state_id, values in enumerate(game["states"]):
            x = np.asarray(values, dtype=float)
            for kernel in config["kernels"]:
                for sampling_mode in config["sampling_modes"]:
                    for sample_size in config["phase_sample_sizes"]:
                        for beta_value in config["phase_betas"]:
                            beta = float(beta_value)
                            exact = model.exact_rhs(
                                x, A, beta, sample_size,
                                kernel=kernel, sampling_mode=sampling_mode,
                            )
                            exact_norm = float(np.linalg.norm(exact))
                            predictions = {
                                name: named_rhs(
                                    name, A, beta, sample_size,
                                    kernel=kernel, sampling_mode=sampling_mode,
                                )(x)
                                for name in models
                            }
                            skew_signal = float(
                                np.linalg.norm(
                                    predictions["moment3_tail"]
                                    - predictions["variance_tail"]
                                )
                            )
                            fifth_signal = float(
                                np.linalg.norm(
                                    predictions["moment5_tail"]
                                    - predictions["moment3_tail"]
                                )
                            )
                            for name, prediction in predictions.items():
                                error = float(np.linalg.norm(prediction - exact))
                                yield {
                                    "schema_version": config["schema_version"],
                                    "config_hash": config_hash,
                                    "game": game["name"],
                                    "state_id": state_id,
                                    "kernel": kernel,
                                    "sampling_mode": sampling_mode,
                                    "m": m_label(sample_size),
                                    "beta": f"{beta:.12g}",
                                    "model": name,
                                    "exact_norm": f"{exact_norm:.17g}",
                                    "error_l2": f"{error:.17g}",
                                    "relative_error": f"{error / max(exact_norm, 1e-14):.17g}",
                                    "skew_signal": f"{skew_signal:.17g}",
                                    "fifth_signal": f"{fifth_signal:.17g}",
                                    "mass_error": f"{abs(float(prediction.sum())):.17g}",
                                }


def random_stress_rows(config: dict, config_hash: str):
    settings = config["random_stress"]
    rng = np.random.default_rng(int(config["seed"]) + 101)
    dimensions = list(settings["dimensions"])
    betas = list(settings["betas"])
    sample_sizes = list(settings["sample_sizes"])
    kernels = list(config["kernels"])
    modes = list(config["sampling_modes"])
    alpha = float(settings["dirichlet_alpha"])
    for case_id in range(int(settings["cases"])):
        k = int(dimensions[case_id % len(dimensions)])
        A = rng.normal(size=(k, k))
        np.fill_diagonal(A, 0.0)
        scale = float(np.std(A))
        if scale > 0.0:
            A /= scale
        x = rng.dirichlet(np.full(k, alpha))
        beta = float(betas[int(rng.integers(len(betas)))])
        sample_size = sample_sizes[int(rng.integers(len(sample_sizes)))]
        kernel = kernels[int(rng.integers(len(kernels)))]
        sampling_mode = modes[int(rng.integers(len(modes)))]
        exact = model.exact_rhs(
            x, A, beta, sample_size, kernel=kernel, sampling_mode=sampling_mode
        )
        classical = model.classical_rhs(x, A)
        variance = model.tail_rhs(
            x, A, beta, sample_size, include_skewness=False,
            kernel=kernel, sampling_mode=sampling_mode,
        )
        moment3 = model.moment_tail_rhs(
            x, A, beta, sample_size, order=3,
            kernel=kernel, sampling_mode=sampling_mode,
        )
        moment5 = model.moment_tail_rhs(
            x, A, beta, sample_size, order=5,
            kernel=kernel, sampling_mode=sampling_mode,
        )
        exact_norm = float(np.linalg.norm(exact))
        game_hash = hashlib.sha256(A.tobytes()).hexdigest()[:12]
        errors = {
            "classical": float(np.linalg.norm(classical - exact)),
            "variance_tail": float(np.linalg.norm(variance - exact)),
            "moment3_tail": float(np.linalg.norm(moment3 - exact)),
            "moment5_tail": float(np.linalg.norm(moment5 - exact)),
        }
        yield {
            "schema_version": config["schema_version"],
            "config_hash": config_hash,
            "case_id": case_id,
            "game_hash": game_hash,
            "dimension": k,
            "kernel": kernel,
            "sampling_mode": sampling_mode,
            "m": sample_size,
            "beta": f"{beta:.12g}",
            "exact_norm": f"{exact_norm:.17g}",
            "error_classical": f"{errors['classical']:.17g}",
            "error_variance_tail": f"{errors['variance_tail']:.17g}",
            "error_moment3_tail": f"{errors['moment3_tail']:.17g}",
            "error_moment5_tail": f"{errors['moment5_tail']:.17g}",
            "relative_classical": f"{errors['classical'] / max(exact_norm, 1e-14):.17g}",
            "relative_variance_tail": f"{errors['variance_tail'] / max(exact_norm, 1e-14):.17g}",
            "relative_moment3_tail": f"{errors['moment3_tail'] / max(exact_norm, 1e-14):.17g}",
            "relative_moment5_tail": f"{errors['moment5_tail'] / max(exact_norm, 1e-14):.17g}",
            "skew_signal": f"{float(np.linalg.norm(moment3 - variance)):.17g}",
            "fifth_signal": f"{float(np.linalg.norm(moment5 - moment3)):.17g}",
            "state_min": f"{float(np.min(x)):.17g}",
        }


def heterogeneity_rows(config: dict, config_hash: str):
    case = config["constructive_case"]
    A = np.asarray(case["payoff"], dtype=float)
    x = np.asarray(case["initial_state"], dtype=float)
    beta = float(case["beta"])
    for sample_sizes in config["heterogeneous_samples"]:
        exact = model.exact_rhs(x, A, beta, sample_sizes)
        predictions = {
            "classical": model.classical_rhs(x, A),
            "variance_tail": model.tail_rhs(
                x, A, beta, sample_sizes, include_skewness=False
            ),
            "moment3_tail": model.moment_tail_rhs(
                x, A, beta, sample_sizes, order=3
            ),
            "moment5_tail": model.moment_tail_rhs(
                x, A, beta, sample_sizes, order=5
            ),
        }
        for name, prediction in predictions.items():
            yield {
                "schema_version": config["schema_version"],
                "config_hash": config_hash,
                "sample_sizes": m_label(sample_sizes),
                "model": name,
                "error_l2": f"{float(np.linalg.norm(prediction - exact)):.17g}",
                "rhs": json.dumps(prediction.tolist(), separators=(",", ":")),
                "exact_rhs": json.dumps(exact.tolist(), separators=(",", ":")),
            }


def finite_population_outputs(config: dict, config_hash: str, output: Path) -> dict:
    case = config["constructive_case"]
    settings = config["finite_population"]
    A = np.asarray(case["payoff"], dtype=float)
    x0 = np.asarray(case["initial_state"], dtype=float)
    beta = float(case["beta"])
    sample_size = case["m"]
    horizon = float(settings["horizon"])
    record_dt = float(settings["record_dt"])
    times, reference = model.rk4_integrate(
        named_rhs("exact", A, beta, sample_size),
        x0,
        horizon=horizon,
        record_dt=record_dt,
        internal_dt=min(0.005, record_dt / 10.0),
    )
    run_rows = []
    mean_rows = []
    for replacement in settings["replacement_modes"]:
        for population in settings["populations"]:
            trajectories = []
            for seed_index in range(int(settings["seeds"])):
                seed = int(config["seed"]) + 100000 * int(bool(replacement)) + 1000 * int(population) + seed_index
                _, trajectory = model.agent_trajectory(
                    A,
                    x0,
                    population=int(population),
                    horizon=horizon,
                    record_dt=record_dt,
                    beta=beta,
                    m=sample_size,
                    seed=seed,
                    replacement=bool(replacement),
                )
                trajectories.append(trajectory)
                absolute = np.sum(np.abs(trajectory - reference), axis=1)
                integrated = float(np.trapezoid(absolute, times) / max(horizon, 1e-12))
                endpoint = float(np.linalg.norm(trajectory[-1] - reference[-1], ord=1))
                run_rows.append(
                    {
                        "schema_version": config["schema_version"],
                        "config_hash": config_hash,
                        "population": population,
                        "replacement": int(bool(replacement)),
                        "seed_index": seed_index,
                        "integrated_l1": f"{integrated:.17g}",
                        "endpoint_l1": f"{endpoint:.17g}",
                    }
                )
            stack = np.asarray(trajectories)
            mean = stack.mean(axis=0)
            standard_error = stack.std(axis=0, ddof=1) / math.sqrt(stack.shape[0])
            for row, instant in enumerate(times):
                for strategy in range(A.shape[0]):
                    mean_rows.append(
                        {
                            "schema_version": config["schema_version"],
                            "config_hash": config_hash,
                            "population": population,
                            "replacement": int(bool(replacement)),
                            "time": f"{float(instant):.12g}",
                            "strategy": strategy,
                            "mean_share": f"{float(mean[row, strategy]):.17g}",
                            "se_share": f"{float(standard_error[row, strategy]):.17g}",
                            "exact_share": f"{float(reference[row, strategy]):.17g}",
                        }
                    )
            del stack, trajectories, mean, standard_error
            gc.collect()
    run_count = atomic_csv(
        output / f"{config['profile']}_e4_finite_n_runs.csv",
        ["schema_version", "config_hash", "population", "replacement", "seed_index", "integrated_l1", "endpoint_l1"],
        run_rows,
    )
    mean_count = atomic_csv(
        output / f"{config['profile']}_e4_finite_n_mean.csv",
        ["schema_version", "config_hash", "population", "replacement", "time", "strategy", "mean_share", "se_share", "exact_share"],
        mean_rows,
    )
    return {"e4_finite_n_runs": run_count, "e4_finite_n_mean": mean_count}


def benchmark_rows(config: dict, config_hash: str):
    settings = config["benchmark"]
    rng = np.random.default_rng(int(config["seed"]) + 707)
    maximum_pair_work = int(settings.get("max_support_pair_product", 2_000_000))
    for k in settings["dimensions"]:
        k = int(k)
        A = rng.normal(size=(k, k))
        x = rng.dirichlet(np.ones(k))
        for sample_size in settings["sample_sizes"]:
            support_bound = math.comb(int(sample_size) + k - 1, k - 1)
            for name in ("classical", "moment3_tail", "moment5_tail", "exact"):
                estimated_pair_work = support_bound**2 if name == "exact" else 1
                if estimated_pair_work > maximum_pair_work:
                    yield {
                        "schema_version": config["schema_version"],
                        "config_hash": config_hash,
                        "dimension": k,
                        "m": sample_size,
                        "model": name,
                        "support_bound": support_bound,
                        "status": "skipped_work_guard",
                        "median_seconds": "",
                    }
                    continue
                timings = []
                rhs = named_rhs(name, A, 0.5, sample_size)
                for _ in range(int(settings["repeats"])):
                    started = time.perf_counter()
                    rhs(x)
                    timings.append(time.perf_counter() - started)
                yield {
                    "schema_version": config["schema_version"],
                    "config_hash": config_hash,
                    "dimension": k,
                    "m": sample_size,
                    "model": name,
                    "support_bound": support_bound,
                    "status": "measured",
                    "median_seconds": f"{float(np.median(timings)):.17g}",
                }
                gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "extended_smoke.json"
    )
    args = parser.parse_args()
    config, config_hash = load_config(args.config.resolve())
    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    profile = config["profile"]
    started = time.perf_counter()
    counts = constructive_outputs(config, config_hash, output)
    counts["e5_phase_map"] = atomic_csv(
        output / f"{profile}_e5_phase_map.csv",
        [
            "schema_version", "config_hash", "game", "state_id", "kernel",
            "sampling_mode", "m", "beta", "model", "exact_norm", "error_l2",
            "relative_error", "skew_signal", "fifth_signal", "mass_error",
        ],
        phase_rows(config, config_hash),
    )
    counts["e6_random_stress"] = atomic_csv(
        output / f"{profile}_e6_random_stress.csv",
        [
            "schema_version", "config_hash", "case_id", "game_hash", "dimension",
            "kernel", "sampling_mode", "m", "beta", "exact_norm",
            "error_classical", "error_variance_tail", "error_moment3_tail",
            "error_moment5_tail", "relative_classical", "relative_variance_tail",
            "relative_moment3_tail", "relative_moment5_tail", "skew_signal",
            "fifth_signal", "state_min",
        ],
        random_stress_rows(config, config_hash),
    )
    counts["e7_heterogeneity"] = atomic_csv(
        output / f"{profile}_e7_heterogeneity.csv",
        ["schema_version", "config_hash", "sample_sizes", "model", "error_l2", "rhs", "exact_rhs"],
        heterogeneity_rows(config, config_hash),
    )
    counts.update(finite_population_outputs(config, config_hash, output))
    counts["e8_benchmark"] = atomic_csv(
        output / f"{profile}_e8_benchmark.csv",
        ["schema_version", "config_hash", "dimension", "m", "model", "support_bound", "status", "median_seconds"],
        benchmark_rows(config, config_hash),
    )
    manifest = {
        "schema_version": config["schema_version"],
        "profile": profile,
        "config_hash": config_hash,
        "config_path": str(args.config.resolve()),
        "rows": counts,
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = output / f"{profile}_extended_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
