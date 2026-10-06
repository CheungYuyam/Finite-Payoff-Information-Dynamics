"""Nature-level evidence suite for the finite-feedback replicator model.

The suite adds three kinds of evidence that are not supplied by a larger
parameter sweep alone:

E9  population-level frequency of qualitative stability mistakes;
E10 an out-of-sample test against the best fitted effective classical game;
E11 finite-population/macro closure across mechanisms and games.

All files are written atomically and are independent of v1/v2 outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Iterable

import numpy as np

import automatica_model as model
from run_extended_v3 import (
    classical_interior_equilibrium,
    interior_equilibrium,
    named_rhs,
    simplex_jacobian,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("classical", "variance_tail", "moment3_tail", "moment5_tail", "exact")


def load_config(path: Path) -> tuple[dict, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
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


def random_coexistence_game(
    rng: np.random.Generator,
    lognormal_sigma: float,
    *,
    near_boundary: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Draw a normalized 3x3 game with a non-degenerate interior equilibrium.

    Half of E9 can be sampled without a stability restriction and half close
    to the classical stability boundary.  Keeping the strata explicit avoids
    presenting a boundary-enriched ensemble as an unconditional frequency.
    """
    for _ in range(20_000):
        A = rng.normal(size=(3, 3))
        row_scales = rng.lognormal(mean=0.0, sigma=lognormal_sigma, size=3)
        A *= row_scales[:, None]
        A -= float(A.mean())
        scale = float(np.std(A))
        if scale < 1e-8:
            continue
        A /= scale
        try:
            equilibrium = classical_interior_equilibrium(A)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if float(np.min(equilibrium)) <= 0.05:
            continue
        classical = lambda x: model.classical_rhs(x, A)
        max_real = float(np.max(np.real(np.linalg.eigvals(simplex_jacobian(classical, equilibrium)))))
        if near_boundary and abs(max_real) > 0.08:
            continue
        return A, equilibrium, max_real
    raise RuntimeError("failed to draw an interior 3x3 game")


def constructive_neighborhood_game(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    base = np.asarray(
        [[0.0, -0.37061410, 2.77335386], [0.24483140, 0.0, -2.70698466], [-2.08439346, 3.52730968, 0.0]],
        dtype=float,
    )
    for _ in range(20_000):
        A = base + rng.normal(scale=float(rng.uniform(0.02, 0.35)), size=(3, 3))
        try:
            equilibrium = classical_interior_equilibrium(A)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if float(np.min(equilibrium)) <= 0.03:
            continue
        classical = lambda x: model.classical_rhs(x, A)
        max_real = float(np.max(np.real(np.linalg.eigvals(simplex_jacobian(classical, equilibrium)))))
        return A, equilibrium, max_real
    raise RuntimeError("failed to draw from the constructive neighborhood")


def stability_value(rhs: Callable[[np.ndarray], np.ndarray], state: np.ndarray) -> float:
    return float(np.max(np.real(np.linalg.eigvals(simplex_jacobian(rhs, state)))))


def stability_label(value: float, tolerance: float) -> int:
    if value < -tolerance:
        return -1
    if value > tolerance:
        return 1
    return 0


def normalized_error(prediction: np.ndarray, exact: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - exact) / max(np.linalg.norm(exact), 1e-12))


def qualitative_rows(config: dict, config_hash: str):
    settings = config["qualitative_map"]
    rng = np.random.default_rng(int(config["seed"]) + 901)
    betas = list(settings["betas"])
    sample_sizes = list(settings["sample_sizes"])
    kernels = list(settings["kernels"])
    modes = list(settings["sampling_modes"])
    tolerance = float(settings["stability_tolerance"])
    for case_id in range(int(settings["cases"])):
        stratum = case_id % 3
        if stratum == 2:
            ensemble = "constructive_neighborhood"
            A, classical_equilibrium, generated_classical_stability = constructive_neighborhood_game(rng)
        else:
            near_boundary = bool(stratum)
            ensemble = "near_boundary" if near_boundary else "unconditional_interior"
            A, classical_equilibrium, generated_classical_stability = random_coexistence_game(
                rng,
                float(settings["payoff_lognormal_sigma"]),
                near_boundary=near_boundary,
            )
        beta = float(betas[int(rng.integers(len(betas)))])
        sample_size = int(sample_sizes[int(rng.integers(len(sample_sizes)))])
        kernel = kernels[int(rng.integers(len(kernels)))]
        sampling_mode = modes[int(rng.integers(len(modes)))]
        probe = rng.dirichlet(np.full(3, 1.5))

        rhs_functions = {
            name: named_rhs(
                name, A, beta, sample_size, kernel=kernel, sampling_mode=sampling_mode
            )
            for name in MODEL_NAMES
        }
        exact_probe = rhs_functions["exact"](probe)
        pair_variances = []
        pair_thirds = []
        pair_weights = []
        for i in range(2):
            for j in range(i + 1, 3):
                _, variance, third = model.payoff_difference_moments(
                    A, probe, i, j, sample_size, sample_size, sampling_mode=sampling_mode
                )
                pair_variances.append(max(variance, 0.0))
                pair_thirds.append(third)
                pair_weights.append(probe[i] * probe[j])
        weights = np.asarray(pair_weights)
        weights /= float(weights.sum())
        variance = float(weights @ np.asarray(pair_variances))
        third = float(weights @ np.asarray(pair_thirds))
        sigma = float(np.sqrt(variance))
        chi = beta * sigma
        skewness = float(third / max(sigma**3, 1e-15))

        equilibria: dict[str, tuple[np.ndarray, bool, float]] = {}
        stabilities: dict[str, float] = {}
        for name, rhs in rhs_functions.items():
            equilibrium, valid, residual = interior_equilibrium(rhs, classical_equilibrium)
            equilibria[name] = (equilibrium, valid, residual)
            stabilities[name] = stability_value(rhs, equilibrium) if valid else np.nan

        exact_state, exact_valid, _ = equilibria["exact"]
        exact_label = stability_label(stabilities["exact"], tolerance) if exact_valid else 9
        payload = {
            "schema_version": config["schema_version"],
            "config_hash": config_hash,
            "case_id": case_id,
            "ensemble": ensemble,
            "game": json.dumps(A.tolist(), separators=(",", ":")),
            "beta": f"{beta:.12g}",
            "m": sample_size,
            "kernel": kernel,
            "sampling_mode": sampling_mode,
            "probe_state": json.dumps(probe.tolist(), separators=(",", ":")),
            "chi": f"{chi:.17g}",
            "feedback_skewness": f"{skewness:.17g}",
            "generated_classical_stability": f"{generated_classical_stability:.17g}",
            "exact_valid": int(exact_valid),
            "exact_equilibrium": json.dumps(exact_state.tolist(), separators=(",", ":")),
            "exact_stability": f"{stabilities['exact']:.17g}",
            "exact_label": exact_label,
        }
        for name in MODEL_NAMES[:-1]:
            state, valid, _ = equilibria[name]
            label = stability_label(stabilities[name], tolerance) if valid else 9
            payload[f"{name}_valid"] = int(valid)
            payload[f"{name}_equilibrium_shift"] = (
                f"{float(np.linalg.norm(state - exact_state)):.17g}"
                if valid and exact_valid else ""
            )
            payload[f"{name}_stability"] = f"{stabilities[name]:.17g}"
            payload[f"{name}_label"] = label
            payload[f"{name}_stability_mismatch"] = int(
                valid and exact_valid and label != exact_label
            )
            payload[f"{name}_probe_relative_error"] = (
                f"{normalized_error(rhs_functions[name](probe), exact_probe):.17g}"
            )
        yield payload


def classical_design_row(x: np.ndarray, k: int) -> np.ndarray:
    """Linear map from vec(A) to the first k-1 replicator components."""
    rows = np.zeros((k - 1, k * k))
    for i in range(k - 1):
        for a in range(k):
            for b in range(k):
                coefficient = x[i] * x[b] * ((1.0 if a == i else 0.0) - x[a])
                rows[i, a * k + b] = coefficient
    return rows


def fit_effective_game(states: np.ndarray, targets: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    k = states.shape[1]
    design = np.vstack([classical_design_row(state, k) for state in states])
    response = targets[:, :-1].reshape(-1)
    normal = design.T @ design + ridge * np.eye(k * k)
    coefficients = np.linalg.solve(normal, design.T @ response)
    return coefficients.reshape(k, k)


def fit_time_scale(classical: np.ndarray, targets: np.ndarray) -> float:
    numerator = float(np.sum(classical * targets))
    denominator = float(np.sum(classical * classical))
    return numerator / max(denominator, 1e-24)


def relative_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    numerator = float(np.mean(np.sum((predictions - targets) ** 2, axis=1)))
    denominator = float(np.mean(np.sum(targets**2, axis=1)))
    return float(np.sqrt(numerator / max(denominator, 1e-24)))


def nonequivalence_rows(config: dict, config_hash: str):
    settings = config["non_equivalence"]
    rng = np.random.default_rng(int(config["seed"]) + 1001)
    base_A = np.asarray(
        [[0.0, -0.37061410, 2.77335386], [0.24483140, 0.0, -2.70698466], [-2.08439346, 3.52730968, 0.0]],
        dtype=float,
    )
    conditions = [
        (0.15, 1, "fermi", "independent"),
        (0.4, 2, "fermi", "common"),
        (0.15, 2, "arctan", "independent"),
        (0.3, 5, "arctan", "common"),
    ]
    for case_id in range(int(settings["cases"])):
        beta, sample_size, kernel, sampling_mode = conditions[case_id % len(conditions)]
        perturbation = rng.normal(scale=0.18, size=(3, 3))
        np.fill_diagonal(perturbation, 0.0)
        A = base_A + perturbation
        train = rng.dirichlet(np.full(3, float(settings["dirichlet_alpha"])), size=int(settings["train_states"]))
        test = rng.dirichlet(np.full(3, float(settings["dirichlet_alpha"])), size=int(settings["test_states"]))
        exact_rhs = named_rhs("exact", A, beta, sample_size, kernel=kernel, sampling_mode=sampling_mode)
        train_targets = np.asarray([exact_rhs(state) for state in train])
        test_targets = np.asarray([exact_rhs(state) for state in test])
        fitted_A = fit_effective_game(train, train_targets)
        train_classical = np.asarray([model.classical_rhs(state, A) for state in train])
        test_classical = np.asarray([model.classical_rhs(state, A) for state in test])
        fitted_scale = fit_time_scale(train_classical, train_targets)

        predictions = {
            "classical": test_classical,
            "best_time_scaled_classical": fitted_scale * test_classical,
            "best_effective_classical": np.asarray([model.classical_rhs(state, fitted_A) for state in test]),
            "moment3_tail": np.asarray([
                model.moment_tail_rhs(state, A, beta, sample_size, order=3, kernel=kernel, sampling_mode=sampling_mode)
                for state in test
            ]),
            "moment5_tail": np.asarray([
                model.moment_tail_rhs(state, A, beta, sample_size, order=5, kernel=kernel, sampling_mode=sampling_mode)
                for state in test
            ]),
        }
        train_fitted = np.asarray([model.classical_rhs(state, fitted_A) for state in train])
        for name, prediction in predictions.items():
            yield {
                "schema_version": config["schema_version"],
                "config_hash": config_hash,
                "case_id": case_id,
                "beta": beta,
                "m": sample_size,
                "kernel": kernel,
                "sampling_mode": sampling_mode,
                "model": name,
                "train_relative_rmse": f"{relative_rmse(train_fitted, train_targets):.17g}" if name == "best_effective_classical" else "",
                "test_relative_rmse": f"{relative_rmse(prediction, test_targets):.17g}",
                "fitted_payoff": json.dumps(fitted_A.tolist(), separators=(",", ":")) if name == "best_effective_classical" else "",
                "fitted_time_scale": f"{fitted_scale:.17g}" if name == "best_time_scaled_classical" else "",
            }


def paired_accuracy_rows(config: dict, config_hash: str):
    """Balanced, paired stress test over a frozen set of games and states."""
    settings = config["paired_accuracy"]
    rng = np.random.default_rng(int(config["seed"]) + 1101)
    dimensions = list(settings["dimensions"])
    for base_case in range(int(settings["base_cases"])):
        k = int(dimensions[base_case % len(dimensions)])
        A = rng.normal(size=(k, k))
        A -= float(A.mean())
        A /= max(float(np.std(A)), 1e-12)
        state = rng.dirichlet(np.full(k, float(settings["dirichlet_alpha"])))
        for beta in settings["betas"]:
            for sample_size in settings["sample_sizes"]:
                for kernel in settings["kernels"]:
                    for sampling_mode in settings["sampling_modes"]:
                        exact = model.exact_rhs(
                            state, A, float(beta), int(sample_size), kernel=kernel, sampling_mode=sampling_mode
                        )
                        exact_norm = float(np.linalg.norm(exact))
                        pair_variances = []
                        pair_second_moments = []
                        pair_weights = []
                        for i in range(k - 1):
                            for j in range(i + 1, k):
                                mean, variance, _ = model.payoff_difference_moments(
                                    A, state, i, j, int(sample_size), int(sample_size), sampling_mode=sampling_mode
                                )
                                pair_variances.append(max(variance, 0.0))
                                pair_second_moments.append(max(variance + mean * mean, 0.0))
                                pair_weights.append(state[i] * state[j])
                        weights = np.asarray(pair_weights)
                        weights /= max(float(weights.sum()), 1e-15)
                        chi = float(beta) * float(np.sqrt(weights @ np.asarray(pair_variances)))
                        # The Taylor expansion is controlled by the magnitude of the
                        # complete comparison signal, not by its stochastic variance
                        # alone. psi therefore includes both the mean payoff gap and
                        # finite-feedback fluctuations through E[(Delta pi)^2].
                        psi = float(beta) * float(np.sqrt(weights @ np.asarray(pair_second_moments)))
                        predictions = {
                            "classical": model.classical_rhs(state, A),
                            "variance_tail": model.tail_rhs(
                                state, A, float(beta), int(sample_size), include_skewness=False,
                                kernel=kernel, sampling_mode=sampling_mode,
                            ),
                            "moment3_tail": model.moment_tail_rhs(
                                state, A, float(beta), int(sample_size), order=3,
                                kernel=kernel, sampling_mode=sampling_mode,
                            ),
                            "moment5_tail": model.moment_tail_rhs(
                                state, A, float(beta), int(sample_size), order=5,
                                kernel=kernel, sampling_mode=sampling_mode,
                            ),
                        }
                        for name, prediction in predictions.items():
                            absolute = float(np.linalg.norm(prediction - exact))
                            symmetric = float(2.0 * absolute / max(np.linalg.norm(prediction) + exact_norm, 1e-12))
                            yield {
                                "schema_version": config["schema_version"],
                                "config_hash": config_hash,
                                "base_case": base_case,
                                "dimension": k,
                                "beta": beta,
                                "m": sample_size,
                                "kernel": kernel,
                                "sampling_mode": sampling_mode,
                                "chi": f"{chi:.17g}",
                                "psi": f"{psi:.17g}",
                                "exact_norm": f"{exact_norm:.17g}",
                                "model": name,
                                "absolute_error": f"{absolute:.17g}",
                                "symmetric_relative_error": f"{symmetric:.17g}",
                            }


def micro_macro_rows(config: dict, config_hash: str):
    settings = config["micro_macro"]
    seed_base = int(config["seed"]) + 1201
    for game_id, game in enumerate(settings["games"]):
        A = np.asarray(game["payoff"], dtype=float)
        x0 = np.asarray(game["initial_state"], dtype=float)
        for condition_id, condition in enumerate(settings["conditions"]):
            beta = float(condition["beta"])
            sample_size = int(condition["m"])
            kernel = condition["kernel"]
            sampling_mode = condition["sampling_mode"]
            exact_rhs = named_rhs("exact", A, beta, sample_size, kernel=kernel, sampling_mode=sampling_mode)
            times, reference = model.rk4_integrate(
                exact_rhs,
                x0,
                float(settings["horizon"]),
                float(settings["record_dt"]),
                internal_dt=float(settings["internal_dt"]),
            )
            for population in settings["populations"]:
                trajectories = []
                for seed_index in range(int(settings["seeds"])):
                    _, trajectory = model.agent_trajectory(
                        A,
                        x0,
                        int(population),
                        float(settings["horizon"]),
                        float(settings["record_dt"]),
                        beta,
                        sample_size,
                        seed_base + 100_000 * game_id + 10_000 * condition_id + 100 * int(population) + seed_index,
                        kernel=kernel,
                        sampling_mode=sampling_mode,
                        replacement=True,
                    )
                    trajectories.append(trajectory)
                stack = np.asarray(trajectories)
                mean = stack.mean(axis=0)
                deviations = stack - reference[None, :, :]
                integrated = np.trapezoid(np.abs(mean - reference).sum(axis=1), times) / float(times[-1])
                endpoint = float(np.abs(mean[-1] - reference[-1]).sum())
                fluctuation = float(np.sqrt(population) * np.mean(np.std(stack, axis=0, ddof=1)))
                individual_integrated = np.trapezoid(
                    np.abs(deviations).sum(axis=2), times, axis=1
                ) / float(times[-1])
                individual_rms_l2 = float(np.sqrt(np.mean(np.sum(deviations**2, axis=2))))
                endpoint_rms_l2 = float(np.sqrt(np.mean(np.sum(deviations[:, -1, :] ** 2, axis=1))))
                yield {
                    "schema_version": config["schema_version"],
                    "config_hash": config_hash,
                    "game": game["name"],
                    "condition": condition["name"],
                    "beta": beta,
                    "m": sample_size,
                    "kernel": kernel,
                    "sampling_mode": sampling_mode,
                    "population": population,
                    "seeds": settings["seeds"],
                    "integrated_l1": f"{float(integrated):.17g}",
                    "endpoint_l1": f"{endpoint:.17g}",
                    "sqrt_n_scaled_fluctuation": f"{fluctuation:.17g}",
                    "individual_integrated_l1_mean": f"{float(individual_integrated.mean()):.17g}",
                    "individual_integrated_l1_sem": f"{float(individual_integrated.std(ddof=1) / np.sqrt(len(individual_integrated))):.17g}",
                    "individual_rms_l2": f"{individual_rms_l2:.17g}",
                    "endpoint_rms_l2": f"{endpoint_rms_l2:.17g}",
                }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "nature_smoke.json")
    parser.add_argument(
        "--stages", nargs="+", choices=("e9", "e10", "e11", "e12"),
        default=("e9", "e10", "e12", "e11"),
        help="Run selected experiment stages; useful for refining one stage without recomputing all outputs.",
    )
    args = parser.parse_args()
    config, config_hash = load_config(args.config.resolve())
    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    profile = config["profile"]
    started = time.perf_counter()
    qualitative_fields = [
        "schema_version", "config_hash", "case_id", "ensemble", "game", "beta", "m", "kernel", "sampling_mode",
        "probe_state", "chi", "feedback_skewness", "generated_classical_stability", "exact_valid", "exact_equilibrium", "exact_stability", "exact_label",
    ]
    for name in MODEL_NAMES[:-1]:
        qualitative_fields.extend([
            f"{name}_valid", f"{name}_equilibrium_shift", f"{name}_stability", f"{name}_label",
            f"{name}_stability_mismatch", f"{name}_probe_relative_error",
        ])
    counts = {}
    stages = set(args.stages)
    if "e9" in stages:
        phase_started = time.perf_counter()
        print(f"[{profile}] E9 qualitative stability map started", flush=True)
        counts["e9_qualitative_map"] = atomic_csv(
            output / f"{profile}_e9_qualitative_map.csv", qualitative_fields, qualitative_rows(config, config_hash)
        )
        print(f"[{profile}] E9 completed: {counts['e9_qualitative_map']} rows in {time.perf_counter() - phase_started:.1f}s", flush=True)
    if "e10" in stages:
        phase_started = time.perf_counter()
        print(f"[{profile}] E10 non-equivalence baselines started", flush=True)
        counts["e10_non_equivalence"] = atomic_csv(
            output / f"{profile}_e10_non_equivalence.csv",
            ["schema_version", "config_hash", "case_id", "beta", "m", "kernel", "sampling_mode", "model", "train_relative_rmse", "test_relative_rmse", "fitted_payoff", "fitted_time_scale"],
            nonequivalence_rows(config, config_hash),
        )
        print(f"[{profile}] E10 completed: {counts['e10_non_equivalence']} rows in {time.perf_counter() - phase_started:.1f}s", flush=True)
    if "e12" in stages:
        phase_started = time.perf_counter()
        print(f"[{profile}] E12 paired accuracy map started", flush=True)
        counts["e12_paired_accuracy"] = atomic_csv(
            output / f"{profile}_e12_paired_accuracy.csv",
            ["schema_version", "config_hash", "base_case", "dimension", "beta", "m", "kernel", "sampling_mode", "chi", "psi", "exact_norm", "model", "absolute_error", "symmetric_relative_error"],
            paired_accuracy_rows(config, config_hash),
        )
        print(f"[{profile}] E12 completed: {counts['e12_paired_accuracy']} rows in {time.perf_counter() - phase_started:.1f}s", flush=True)
    if "e11" in stages:
        phase_started = time.perf_counter()
        print(f"[{profile}] E11 micro-macro closure started", flush=True)
        counts["e11_micro_macro"] = atomic_csv(
            output / f"{profile}_e11_micro_macro.csv",
            ["schema_version", "config_hash", "game", "condition", "beta", "m", "kernel", "sampling_mode", "population", "seeds", "integrated_l1", "endpoint_l1", "sqrt_n_scaled_fluctuation", "individual_integrated_l1_mean", "individual_integrated_l1_sem", "individual_rms_l2", "endpoint_rms_l2"],
            micro_macro_rows(config, config_hash),
        )
        print(f"[{profile}] E11 completed: {counts['e11_micro_macro']} rows in {time.perf_counter() - phase_started:.1f}s", flush=True)
    manifest = {
        "schema_version": config["schema_version"],
        "profile": profile,
        "config_hash": config_hash,
        "config_path": str(args.config.resolve()),
        "rows": counts,
        "elapsed_seconds": time.perf_counter() - started,
    }
    suffix = "" if stages == {"e9", "e10", "e11", "e12"} else "_" + "_".join(sorted(stages))
    path = output / f"{profile}_nature_manifest{suffix}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
