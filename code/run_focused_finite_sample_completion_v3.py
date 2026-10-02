"""Focused completion checks for the finite-sample replicator narrative.

This script deliberately adds no new application setting and no random-game
ensemble.  It performs four small, fixed checks needed by the paper's narrow
claim: (i) a two-strategy explanatory example, (ii) the finite-feedback
curvature certificate, (iii) a same-game three-layer counterfactual
``Classical / full-feedback kernel / finite-sample Exact``, and (iv) an
illustrative approximation-regime table based on the proved C0 remainder
bounds.  All inputs are frozen in this file or in ``configs/value_full.json``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import brentq

import automatica_model as model
from run_extended_v3 import (
    classical_interior_equilibrium,
    interior_equilibrium,
    simplex_jacobian,
)


ROOT = Path(__file__).resolve().parents[1]
JACOBIAN_STEPS = (3e-7, 1e-6, 3e-6)


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def exact_rhs(
    A: np.ndarray, beta: float, sample_size: int | None
) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: model.exact_rhs(
        x, A, beta, sample_size, kernel="fermi", sampling_mode="independent"
    )


def stability_record(
    rhs: Callable[[np.ndarray], np.ndarray],
    equilibrium: np.ndarray,
) -> tuple[float, float, float]:
    eigenvalues = [
        float(np.max(np.real(np.linalg.eigvals(simplex_jacobian(rhs, equilibrium, h=h)))))
        for h in JACOBIAN_STEPS
    ]
    return (
        float(np.linalg.norm(rhs(equilibrium))),
        float(np.median(eigenvalues)),
        float(np.ptp(eigenvalues)),
    )


def resolve_three_strategy(
    rhs: Callable[[np.ndarray], np.ndarray],
    reference: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    equilibrium, valid, residual = interior_equilibrium(rhs, reference)
    if not valid:
        raise RuntimeError(f"no interior equilibrium; residual={residual}")
    checked_residual, spectral_abscissa, step_spread = stability_record(rhs, equilibrium)
    return equilibrium, checked_residual, spectral_abscissa, step_spread


def two_strategy_root(rhs: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    def scalar(q: float) -> float:
        return float(rhs(np.asarray([q, 1.0 - q]))[0])

    grid = np.linspace(1e-8, 1.0 - 1e-8, 1001)
    values = [scalar(float(q)) for q in grid]
    brackets = [
        (float(left), float(right))
        for left, right, f_left, f_right in zip(
            grid[:-1], grid[1:], values[:-1], values[1:]
        )
        if f_left * f_right < 0.0
    ]
    if len(brackets) != 1:
        raise RuntimeError(f"expected one interior equilibrium, found {len(brackets)}")
    root = brentq(scalar, *brackets[0], xtol=1e-13, rtol=1e-13)
    return np.asarray([root, 1.0 - root])


def two_strategy_derivative(rhs: Callable[[np.ndarray], np.ndarray], q: float) -> float:
    h = 1e-6
    plus = rhs(np.asarray([q + h, 1.0 - q - h]))[0]
    minus = rhs(np.asarray([q - h, 1.0 - q + h]))[0]
    return float((plus - minus) / (2.0 * h))


def two_strategy_rows() -> tuple[list[dict], dict]:
    # This asymmetric anti-coordination game has the Classical interior point
    # x_1=3/5.  Its finite-sample shift is nonzero, yet the example remains
    # one-dimensional and easy to read.
    A = np.asarray([[0.0, 3.0], [2.0, 0.0]])
    beta = 1.0
    sample_sizes: list[int | None] = [1, 2, 3, 5, 10, 20, 50, None]
    classical = lambda x: model.classical_rhs(x, A)
    q_classical = float(two_strategy_root(classical)[0])
    q_full = float(two_strategy_root(exact_rhs(A, beta, None))[0])
    rows: list[dict] = []
    for sample_size in sample_sizes:
        exact = exact_rhs(A, beta, sample_size)
        moment3 = lambda x, m=sample_size: model.moment_tail_rhs(
            x, A, beta, m, order=3, kernel="fermi", sampling_mode="independent"
        )
        moment5 = lambda x, m=sample_size: model.moment_tail_rhs(
            x, A, beta, m, order=5, kernel="fermi", sampling_mode="independent"
        )
        x_exact = two_strategy_root(exact)
        x_m3 = two_strategy_root(moment3)
        x_m5 = two_strategy_root(moment5)
        correction = exact(x_exact) - exact_rhs(A, beta, None)(x_exact)
        correction_bound = model.finite_feedback_rhs_deviation_bound(
            x_exact, A, beta, sample_size, kernel="fermi", sampling_mode="independent"
        )
        rows.append({
            "example": "two_strategy_asymmetric_anti_coordination",
            "payoff": "[[0,3],[2,0]]",
            "beta": beta,
            "m": "inf" if sample_size is None else sample_size,
            "inverse_m": 0.0 if sample_size is None else 1.0 / float(sample_size),
            "classical_equilibrium_x1": q_classical,
            "full_feedback_equilibrium_x1": q_full,
            "exact_equilibrium_x1": float(x_exact[0]),
            "moment3_equilibrium_x1": float(x_m3[0]),
            "moment5_equilibrium_x1": float(x_m5[0]),
            "exact_shift_from_full_feedback": float(x_exact[0] - q_full),
            "moment3_root_error": float(abs(x_m3[0] - x_exact[0])),
            "moment5_root_error": float(abs(x_m5[0] - x_exact[0])),
            "exact_local_derivative": two_strategy_derivative(exact, float(x_exact[0])),
            "finite_sample_rhs_difference_inf": float(np.linalg.norm(correction, ord=np.inf)),
            "finite_sample_rhs_bound_inf": float(np.linalg.norm(correction_bound, ord=np.inf)),
        })
    return rows, {"A": A.tolist(), "beta": beta, "classical_x1": q_classical, "full_feedback_x1": q_full}


def three_layer_rows() -> tuple[list[dict], list[dict], dict]:
    config = json.loads((ROOT / "configs" / "value_full.json").read_text(encoding="utf-8"))
    A = np.asarray(config["payoff"], dtype=float)
    classical_eq = classical_interior_equilibrium(A)
    betas = np.round(np.arange(0.05, 1.4001, 0.01), 12)
    layers = [
        ("classical", None),
        ("full_feedback_kernel", None),
        ("finite_sample_exact_m2", 2),
    ]
    references = {name: classical_eq.copy() for name, _ in layers}
    rows: list[dict] = []
    points: dict[str, list[tuple[float, float, np.ndarray]]] = {name: [] for name, _ in layers}
    for beta in betas:
        for name, sample_size in layers:
            if name == "classical":
                rhs = lambda x, game=A: model.classical_rhs(x, game)
                equilibrium = classical_eq.copy()
                residual, spectral_abscissa, step_spread = stability_record(rhs, equilibrium)
                sample_bound = 0.0
                sample_gap = 0.0
            else:
                rhs = exact_rhs(A, float(beta), sample_size)
                equilibrium, residual, spectral_abscissa, step_spread = resolve_three_strategy(
                    rhs, references[name]
                )
                references[name] = equilibrium
                if sample_size is None:
                    sample_bound = 0.0
                    sample_gap = 0.0
                else:
                    full_rhs = exact_rhs(A, float(beta), None)
                    sample_gap = float(np.linalg.norm(rhs(equilibrium) - full_rhs(equilibrium), ord=np.inf))
                    sample_bound = float(np.linalg.norm(
                        model.finite_feedback_rhs_deviation_bound(
                            equilibrium, A, float(beta), sample_size,
                            kernel="fermi", sampling_mode="independent",
                        ), ord=np.inf,
                    ))
            points[name].append((float(beta), spectral_abscissa, equilibrium.copy()))
            rows.append({
                "example": "frozen_E14_three_strategy_game",
                "layer": name,
                "beta": float(beta),
                "m": "not_applicable" if name == "classical" else ("inf" if sample_size is None else sample_size),
                "equilibrium": json.dumps(equilibrium.tolist(), separators=(",", ":")),
                "equilibrium_distance_to_classical_l2": float(np.linalg.norm(equilibrium - classical_eq)),
                "residual_l2": residual,
                "max_real_eigenvalue": spectral_abscissa,
                "jacobian_step_spread": step_spread,
                "finite_sample_rhs_difference_inf": sample_gap,
                "finite_sample_rhs_bound_inf": sample_bound,
            })

    boundary_rows: list[dict] = []
    for name, sample_size in layers:
        sequence = points[name]
        for (left_beta, left_value, left_state), (right_beta, right_value, right_state) in zip(sequence[:-1], sequence[1:]):
            if left_value == 0.0 or right_value == 0.0 or left_value * right_value > 0.0:
                continue

            def objective(beta_value: float) -> float:
                weight = (beta_value - left_beta) / (right_beta - left_beta)
                guess = (1.0 - weight) * left_state + weight * right_state
                rhs = exact_rhs(A, beta_value, sample_size)
                _, _, spectral, _ = resolve_three_strategy(rhs, guess)
                return spectral

            critical_beta = float(brentq(objective, left_beta, right_beta, xtol=1e-10, rtol=1e-10))
            weight = (critical_beta - left_beta) / (right_beta - left_beta)
            critical_rhs = exact_rhs(A, critical_beta, sample_size)
            critical_eq, residual, spectral, step_spread = resolve_three_strategy(
                critical_rhs, (1.0 - weight) * left_state + weight * right_state
            )
            boundary_rows.append({
                "example": "frozen_E14_three_strategy_game",
                "layer": name,
                "m": "not_applicable" if name == "classical" else ("inf" if sample_size is None else sample_size),
                "left_beta": left_beta,
                "right_beta": right_beta,
                "critical_beta": critical_beta,
                "crossing_direction": "stable_to_unstable" if left_value < right_value else "unstable_to_stable",
                "critical_equilibrium": json.dumps(critical_eq.tolist(), separators=(",", ":")),
                "residual_l2": residual,
                "root_max_real_eigenvalue": spectral,
                "jacobian_step_spread": step_spread,
            })
    return rows, boundary_rows, {"A": A.tolist(), "classical_equilibrium": classical_eq.tolist()}


def regime_rows(two_strategy_game: np.ndarray, three_strategy_game: np.ndarray) -> list[dict]:
    # A tolerance is an author-specified absolute flow budget, not an error-fit
    # threshold.  The table makes the proved bound inspectable; the manuscript
    # need not endorse this numerical value as universal.
    tolerance = 1e-3
    rows: list[dict] = []
    for name, A in (("two_strategy_example", two_strategy_game), ("frozen_E14_game", three_strategy_game)):
        for beta in (0.05, 0.10, 0.20, 0.40, 0.80):
            bounds = {
                order: model.reduced_tail_c1_bound(
                    A, beta, 2, order=order, kernel="fermi", sampling_mode="independent"
                )[0]
                for order in (1, 3, 5)
            }
            if bounds[1] <= tolerance:
                recommended = "Classical"
            elif bounds[3] <= tolerance:
                recommended = "Moment3"
            elif bounds[5] <= tolerance:
                recommended = "Moment5"
            else:
                recommended = "Exact"
            rows.append({
                "game": name,
                "beta": beta,
                "absolute_flow_tolerance": tolerance,
                "classical_C0_bound": bounds[1],
                "moment3_C0_bound": bounds[3],
                "moment5_C0_bound": bounds[5],
                "illustrative_recommendation": recommended,
            })
    return rows


def feedback_budget_rows(two_strategy_game: np.ndarray, three_strategy_game: np.ndarray) -> list[dict]:
    """Build the small operational certificate table for the two frozen games."""
    two_state = np.asarray([0.6, 0.4], dtype=float)
    three_state = classical_interior_equilibrium(three_strategy_game)
    rows: list[dict] = []
    for name, A, x in (
        ("two_strategy_example", two_strategy_game, two_state),
        ("frozen_E14_game", three_strategy_game, three_state),
    ):
        for beta in (0.05, 0.10, 0.20, 0.40):
            for tolerance in (1e-2, 1e-3, 1e-4):
                certificate = model.finite_feedback_sample_budget(
                    x, A, beta, tolerance,
                    kernel="fermi", sampling_mode="independent",
                )
                m = int(certificate["sample_size"])
                # Keep an exact check at the smallest nontrivial sample size;
                # large certified m values need not be enumerated again here.
                verification_m = min(m, 2)
                exact = exact_rhs(A, beta, verification_m)
                full = exact_rhs(A, beta, None)
                actual_gap = float(np.linalg.norm(exact(x) - full(x), ord=np.inf))
                check_bound = float(np.linalg.norm(
                    model.finite_feedback_rhs_deviation_bound(
                        x, A, beta, verification_m,
                        kernel="fermi", sampling_mode="independent",
                    ),
                    ord=np.inf,
                ))
                rows.append({
                    "game": name,
                    "reference_state": json.dumps(x.tolist(), separators=(",", ":")),
                    "beta": beta,
                    "absolute_flow_tolerance": tolerance,
                    "certified_sample_size": m,
                    "certified_bound_inf": float(certificate["certified_bound_inf"]),
                    "bound_numerator_m1_inf": float(certificate["bound_numerator_m1_inf"]),
                    "verification_m": verification_m,
                    "verification_actual_gap_inf": actual_gap,
                    "verification_bound_inf": check_bound,
                    "verification_bound_covers": actual_gap <= check_bound + 1e-12,
                    "interpretation": "state-dependent instantaneous field certificate",
                })
    return rows


def main() -> None:
    results = ROOT / "results" / "focused_finite_sample_completion"
    results.mkdir(parents=True, exist_ok=True)
    two_rows, two_meta = two_strategy_rows()
    layers, boundaries, three_meta = three_layer_rows()
    regimes = regime_rows(np.asarray(two_meta["A"]), np.asarray(three_meta["A"]))
    budgets = feedback_budget_rows(np.asarray(two_meta["A"]), np.asarray(three_meta["A"]))
    write_csv(results / "two_strategy_finite_sample_correction.csv", two_rows)
    write_csv(results / "three_layer_counterfactual.csv", layers)
    if boundaries:
        write_csv(results / "three_layer_critical_boundaries.csv", boundaries)
    write_csv(results / "approximation_regime_certificate.csv", regimes)
    write_csv(results / "feedback_budget_certificate.csv", budgets)
    audit = {
        "status": "passed",
        "scope": "fixed two-strategy explanation and frozen E14 three-layer counterfactual only",
        "two_strategy": two_meta,
        "three_layer": three_meta,
        "rows": {"two_strategy": len(two_rows), "three_layer": len(layers), "boundaries": len(boundaries), "regimes": len(regimes), "feedback_budget": len(budgets)},
        "important_interpretation": [
            "The full-feedback bounded-kernel layer is a necessary control: it holds beta fixed while removing only sampling noise.",
            "A crossing in the finite-sample layer that is absent from the full-feedback layer is evidence for the finite-sample mechanism in this frozen example, not a universal statement over games.",
            "The approximation table is a certificate based on an author-selected absolute tolerance; it is not an error-fitted universal deployment rule.",
            "The feedback-budget table inverses the proved finite-sample field bound at two pre-existing reference states; it is state-dependent and instantaneous, not an optimal or global sampling policy.",
        ],
    }
    (results / "focused_completion_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
