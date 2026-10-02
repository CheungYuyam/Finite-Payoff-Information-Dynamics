"""Instantiate Theorems 3--4 in one frozen weak-feedback Fermi case.

This deliberately does not try to certify the E14 stability crossings: their
purpose is to show a non-perturbative qualitative effect, whereas a global
Taylor remainder certificate is expected to be conservative there.  Instead,
the script certifies the theorem's intended weak-feedback regime without
adding any simulation scale or stochastic runs.
"""

from __future__ import annotations

import argparse
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


def classical_reduced_jacobian_lipschitz_bound(
    payoff_span: float, dimension: int
) -> float:
    """A global infinity-norm Lipschitz bound for the classical Jacobian.

    In the chart ``x=(y,1-1^T y)``, shifting the payoff matrix by a constant
    permits entries in ``[0, M]``.  Differentiating the quadratic classical
    vector field twice gives an entrywise Hessian bound ``10 M``.  Summing a
    Hessian row and then a Jacobian row yields ``10 M d^2`` for ``d=K-1``.
    This is conservative by design and is the constant used in Theorem 3.
    """
    return float(10.0 * payoff_span * dimension**2)


def certificate() -> dict:
    config = json.loads((ROOT / "configs" / "value_full.json").read_text(encoding="utf-8"))
    A = np.asarray(config["payoff"], dtype=float)
    beta = 2.0e-4
    sample_size = 2
    kernel = "fermi"
    sampling_mode = "independent"
    order = 1
    radius_inf = 1.0e-5
    jacobian_steps = (3.0e-7, 1.0e-6, 3.0e-6)

    classical_state = classical_interior_equilibrium(A)
    classical_rhs = named_rhs("classical", A, beta, sample_size)
    classical_jacobians = [
        simplex_jacobian(classical_rhs, classical_state, h=step)
        for step in jacobian_steps
    ]
    J_classical = classical_jacobians[1]
    eigenvalues, eigenvectors = np.linalg.eig(J_classical)
    stability_margin = float(-np.max(np.real(eigenvalues)))
    inverse_norm_inf = float(np.linalg.norm(np.linalg.inv(J_classical), ord=np.inf))
    eigenvector_condition_2 = float(np.linalg.cond(eigenvectors, p=2))
    jacobian_step_spread = float(max(
        np.max(np.abs(jacobian - J_classical)) for jacobian in classical_jacobians
    ))

    epsilon0, epsilon1 = model.reduced_tail_c1_bound(
        A, beta, sample_size, order=order, kernel=kernel,
        sampling_mode=sampling_mode,
    )
    dimension = A.shape[0] - 1
    payoff_span = float(A.max() - A.min())
    L_classical_inf = classical_reduced_jacobian_lipschitz_bound(
        payoff_span, dimension
    )
    eta = inverse_norm_inf * (L_classical_inf * radius_inf + epsilon1)
    contraction_rhs = (1.0 - eta) * radius_inf
    contraction_lhs = inverse_norm_inf * epsilon0
    displacement_bound_inf = contraction_lhs / (1.0 - eta)

    # Conversion from the explicit infinity-norm bounds in Lemma 3 to the
    # Euclidean norm required by the spectral Bauer--Fike statement.
    L_classical_2 = np.sqrt(dimension) * L_classical_inf
    delta_jacobian_2 = (
        L_classical_2 * np.sqrt(dimension) * displacement_bound_inf
        + np.sqrt(dimension) * epsilon1
    )
    spectral_rhs = eigenvector_condition_2 * delta_jacobian_2

    exact_rhs = named_rhs(
        "exact", A, beta, sample_size, kernel=kernel,
        sampling_mode=sampling_mode,
    )
    exact_state, exact_valid, exact_residual = interior_equilibrium(
        exact_rhs, classical_state
    )
    exact_jacobians = [
        simplex_jacobian(exact_rhs, exact_state, h=step)
        for step in jacobian_steps
    ]
    exact_eigenvalues = np.linalg.eigvals(exact_jacobians[1])
    exact_jacobian_step_spread = float(max(
        np.max(np.abs(jacobian - exact_jacobians[1])) for jacobian in exact_jacobians
    ))
    actual_distance_inf = float(np.linalg.norm(
        exact_state[:-1] - classical_state[:-1], ord=np.inf
    ))

    theorem3_passed = bool(
        eta < 1.0 and contraction_lhs <= contraction_rhs
        and displacement_bound_inf <= radius_inf
    )
    theorem4_spectral_passed = bool(
        stability_margin > 0.0 and spectral_rhs < stability_margin
    )
    return {
        "schema_version": "v3-weak-feedback-certificate-2",
        "status": "passed" if theorem3_passed and theorem4_spectral_passed else "failed",
        "scope": (
            "Theorems 3--4 instantiated for the unbounded Classical truncation "
            "in one frozen weak-feedback Fermi independent-sampling case, using "
            "the corrected standard-simplex Jacobian row-sum bound in Lemma 3."
        ),
        "limitations": [
            "This is a sufficient-condition certificate, not a claim that the bound is tight.",
            "It does not certify the non-perturbative E14 stability crossings.",
            "It does not apply to bounded_moment_tail_rhs because clipping is nonsmooth.",
        ],
        "case": {
            "payoff": A.tolist(),
            "beta": beta,
            "m": sample_size,
            "kernel": kernel,
            "sampling_mode": sampling_mode,
            "comparison_baseline": "classical replicator (order 1)",
            "radius_inf": radius_inf,
        },
        "theorem3": {
            "epsilon0_inf": epsilon0,
            "epsilon1_inf": epsilon1,
            "L_classical_inf": L_classical_inf,
            "inverse_J_inf": inverse_norm_inf,
            "eta": eta,
            "contraction_lhs": contraction_lhs,
            "contraction_rhs": contraction_rhs,
            "displacement_bound_inf": displacement_bound_inf,
            "passed": theorem3_passed,
        },
        "theorem4_spectral": {
            "classical_eigenvalues": [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in eigenvalues
            ],
            "classical_stability_margin": stability_margin,
            "eigenvector_condition_2": eigenvector_condition_2,
            "L_classical_2": float(L_classical_2),
            "delta_jacobian_2": float(delta_jacobian_2),
            "bauer_fike_rhs": spectral_rhs,
            "passed": theorem4_spectral_passed,
        },
        "numerical_consistency_only": {
            "classical_state": classical_state.tolist(),
            "exact_state": exact_state.tolist(),
            "exact_valid": bool(exact_valid),
            "exact_residual_l2": float(exact_residual),
            "actual_reduced_distance_inf": actual_distance_inf,
            "exact_eigenvalues": [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in exact_eigenvalues
            ],
            "classical_jacobian_step_spread": jacobian_step_spread,
            "exact_jacobian_step_spread": exact_jacobian_step_spread,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "weak_feedback_stability_certificate_v3.json",
    )
    args = parser.parse_args()
    result = certificate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
