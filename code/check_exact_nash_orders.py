"""Exact-field Nash-order check for the existing Figure 2 control game.

No moment truncation is used. Enumerate independent binomial payoff counts,
differentiate their probabilities analytically, and check against exact_rhs
and centered finite differences. Existing Figure 2 data and figures are kept.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

import automatica_model as model


ROOT = Path(__file__).resolve().parents[1]
A = np.array([[0.0, 3.0], [2.0, 0.0]])
BETA = 1.0
NASH = 0.6
J_STAR = -1.2
SIZES = (32, 64, 128, 256, 512)
FIT_SIZES = (64, 128, 256, 512)


def evaluate(q: float, m: int, response: np.ndarray) -> tuple[float, float]:
    k = np.arange(m + 1, dtype=float)
    log_coefficients = np.array([math.lgamma(m+1)-math.lgamma(count+1)
                                -math.lgamma(m-count+1) for count in k])
    probabilities = np.exp(log_coefficients + k*np.log(q) + (m-k)*np.log1p(-q))
    probabilities /= probabilities.sum()
    # The two strategies use independent counts, both drawn from Cat(x).
    weights = probabilities[:, None] * probabilities[None, :]
    score = k / q - (m - k) / (1.0 - q)
    score -= float(probabilities @ score)
    mean_response = float(np.sum(weights * response))
    derivative_response = float(np.sum(
        weights * response * (score[:, None] + score[None, :])
    ))
    field = q * (1.0 - q) * mean_response
    derivative = ((1.0 - 2.0 * q) * mean_response
                  + q * (1.0 - q) * derivative_response)
    return field, derivative


def fit(rows: list[dict], key: str) -> dict:
    selected = [row for row in rows if row["m"] in FIT_SIZES]
    log_m = np.log([row["m"] for row in selected])
    log_error = np.log([abs(row[key]) for row in selected])
    slope, intercept = np.polyfit(log_m, log_error, 1)
    residual = log_error - (slope * log_m + intercept)
    r_squared = 1.0 - np.sum(residual**2) / np.sum((log_error-log_error.mean())**2)
    return {"slope": float(slope), "r_squared": float(r_squared)}


def main() -> None:
    rows = []
    for m in SIZES:
        k = np.arange(m + 1, dtype=float)
        differences = 3.0 - 3.0 * k[:, None] / m - 2.0 * k[None, :] / m
        response = model.normalized_revision_flux(differences, BETA, kernel="fermi")
        left, right = 0.55, 0.65
        left_value = evaluate(left, m, response)[0]
        assert left_value * evaluate(right, m, response)[0] < 0
        while right-left > 5e-15:
            middle = (left+right)/2
            middle_value = evaluate(middle, m, response)[0]
            if left_value*middle_value <= 0:
                right = middle
            else:
                left, left_value = middle, middle_value
        root = (left+right)/2
        residual, jacobian = evaluate(root, m, response)
        finite_differences = []
        for h in (1e-5, 3e-6, 1e-6):
            plus = evaluate(root+h, m, response)[0]
            minus = evaluate(root-h, m, response)[0]
            finite_differences.append((plus-minus)/(2*h))
        # Independent existing implementation check at the root and off it.
        implementation_gap = max(
            abs(evaluate(q, m, response)[0] - model.exact_rhs(
                np.array([q, 1-q]), A, BETA, m,
                kernel="fermi", sampling_mode="independent")[0])
            for q in (root, 0.37)
        )
        fd_gap = max(abs(value-jacobian) for value in finite_differences)
        assert abs(residual) < 1e-13
        assert implementation_gap < 1e-12
        assert fd_gap < 1e-8
        shift = root - NASH
        jacobian_gap = jacobian - J_STAR
        rows.append({
            "m": m, "exact_stationary_x1": root, "exact_shift": shift,
            "exact_jacobian": jacobian, "jacobian_difference": jacobian_gap,
            "m_squared_shift": m*m*shift, "m_jacobian_difference": m*jacobian_gap,
            "field_residual": abs(residual),
            "existing_implementation_difference": implementation_gap,
            "analytic_vs_finite_difference_max": fd_gap,
        })
    summary = {
        "game": A.tolist(), "beta": BETA, "kernel": "fermi",
        "sampling": "independent", "sample_sizes": SIZES, "fit_sizes": FIT_SIZES,
        "nash_x1": NASH, "reference_jacobian": J_STAR,
        "predicted_shift_coefficient": -0.028,
        "predicted_jacobian_coefficient": 0.936,
        "displacement_fit": fit(rows, "exact_shift"),
        "jacobian_fit": fit(rows, "jacobian_difference"),
        "root_method": "bisection", "root_bracket": [0.55, 0.65],
        "root_bracket_width_tolerance": 5e-15,
        "jacobian_method": "analytic differentiation of binomial probabilities",
        "finite_difference_steps": [1e-5, 3e-6, 1e-6],
        "max_field_residual": max(row["field_residual"] for row in rows),
        "max_implementation_difference": max(row["existing_implementation_difference"] for row in rows),
        "max_finite_difference_discrepancy": max(row["analytic_vs_finite_difference_max"] for row in rows),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), Path(model.__file__))},
    }
    output = ROOT / "results" / "exact_nash_orders"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "exact_nash_orders.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
