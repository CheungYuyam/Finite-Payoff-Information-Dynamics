"""Numerical regression checks for the analytic scalar remainder constants.

These grid checks do not constitute proofs.  They protect the code-level
normalization and Taylor coefficients used by the manuscript theorem draft.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))
import automatica_model as model  # noqa: E402


class KernelRemainderRegressionTests(unittest.TestCase):
    def test_simplex_prefactor_row_sum_used_by_c1_bound(self) -> None:
        """Check the chart-specific product-rule count in Lemma 3.

        This protects against confusing a bound for one partial derivative
        with the induced-infinity (row-sum) Jacobian norm.
        """
        rng = np.random.default_rng(20260904)
        for k in range(3, 8):
            for _ in range(200):
                x = rng.dirichlet(np.ones(k))
                r = int(rng.integers(k - 1))
                row_sum = 0.0
                for j in range(k):
                    if j == r:
                        continue
                    gradient = np.zeros(k - 1)
                    for q in range(k - 1):
                        dx_r = 1.0 if q == r else 0.0
                        dx_j = (-1.0 if j == k - 1 else (1.0 if q == j else 0.0))
                        gradient[q] = dx_r * x[j] + x[r] * dx_j
                    row_sum += float(np.linalg.norm(gradient, ord=1))
                self.assertLessEqual(row_sum, 2.0 * k - 3.0 + 1e-12)

    def test_global_taylor_remainder_bounds_on_deterministic_grid(self) -> None:
        constants = {
            "fermi": (1.0 / 12.0, 1.0 / 120.0, 17.0 / 20160.0),
            "arctan": (1.0 / 3.0, 1.0 / 5.0, 1.0 / 7.0),
        }
        z = np.linspace(-8.0, 8.0, 4001)
        for kernel, (d3, d5, d7) in constants.items():
            c3, c5 = model.KERNEL_COEFFICIENTS[kernel]
            for beta in (0.05, 0.4, 1.7):
                q = model.normalized_revision_flux(z, beta, kernel=kernel)
                first = z
                third = first - c3 * beta**2 * z**3
                fifth = third + c5 * beta**4 * z**5
                tolerance = 2e-11
                self.assertTrue(
                    np.all(np.abs(q - first) <= d3 * beta**2 * np.abs(z) ** 3 + tolerance)
                )
                self.assertTrue(
                    np.all(np.abs(q - third) <= d5 * beta**4 * np.abs(z) ** 5 + tolerance)
                )
                self.assertTrue(
                    np.all(np.abs(q - fifth) <= d7 * beta**6 * np.abs(z) ** 7 + tolerance)
                )

    def test_reduced_c1_bound_covers_finite_difference_jacobians(self) -> None:
        """Regression-check the manuscript's conservative C1 bound.

        This is an implementation check only. The analytic proof uses the
        multinomial-polynomial derivative bound stated in Lemma 3.
        """
        A = np.asarray(
            [[0.2, 1.1, -0.4], [0.7, -0.3, 0.8], [-0.6, 0.4, 0.9]],
            dtype=float,
        )
        y = np.asarray([0.27, 0.31])
        step = 1e-6

        for kernel, sampling_mode, m in (
            ("fermi", "independent", 2),
            ("fermi", "common", 2),
            ("arctan", "independent", (2, None, 3)),
        ):
            for order in (1, 3, 5):
                def residual(chart_y: np.ndarray) -> np.ndarray:
                    x = np.append(chart_y, 1.0 - chart_y.sum())
                    exact = model.exact_rhs(
                        x, A, 0.55, m, kernel=kernel, sampling_mode=sampling_mode
                    )
                    if order == 1:
                        approximation = model.classical_rhs(x, A)
                    else:
                        approximation = model.moment_tail_rhs(
                            x, A, 0.55, m, order=order,
                            kernel=kernel, sampling_mode=sampling_mode,
                        )
                    return (exact - approximation)[:-1]

                jacobian = np.column_stack(
                    [
                        (residual(y + step * np.eye(2)[column])
                         - residual(y - step * np.eye(2)[column])) / (2.0 * step)
                        for column in range(2)
                    ]
                )
                epsilon0, epsilon1 = model.reduced_tail_c1_bound(
                    A, 0.55, m, order=order, kernel=kernel,
                    sampling_mode=sampling_mode,
                )
                self.assertLessEqual(float(np.max(np.abs(residual(y)))), epsilon0 + 1e-9)
                self.assertLessEqual(
                    float(np.linalg.norm(jacobian, ord=np.inf)), epsilon1 + 1e-7
                )

    def test_infinite_feedback_degenerate_and_mixed_specs_are_consistent(self) -> None:
        """Check m=None as a point-mass feedback endpoint in every theorem path."""
        A = np.asarray(
            [[0.2, 1.1, -0.4], [0.7, -0.3, 0.8], [-0.6, 0.4, 0.9]],
            dtype=float,
        )
        x = np.asarray([0.27, 0.31, 0.42])
        beta = 0.35

        # Both ends deterministic: independent and common are the same field.
        exact_independent = model.exact_rhs(
            x, A, beta, None, kernel="fermi", sampling_mode="independent"
        )
        exact_common = model.exact_rhs(
            x, A, beta, None, kernel="fermi", sampling_mode="common"
        )
        np.testing.assert_allclose(exact_independent, exact_common, atol=1e-13)

        # One deterministic endpoint plus finite feedback is a valid independent
        # specification and must have a finite, nonnegative C1 certificate.
        epsilon0, epsilon1 = model.reduced_tail_c1_bound(
            A, beta, (2, None, 3), order=3,
            kernel="fermi", sampling_mode="independent",
        )
        self.assertGreaterEqual(epsilon0, 0.0)
        self.assertGreaterEqual(epsilon1, 0.0)

        # Common sampling requires the same sample size at both endpoints;
        # heterogeneous finite/infinite common feedback must be rejected.
        with self.assertRaises(ValueError):
            model.exact_rhs(
                x, A, beta, (2, None, 2),
                kernel="fermi", sampling_mode="common",
            )

    def test_finite_population_closure_bounds_cover_generator_drift(self) -> None:
        """Check the explicit O(1/N) closure bounds in Theorem 5."""
        A = np.asarray(
            [[0.2, 1.1, -0.4], [0.7, -0.3, 0.8], [-0.6, 0.4, 0.9]],
            dtype=float,
        )
        payoff_span = float(A.max() - A.min())
        for sampling_mode, m in (
            ("independent", 2),
            ("independent", (2, None, 3)),
            ("common", 2),
            ("common", None),
        ):
            for population in (17, 41, 97):
                counts = np.asarray(
                    [population // 3, population // 3,
                     population - 2 * (population // 3)],
                    dtype=int,
                )
                x = counts / population
                finite_drift = model.finite_population_generator_rhs(
                    counts, A, 0.55, m, kernel="fermi",
                    sampling_mode=sampling_mode,
                )
                mean_field = model.exact_rhs(
                    x, A, 0.55, m, kernel="fermi",
                    sampling_mode=sampling_mode,
                )
                error = float(np.linalg.norm(finite_drift - mean_field, ord=np.inf))
                if sampling_mode == "independent":
                    sample_sizes = model.normalize_sample_sizes(m, len(x))
                    ell_max = max(1 if value is None else value for value in sample_sizes)
                    bound = (payoff_span / (4.0 * (population - 1))
                             + population * payoff_span * ell_max / (population - 1) ** 2)
                else:
                    ell = 1 if m is None else m
                    bound = (payoff_span / (4.0 * (population - 1))
                             + population * payoff_span * ell
                             / ((population - 1) * (population - 2)))
                self.assertLessEqual(error, bound + 1e-12)

    def test_finite_feedback_curvature_constants_are_exact(self) -> None:
        """Check the kernel-specific suprema used by Proposition F."""
        beta = 0.73
        self.assertAlmostEqual(
            model.normalized_flux_second_derivative_sup(beta, kernel="fermi"),
            2.0 * beta / (3.0 * np.sqrt(3.0)),
        )
        self.assertAlmostEqual(
            model.normalized_flux_second_derivative_sup(beta, kernel="arctan"),
            9.0 * beta / (8.0 * np.sqrt(3.0)),
        )

    def test_finite_feedback_deviation_bound_covers_exact_difference(self) -> None:
        """The statewise Proposition-F bound covers both sampling mechanisms."""
        A = np.asarray(
            [[0.2, 1.1, -0.4], [0.7, -0.3, 0.8], [-0.6, 0.4, 0.9]],
            dtype=float,
        )
        x = np.asarray([0.27, 0.31, 0.42])
        for kernel, sampling_mode, m in (
            ("fermi", "independent", 2),
            ("fermi", "independent", (2, None, 3)),
            ("arctan", "common", 3),
        ):
            exact = model.exact_rhs(
                x, A, 0.55, m, kernel=kernel, sampling_mode=sampling_mode
            )
            full_feedback = model.exact_rhs(
                x, A, 0.55, None, kernel=kernel, sampling_mode=sampling_mode
            )
            bound = model.finite_feedback_rhs_deviation_bound(
                x, A, 0.55, m, kernel=kernel, sampling_mode=sampling_mode
            )
            self.assertTrue(np.all(np.abs(exact - full_feedback) <= bound + 1e-12))

        endpoint = model.finite_feedback_rhs_deviation_bound(
            x, A, 0.55, None, kernel="fermi", sampling_mode="independent"
        )
        np.testing.assert_allclose(endpoint, 0.0, atol=1e-15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
