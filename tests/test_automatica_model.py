from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

import automatica_model as model


class DistributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.asarray([0.2, 0.5, 0.3])
        self.payoffs = np.asarray([-1.0, 0.4, 2.0])

    def test_distribution_is_normalized_and_has_exact_sample_moments(self) -> None:
        mean, variance, third = model.one_draw_moments(self.payoffs, self.x)
        for m in (1, 2, 5, 20):
            values, probabilities = model.sample_mean_distribution(
                self.payoffs, self.x, m
            )
            observed_mean = float(probabilities @ values)
            centered = values - observed_mean
            self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=13)
            self.assertAlmostEqual(observed_mean, mean, places=12)
            self.assertAlmostEqual(
                float(probabilities @ centered**2), variance / m, places=12
            )
            self.assertAlmostEqual(
                float(probabilities @ centered**3), third / (m**2), places=11
            )

    def test_full_feedback_is_a_point_mass(self) -> None:
        values, probabilities = model.sample_mean_distribution(
            self.payoffs, self.x, None
        )
        np.testing.assert_allclose(values, [self.payoffs @ self.x])
        np.testing.assert_allclose(probabilities, [1.0])

    def test_difference_moments_have_correct_signs(self) -> None:
        A = np.asarray([[0.0, -1.0, 2.0], [1.5, 0.0, -0.5], [-1.0, 1.0, 0.0]])
        delta, variance, third = model.payoff_difference_moments(
            A, self.x, 0, 1, 2, 5
        )
        mi, vi, ti = model.one_draw_moments(A[0], self.x)
        mj, vj, tj = model.one_draw_moments(A[1], self.x)
        self.assertAlmostEqual(delta, mi - mj)
        self.assertAlmostEqual(variance, vi / 2 + vj / 5)
        self.assertAlmostEqual(third, ti / 4 - tj / 25)

    def test_common_sample_difference_moments_match_exact_distribution(self) -> None:
        A = np.asarray([[0.0, -1.0, 2.0], [1.5, 0.0, -0.5], [-1.0, 1.0, 0.0]])
        for sample_size in (1, 3, 8):
            delta, variance, third = model.payoff_difference_moments(
                A,
                self.x,
                0,
                1,
                sample_size,
                sample_size,
                sampling_mode="common",
            )
            values, probabilities = model.sample_mean_distribution(
                A[0] - A[1], self.x, sample_size
            )
            mean = float(probabilities @ values)
            centered = values - mean
            self.assertAlmostEqual(delta, mean, places=12)
            self.assertAlmostEqual(
                variance, float(probabilities @ centered**2), places=12
            )
            self.assertAlmostEqual(
                third, float(probabilities @ centered**3), places=11
            )

    def test_closed_difference_raw_moments_match_enumeration_through_five(self) -> None:
        A = np.asarray([[0.0, -1.0, 2.0], [1.5, 0.0, -0.5], [-1.0, 1.0, 0.0]])
        for sample_size in (1, 2, 5):
            for sampling_mode in ("independent", "common"):
                closed = model.payoff_difference_raw_moments(
                    A,
                    self.x,
                    0,
                    1,
                    sample_size,
                    sample_size,
                    sampling_mode=sampling_mode,
                )
                if sampling_mode == "common":
                    values, probabilities = model.sample_mean_distribution(
                        A[0] - A[1], self.x, sample_size
                    )
                    enumerated = np.asarray(
                        [float(probabilities @ values**order) for order in range(1, 6)]
                    )
                else:
                    vi, pi = model.sample_mean_distribution(A[0], self.x, sample_size)
                    vj, pj = model.sample_mean_distribution(A[1], self.x, sample_size)
                    enumerated = np.asarray(
                        [model._difference_raw_moment(vi, pi, vj, pj, order) for order in range(1, 6)]
                    )
                np.testing.assert_allclose(closed, enumerated, rtol=2e-12, atol=2e-12)


class DynamicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.A = np.asarray(
            [[0.0, -1.0, 2.0], [2.0, 0.0, -1.0], [-1.0, 2.0, 0.0]]
        )
        self.x = np.asarray([0.2, 0.5, 0.3])

    def test_all_rhs_preserve_mass(self) -> None:
        right_hand_sides = [
            model.classical_rhs(self.x, self.A),
            model.tail_rhs(
                self.x, self.A, 0.4, 3, include_skewness=False
            ),
            model.tail_rhs(
                self.x, self.A, 0.4, 3, include_skewness=True
            ),
            model.exact_rhs(self.x, self.A, 0.4, 3),
            model.bounded_moment_tail_rhs(
                self.x, self.A, 0.4, 3, order=3
            ),
            model.bounded_moment_tail_rhs(
                self.x, self.A, 0.4, 3, order=5
            ),
        ]
        for rhs in right_hand_sides:
            self.assertLess(abs(float(rhs.sum())), 1e-13)

    def test_bounded_tail_matches_polynomial_in_weak_regime(self) -> None:
        for kernel in ("fermi", "arctan"):
            for sampling_mode in ("independent", "common"):
                for order in (3, 5):
                    ordinary = model.moment_tail_rhs(
                        self.x,
                        self.A,
                        0.01,
                        3,
                        order=order,
                        kernel=kernel,
                        sampling_mode=sampling_mode,
                    )
                    bounded = model.bounded_moment_tail_rhs(
                        self.x,
                        self.A,
                        0.01,
                        3,
                        order=order,
                        kernel=kernel,
                        sampling_mode=sampling_mode,
                    )
                    np.testing.assert_allclose(bounded, ordinary, atol=1e-13)

    def test_full_feedback_weak_selection_recovers_replicator(self) -> None:
        classical = model.classical_rhs(self.x, self.A)
        error_large = np.linalg.norm(
            model.exact_rhs(self.x, self.A, 0.2, None) - classical
        )
        error_small = np.linalg.norm(
            model.exact_rhs(self.x, self.A, 0.1, None) - classical
        )
        self.assertGreater(error_large / error_small, 3.8)
        self.assertLess(error_large / error_small, 4.2)

    def test_expected_error_orders(self) -> None:
        betas = np.asarray([0.2, 0.1, 0.05, 0.025])
        classical = model.classical_rhs(self.x, self.A)
        classical_errors = []
        moment_errors = []
        for beta in betas:
            exact = model.exact_rhs(self.x, self.A, float(beta), 3)
            tail = model.tail_rhs(
                self.x, self.A, float(beta), 3, include_skewness=True
            )
            classical_errors.append(float(np.linalg.norm(exact - classical)))
            moment_errors.append(float(np.linalg.norm(exact - tail)))
        classical_slope = float(
            np.polyfit(np.log(betas), np.log(classical_errors), 1)[0]
        )
        moment_slope = float(
            np.polyfit(np.log(betas), np.log(moment_errors), 1)[0]
        )
        self.assertTrue(1.95 < classical_slope < 2.05, classical_slope)
        self.assertTrue(3.90 < moment_slope < 4.10, moment_slope)

    def test_cubic_implementation_matches_full_third_moment_tail(self) -> None:
        for kernel in ("fermi", "arctan"):
            for sampling_mode in ("independent", "common"):
                legacy = model.tail_rhs(
                    self.x,
                    self.A,
                    0.3,
                    4,
                    include_skewness=True,
                    kernel=kernel,
                    sampling_mode=sampling_mode,
                )
                generic = model.moment_tail_rhs(
                    self.x,
                    self.A,
                    0.3,
                    4,
                    order=3,
                    kernel=kernel,
                    sampling_mode=sampling_mode,
                )
                np.testing.assert_allclose(legacy, generic, atol=2e-14)

    def test_fifth_order_tail_has_sixth_order_error(self) -> None:
        betas = np.asarray([0.2, 0.15, 0.1, 0.075, 0.05])
        for kernel in ("fermi", "arctan"):
            for sampling_mode in ("independent", "common"):
                errors = []
                for beta in betas:
                    exact = model.exact_rhs(
                        self.x,
                        self.A,
                        float(beta),
                        3,
                        kernel=kernel,
                        sampling_mode=sampling_mode,
                    )
                    fifth = model.moment_tail_rhs(
                        self.x,
                        self.A,
                        float(beta),
                        3,
                        order=5,
                        kernel=kernel,
                        sampling_mode=sampling_mode,
                    )
                    errors.append(float(np.linalg.norm(exact - fifth)))
                slope = float(np.polyfit(np.log(betas), np.log(errors), 1)[0])
                self.assertTrue(5.75 < slope < 6.10, (kernel, sampling_mode, slope))

    def test_structural_invariances(self) -> None:
        permutation = np.asarray([2, 0, 1])
        scale = 1.7
        shifted = self.A + 4.25
        for kernel in ("fermi", "arctan"):
            baseline = model.exact_rhs(self.x, self.A, 0.3, 3, kernel=kernel)
            permuted = model.exact_rhs(
                self.x[permutation],
                self.A[np.ix_(permutation, permutation)],
                0.3,
                3,
                kernel=kernel,
            )
            np.testing.assert_allclose(permuted, baseline[permutation], atol=2e-14)
            np.testing.assert_allclose(
                model.exact_rhs(self.x, shifted, 0.3, 3, kernel=kernel),
                baseline,
                atol=2e-13,
            )
            np.testing.assert_allclose(
                model.exact_rhs(self.x, scale * self.A, 0.3, 3, kernel=kernel),
                scale * model.exact_rhs(
                    self.x, self.A, scale * 0.3, 3, kernel=kernel
                ),
                atol=3e-13,
            )

    def test_boundary_faces_are_invariant(self) -> None:
        boundary = np.asarray([0.0, 0.7, 0.3])
        for rhs in (
            model.classical_rhs(boundary, self.A),
            model.exact_rhs(boundary, self.A, 0.5, 2),
            model.moment_tail_rhs(boundary, self.A, 0.5, 2, order=5),
        ):
            self.assertAlmostEqual(float(rhs[0]), 0.0, places=14)
            self.assertAlmostEqual(float(rhs.sum()), 0.0, places=14)

    def test_rk4_and_agent_trajectories_stay_on_simplex(self) -> None:
        _, ode = model.rk4_integrate(
            lambda state: model.exact_rhs(state, self.A, 0.3, 2),
            self.x,
            horizon=0.1,
            record_dt=0.05,
            internal_dt=0.005,
        )
        _, agents = model.agent_trajectory(
            self.A,
            self.x,
            population=60,
            horizon=0.1,
            record_dt=0.05,
            beta=0.3,
            m=2,
            seed=7,
        )
        for trajectory in (ode, agents):
            np.testing.assert_allclose(trajectory.sum(axis=1), 1.0, atol=1e-12)
            self.assertGreaterEqual(float(trajectory.min()), 0.0)

    def test_all_agent_sampling_variants_stay_on_simplex(self) -> None:
        for kernel in ("fermi", "arctan"):
            for sampling_mode in ("independent", "common"):
                for replacement in (True, False):
                    _, trajectory = model.agent_trajectory(
                        self.A,
                        self.x,
                        population=60,
                        horizon=0.05,
                        record_dt=0.025,
                        beta=0.4,
                        m=3,
                        seed=17,
                        kernel=kernel,
                        sampling_mode=sampling_mode,
                        replacement=replacement,
                    )
                    np.testing.assert_allclose(
                        trajectory.sum(axis=1), 1.0, atol=1e-12
                    )
                    self.assertGreaterEqual(float(trajectory.min()), 0.0)


class ValidationTests(unittest.TestCase):
    def test_fermi_is_stable_and_complementary(self) -> None:
        z = np.asarray([-1000.0, -2.0, 0.0, 2.0, 1000.0])
        values = model.fermi(z, 3.0)
        self.assertTrue(np.all(np.isfinite(values)))
        np.testing.assert_allclose(values + model.fermi(-z, 3.0), 1.0)

    def test_initial_counts_conserve_population(self) -> None:
        counts = model.initial_counts(np.asarray([0.333, 0.333, 0.334]), 101)
        self.assertEqual(int(counts.sum()), 101)
        self.assertTrue(np.all(counts >= 0))

    def test_feedback_budget_inverts_the_statewise_bound(self) -> None:
        state = np.asarray([0.27, 0.41, 0.32])
        payoff = np.asarray(
            [[0.0, -0.4, 1.2], [0.7, 0.0, -0.8], [-0.6, 1.1, 0.0]]
        )
        certificates = [
            model.finite_feedback_sample_budget(state, payoff, 0.4, tolerance)
            for tolerance in (1e-2, 1e-3, 1e-4)
        ]
        sizes = [int(certificate["sample_size"]) for certificate in certificates]
        self.assertTrue(all(size >= 1 for size in sizes))
        self.assertTrue(all(a <= b for a, b in zip(sizes, sizes[1:])))
        for certificate in certificates:
            m = int(certificate["sample_size"])
            direct_bound = float(np.linalg.norm(
                model.finite_feedback_rhs_deviation_bound(state, payoff, 0.4, m),
                ord=np.inf,
            ))
            self.assertLessEqual(
                direct_bound, float(certificate["tolerance"]) + 1e-14
            )
            self.assertAlmostEqual(
                direct_bound, float(certificate["certified_bound_inf"]), places=14
            )

    def test_finite_population_generator_converges_to_exact_mean_field(self) -> None:
        state = np.array([0.27, 0.41, 0.32])
        payoff = np.array(
            [[0.0, -0.4, 1.2], [0.7, 0.0, -0.8], [-0.6, 1.1, 0.0]]
        )
        exact = model.exact_rhs(state, payoff, 0.5, 2)
        counts = model.initial_counts(state, 100000)
        finite = model.finite_population_generator_rhs(counts, payoff, 0.5, 2)
        np.testing.assert_allclose(finite, exact, atol=3e-5, rtol=3e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
