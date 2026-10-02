"""Regression checks for the focused finite-feedback completion artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import unittest


RESULTS = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "focused_finite_sample_completion"
)


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class FocusedCompletionArtifactTests(unittest.TestCase):
    def test_audit_and_expected_row_counts_exist(self) -> None:
        with (RESULTS / "focused_completion_audit.json").open(
            "r", encoding="utf-8"
        ) as handle:
            audit = json.load(handle)
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["rows"], {
            "two_strategy": 8,
            "three_layer": 408,
            "boundaries": 2,
            "regimes": 10,
            "feedback_budget": 24,
        })

    def test_two_strategy_correction_converges_and_bounds_cover(self) -> None:
        rows = read_csv("two_strategy_finite_sample_correction.csv")
        shifts = [abs(float(row["exact_shift_from_full_feedback"])) for row in rows]
        m3_errors = [abs(float(row["moment3_root_error"])) for row in rows]
        m5_errors = [abs(float(row["moment5_root_error"])) for row in rows]
        self.assertTrue(all(a >= b for a, b in zip(shifts, shifts[1:])))
        self.assertTrue(all(a >= b for a, b in zip(m3_errors, m3_errors[1:])))
        self.assertTrue(all(a >= b for a, b in zip(m5_errors, m5_errors[1:])))
        self.assertTrue(all(
            float(row["finite_sample_rhs_difference_inf"])
            <= float(row["finite_sample_rhs_bound_inf"]) + 1e-12
            for row in rows
        ))
        self.assertAlmostEqual(float(rows[-1]["exact_shift_from_full_feedback"]), 0.0)

    def test_three_layer_bound_coverage_and_control_layer_identity(self) -> None:
        rows = read_csv("three_layer_counterfactual.csv")
        for layer in ("classical", "full_feedback_kernel", "finite_sample_exact_m2"):
            layer_rows = [row for row in rows if row["layer"] == layer]
            self.assertEqual(len(layer_rows), 136)
            self.assertTrue(all(float(row["residual_l2"]) < 1e-10 for row in layer_rows))
            self.assertTrue(all(float(row["jacobian_step_spread"]) < 1e-8 for row in layer_rows))
        classical = [row for row in rows if row["layer"] == "classical"]
        full = [row for row in rows if row["layer"] == "full_feedback_kernel"]
        self.assertTrue(all(
            float(row["equilibrium_distance_to_classical_l2"]) < 1e-12
            for row in full
        ))
        finite = [row for row in rows if row["layer"] == "finite_sample_exact_m2"]
        self.assertTrue(all(
            float(row["finite_sample_rhs_difference_inf"])
            <= float(row["finite_sample_rhs_bound_inf"]) + 1e-12
            for row in finite
        ))
        self.assertTrue(all(float(row["finite_sample_rhs_difference_inf"]) == 0.0 for row in classical))

    def test_two_critical_boundaries_are_local_stability_crossings(self) -> None:
        rows = read_csv("three_layer_critical_boundaries.csv")
        self.assertEqual(len(rows), 2)
        critical_betas = sorted(float(row["critical_beta"]) for row in rows)
        self.assertAlmostEqual(critical_betas[0], 0.7612251476825862, places=8)
        self.assertAlmostEqual(critical_betas[1], 0.8003041139399212, places=8)
        self.assertTrue(all(abs(float(row["root_max_real_eigenvalue"])) < 1e-8 for row in rows))

    def test_feedback_budget_is_monotone_and_covers_small_check(self) -> None:
        rows = read_csv("feedback_budget_certificate.csv")
        self.assertEqual(len(rows), 24)
        for game in ("two_strategy_example", "frozen_E14_game"):
            for beta in ("0.05", "0.1", "0.2", "0.4"):
                group = [
                    row for row in rows
                    if row["game"] == game and row["beta"] == beta
                ]
                group.sort(key=lambda row: float(row["absolute_flow_tolerance"]), reverse=True)
                sizes = [int(row["certified_sample_size"]) for row in group]
                self.assertTrue(all(a <= b for a, b in zip(sizes, sizes[1:])))
                self.assertTrue(all(row["verification_bound_covers"] == "True" for row in group))
                self.assertTrue(all(
                    float(row["certified_bound_inf"]) <= float(row["absolute_flow_tolerance"]) + 1e-12
                    for row in group
                ))
        # At fixed tolerance, increasing beta cannot reduce the conservative
        # certificate because the kernel curvature grows linearly in beta.
        for game in ("two_strategy_example", "frozen_E14_game"):
            for tolerance in ("0.01", "0.001", "0.0001"):
                group = [
                    row for row in rows
                    if row["game"] == game and row["absolute_flow_tolerance"] == tolerance
                ]
                group.sort(key=lambda row: float(row["beta"]))
                sizes = [int(row["certified_sample_size"]) for row in group]
                self.assertTrue(all(a <= b for a, b in zip(sizes, sizes[1:])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
