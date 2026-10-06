# Release verification

Verified locally on 2026-10-02 with Python 3.13.0, NumPy 2.4.4, SciPy 1.17.1, pandas 3.0.2 and matplotlib 3.10.8.

- All 31 tests passed in `python -m unittest discover -s tests -p "test_*.py" -v`.
- `check_exact_nash_orders.py` passed its residual, analytic-derivative and implementation comparisons. The fitted displacement slope was `-1.9854381024393806`; the Jacobian slope was `-0.9879046982119686`.
- `check_matched_approximation_summary.py` verified 3,000 unique matched cases and reproduced both improvement frequencies and all three tail-error values.
- `audit_cnsns_components.py` passed its third-moment enumeration, cubic-order and descriptive margin-stratification checks.
- `plot_stability_intervals_v3.py` represented all 3,808 scan observations, 28 feedback branches and 11 refined boundaries. Every scanned sign agreed with its displayed interval.

Verification was performed in this separate release directory, not in the manuscript workspace. The full experiment suites were not rerun for this release. Reference data are supplied for those larger calculations; the reproduction commands are in `REPRODUCIBILITY.md`.
