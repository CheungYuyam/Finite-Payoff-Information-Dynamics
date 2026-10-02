# Reproducing the calculations

## Main-result map

| Manuscript calculation | Entry point | Reference output |
| --- | --- | --- |
| Conditional field and weak-response checks | `run_v3.py`, `audit_v3.py` | `full_e1_e2_*.csv` |
| Cubic feedback-size orders and third moments | `audit_cnsns_components.py` | `cnsns_component_audit/sampling_scaling*.csv` |
| Exact stationary-point and Jacobian orders | `check_exact_nash_orders.py` | `exact_nash_orders/` |
| Two-strategy stationary-point comparison and feedback-size certificates | `run_focused_finite_sample_completion_v3.py` | `focused_finite_sample_completion/` |
| Three-strategy fields and trajectories | `run_extended_v3.py` | `extended_full_e3_*.csv` |
| Exact stability continuation and finite-population generator drift | `run_value_checks_v3.py` | `value_full_e13_*.csv`, `value_full_e14_stability_continuation.csv` |
| Refined stability intervals | `refine_stability_boundaries_v3.py` | `value_full_e14_dense_scan.csv`, `value_full_e14_critical_boundaries.csv` |
| Margin ensembles and stratification | `run_nature_v3.py --stages e9`, `audit_cnsns_components.py` | `nature_full_e9_qualitative_map.csv`, `cnsns_component_audit/stability_margin_reanalysis.csv` |
| Persistence and spectral sufficient conditions | `certify_weak_feedback_stability_v3.py` | `weak_feedback_stability_certificate_v3.json` |
| Matched approximation errors | `run_extended_v3.py`, `check_matched_approximation_summary.py` | `extended_full_e6_random_stress.csv` |

All commands below use paths relative to the repository root.

## Full main-result workflows

```bash
python code/run_v3.py --config configs/full_e1_e2.json
python code/audit_v3.py --config configs/full_e1_e2.json
python code/run_extended_v3.py --config configs/extended_full.json
python code/analyze_extended_v3.py --config configs/extended_full.json
python code/audit_extended_v3.py --config configs/extended_full.json
python code/run_nature_v3.py --config configs/nature_full.json --stages e9
python code/run_value_checks_v3.py --config configs/value_full.json
python code/refine_stability_boundaries_v3.py --config configs/value_full.json
python code/run_focused_finite_sample_completion_v3.py
python code/audit_cnsns_components.py
python code/certify_weak_feedback_stability_v3.py
python code/check_exact_nash_orders.py
python code/check_matched_approximation_summary.py
```

The component check reads the saved margin-ensemble data. Generate that data first if starting without reference outputs. The supplied smoke configurations use separate profile names and are intended for checking workflows, not reproducing manuscript statistics.

The retained focused-completion files also contain earlier control calculations. Their boundary estimates need not match the refined continuation values digit-for-digit because the numerical procedures differ. Use `value_full_e14_critical_boundaries.csv` for the refined interval figure.

## Additional experiments

Additional source and configuration files are retained so the computational work is not lost, but their bulk outputs are not included. They are not required to verify the main theorem's numerical examples.

- `run_cross_condition_v3.py` and `analyze_cross_condition_v3.py`: cross-feedback fitting comparisons, using `nature_cross_*.json`.
- Additional stages of `run_nature_v3.py`: payoff fitting, paired approximation errors and finite-population trajectory comparisons. `analyze_nature_v3.py` and `plot_nature_v3.py` require their associated full outputs, not just the supplied margin data.
- `run_boundary_psi_v3.py`: additional approximation-boundary comparisons, using `boundary_psi_*.json`.
- `run_realistic_sensitivity_v3.py`: additional sensitivity analyses, using `realistic_regime_sensitivity_*.json`. Configuration choices and output arguments are exposed by `--help`.
- `run_finite_paths_v3.py`: finite-time trajectory calculations, using `finite_path_*.json`. This optional workflow requires `g++` with C++17 support; the C++ source is included but platform-specific binaries are excluded.

Do not treat additional approximation or trajectory calculations as proofs of fixed-response exact-field orders.

## Reference data and precision

Selected CSVs and result JSONs are copied from the research workspace without modifying numerical entries. The exact-order script includes source hashes and records the runtime used for each rerun. Floating-point roundoff and numerical-library differences can affect the last digits of fitted slopes and refined roots.

The approximation summary uses a linear-interpolated sample quantile: for sorted errors and percentile `p`, the zero-based position is `p * (n - 1)`. Both the improvement frequencies and tail errors come from the same matched sample, not from a separate parameter grid.

For margin ensembles, unsuccessful equilibrium calculations are retained in totals but excluded from valid equilibrium comparisons. Neutral labels and opposite-sign mismatches are distinct quantities.
