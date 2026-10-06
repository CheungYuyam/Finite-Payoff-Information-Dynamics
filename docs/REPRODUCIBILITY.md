# Reproducing the current manuscript

All commands use paths relative to the repository root. The item numbers below follow the final 39-page manuscript. Table 1 is a literature comparison and has no computational runner.

## Figure and table map

| Manuscript item | Entry point under `code/` | Reference files |
| --- | --- | --- |
| Figure 1: responses and pair fields | `plot_model_mechanism_v3.py` | Direct evaluations; `figures/model_mechanism_section2.pdf` |
| Figure 2: stationary-point convergence | `run_focused_finite_sample_completion_v3.py`; `plot_two_strategy_convergence.py` | `results/focused_finite_sample_completion/two_strategy_finite_sample_correction.csv`; `figures/value_full_fig_two_strategy_convergence.pdf` |
| Figure 3: deterministic trajectories | `plot_manuscript_dynamics.py` | `configs/extended_full.json`; `results/revision_checks/three_strategy_trajectories.csv`; `figures/three_strategy_dynamics.pdf` |
| Figure 4: stability continuation | `refine_stability_boundaries_v3.py --config configs/value_full.json`; `plot_value_checks_v3.py --profile value_full` | `results/value_full_e14_dense_scan.csv`, `value_full_e14_critical_boundaries.csv`; `figures/value_full_fig_e14_stability_continuation.pdf` |
| Table 2: cubic-field orders | `audit_cnsns_components.py` | `results/cnsns_component_audit/sampling_scaling.csv`, `sampling_scaling_summary.csv` |
| Table C1: four-combination exact orders | `check_exact_four_variants.py` | `results/exact_four_variants/exact_four_variants.csv`, `summary.csv`, `summary.json` |
| Table C2: margin stratification | `run_nature_v3.py --config configs/nature_full.json --stages e9`; `audit_cnsns_components.py` | `results/nature_full_e9_qualitative_map.csv`; `results/cnsns_component_audit/stability_margin_reanalysis.csv` |
| Table C3: analytic finite-N drift | `check_finite_population_analytic.py` | `results/finite_population_analytic/finite_population_analytic.csv`, `summary.json` |
| Table C4: Hopf classification | `verify_hopf.py` | `results/hopf/hopf_results.json` |
| Table E1: matched approximation errors | `run_extended_v3.py --config configs/extended_full.json`; `check_matched_approximation_summary.py` | `results/extended_full_e6_random_stress.csv`; printed summary |

Other numerical statements:

- Appendix C.2 event means and weak-response slopes: `run_v3.py` and `audit_v3.py`, with `configs/full_e1_e2.json`; outputs `results/full_e1_e2_*.csv`.
- Appendix C.3 smaller-m independent-Fermi slopes and implementation checks: `check_exact_nash_orders.py`; outputs `results/exact_nash_orders/`.
- Appendix C.4 continuation diagnostics: `refine_stability_boundaries_v3.py`; a full run also writes `results/value_full_e14_boundary_audit.json`.
- Appendix C.6 persistence and spectral certificate: `certify_weak_feedback_stability_v3.py`; `results/weak_feedback_stability_certificate_v3.json`.

## Current core reproduction

```sh
python code/check_exact_nash_orders.py
python code/check_exact_four_variants.py
python code/check_finite_population_analytic.py
python code/verify_hopf.py
python code/check_matched_approximation_summary.py
python -m unittest discover -s tests -p "test_*.py" -v
```

`reproduce_current_manuscript.py` runs the first five commands. It does not rerun all historical Monte Carlo experiments. Smoke configurations are separate and do not reproduce full-sample manuscript statistics.

### Exact orders

The game is `[[0,3],[2,0]]`, beta = 1. Independent sampling enumerates two binomial counts; common sampling uses one. The bounded response is evaluated before averaging. Analytic differentiation of binomial weights gives the Jacobian, with centered finite-difference audits at three steps. Fits use 256, 512, 1024 and 2048. Nash location, reference Jacobian, and limiting coefficients are derived from the game and its variance/third moments, not copied from targets.

Expected rounded slope pairs are (-1.996,-0.997), (-1.993,-0.994), (-1.976,-0.985), and (-1.957,-0.972), in Fermi-independent, Fermi-common, arctangent-independent and arctangent-common order. The smaller-m independent-Fermi calculation remains because the manuscript separately reports it; it checks `automatica_model.exact_rhs` at both stationary and off-equilibrium states.

### Hopf calculation

The printed payoff matrix comes from `configs/value_full.json`; m = 2 and independent Fermi sampling. Taylor jets provide analytic state derivatives through order three. Right and left eigenvectors satisfy q* q = 1 and p* q = 1. The raw derivative tensors B and C enter the Taylor expansion as B/2 and C/6. The Lyapunov formula is the Kuznetsov convention in Appendix C.8.

The JSON records critical beta, full equilibrium, conjugate eigenvalues, frequency, l1, transversality, normalization and residuals. Branch derivatives use steps 1e-4, 1e-5 and 1e-6, resolving the stationary state at each shifted parameter. Additional audits compare the field and Jacobian with the separate `automatica_model.exact_rhs` implementation, and recompute B/C and l1 by differencing lower derivatives at three state steps.

The two positive l1 values are approximately 0.030551706 and 0.037060946, at beta 0.761225162 and 0.800304099. This is a floating-point local classification, not an interval-certified bound or a global periodic-orbit continuation. It classifies only these two specified crossings.

### Finite-population calculation

For m = 1, beta = 0.5 and independent Fermi sampling, the script differentiates both the self-excluding payoff pools and pair weights analytically. Rounding uses the existing `initial_counts` function. Same-state drift is checked against the independently implemented generator, and Jacobians against centered differences of the polynomial at three steps.

Table C3 evaluates generally nonstationary rounded centers, not finite-N equilibria or a stationary distribution. The old `results/value_full_e13_finite_generator.csv` and its plot remain historical count-space diagnostics, not Table C3's source.

## Original full experiment workflows

These commands regenerate the original simulations, ensembles, broad scans and controls. They can take substantially longer than the core checks and overwrite existing outputs.

```sh
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
python code/reproduce_current_manuscript.py --figures
```

The focused-completion outputs contain earlier boundary estimates as well as Figure 2 data. Use the dense-scan roots for Figure 4 and the analytic-derivative Hopf calculation for Table C4. The optional interval plot is not an additional submitted figure.

## Additional retained workflows and provenance

Cross-condition fitting, approximation gates, cutoff scans and finite-time trajectory runners remain available for continuity but are outside the final manuscript reproduction path. Their bulk outputs are not all bundled. The optional finite-path simulator requires a C++17 compiler; no executable is included. These workflows do not imply submission of a supplementary experiments PDF.

New results record settings, configuration/source hashes and runtime versions. Historical data retain their provenance and are not relabelled as newly simulated. The new core calculations use the manuscript's matrices and normalization, with no random seeds. Last digits and numerical audit maxima can vary with library versions; see `VALIDATION.md`.

The matched-error table uses the same 3,000 cases for improvement frequencies and linear-interpolated 99th percentiles. Margin comparisons retain unsuccessful solves in attempted totals and distinguish neutral, unequal-label and opposite-sign cases.
