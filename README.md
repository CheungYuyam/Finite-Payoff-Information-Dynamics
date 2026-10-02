# Finite-payoff-information dynamics

Reproducibility code for **Finite-Information Stochastic Dynamics in Population Games: Conditional Fields, Moment Corrections, and Stability Shifts**.

The main result concerns the exact conditional field at fixed response sensitivity: near a fully mixed Nash equilibrium with a nonsingular reduced Jacobian, the stationary-point displacement is `O(m^-2)` and the Jacobian correction is `O(m^-1)`. These are leading orders only when the corresponding coefficients are nonzero.

Independent and common sampling and normalized Fermi and arctangent responses are implemented. The direct exact-field order experiment uses independent sampling and the Fermi response; the four-combination order experiment uses the cubic surrogate. Small-feedback stability comparisons are distinct from large-feedback asymptotic tests.

## Layout

- `code/`: model, experiment runners, analysis and plotting scripts.
- `configs/`: frozen full and smoke-test settings, including additional experiments.
- `tests/`: model identities, derivative bounds and saved-result checks.
- `results/`: selected reference data supporting the manuscript and its appendices.
- `docs/REPRODUCIBILITY.md`: calculation-to-result map and extended workflows.

The original module name `automatica_model.py` and experiment identifiers are retained for compatibility. They do not indicate a different model or a publication venue. Manuscript sources, editorial notes, credentials, caches and compiled executables are not included.

## Installation

Use Python 3.13 for the dependency versions recorded in `requirements.txt`.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run commands from the repository root. No author-specific absolute paths are required.

## Quick verification

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python code/check_exact_nash_orders.py
python code/check_matched_approximation_summary.py
```

The exact-order script enumerates binomial observations rather than using a moment truncation. It writes five stationary points and analytic Jacobians for `m = 32, 64, 128, 256, 512`, checks derivatives numerically, and fits the last four feedback sizes. Expected slopes are approximately `-1.9854` and `-0.9879`, with scaled coefficients approaching `-0.028` and `0.936`.

The approximation-summary script recomputes Appendix E's table from the same 3,000 matched cases for both columns. Expected improvement frequencies are `91.30%` and `87.13%`; the 99th-percentile relative errors are `0.81`, `1.83` and `7.60`. These weak-response approximation comparisons are not tests of the exact-field asymptotic orders.

## Figures from included reference data

```bash
python code/plot_model_mechanism_v3.py
python code/plot_two_strategy_convergence.py
python code/plot_v3.py --profile extended_full
python code/plot_value_checks_v3.py --profile value_full
python code/plot_stability_intervals_v3.py
```

Outputs are written to `figures/`. The last command produces the current interval representation for the appendix stability figure; run it **after** `plot_value_checks_v3.py` to replace that script's older representation. Individual plotting scripts may also produce additional diagnostic figures.

## Recomputing experiments

Full runs can take substantially longer than the quick checks. See [the reproducibility guide](docs/REPRODUCIBILITY.md) for commands and the distinction between main-result and additional experiments. Runners write into `results/` and may overwrite included reference files; preserve a clean checkout for comparisons.

All supplied data are simulated matrix-game results, not empirical financial-market measurements. Numerical sign changes are reported as local-stability reversals, not as identified normal-form bifurcations.

## Citation and reuse

Publication metadata and a DOI have not been added because they are not yet supplied. No reuse license has been selected by the authors in this release; a public repository alone is not an open-source license.
