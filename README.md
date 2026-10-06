# Finite-Payoff-Information Dynamics

Reproducibility code for **Finite-Information Stochastic Dynamics in Population Games: Conditional Fields, Moment Corrections, and Stability Shifts**.

Near a nondegenerate fully mixed Nash equilibrium at fixed response sensitivity, the stationary-point displacement is `O(m^-2)` and the reduced-Jacobian correction is `O(m^-1)`, with these leading orders when the corresponding coefficients are nonzero.

The current workflows reproduce:

- Exact finite-sum orders for all four Fermi/arctangent and independent/common sampling combinations, with fits over `m = 256, 512, 1024, 2048` (Table C1).
- Two nondegenerate subcritical Hopf points for independent Fermi sampling with `m = 2`, using analytic derivatives and the first Lyapunov coefficient (Table C4).
- Analytically differentiated finite-population drift at rounded continuum stationary compositions (Table C3). These centers need not be finite-N equilibria.
- The smaller-m independent-Fermi audit, cubic moment comparisons, matched approximation errors, stability-margin comparisons, and the four manuscript figures.

## Install

Use Python 3.13 and the tested versions in `requirements.txt`:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Reproduce the current calculations

From the repository root:

```sh
python code/reproduce_current_manuscript.py
python -m unittest discover -s tests -p "test_*.py" -v
```

The first command performs full finite sums at the manuscript batch sizes, both Hopf classifications, and all eight population-size calculations. It also reruns the original independent-Fermi order audit and checks the matched-error summary from saved data. Outputs go to `results/`; preserve a clean copy for before/after comparisons.

To also regenerate the four manuscript figures:

```sh
python code/reproduce_current_manuscript.py --figures
```

Figures 1, 2 and 4 use direct field evaluations or included reference data. Figure 3 reintegrates the deterministic fields with a tighter-tolerance check. Plotting programs may also export diagnostic plots. The original ensembles and broad continuation scans have separate full runners described in [the reproduction guide](docs/REPRODUCIBILITY.md).

| Calculation | Entry point | Output directory |
| --- | --- | --- |
| Four-combination exact orders | `code/check_exact_four_variants.py` | `results/exact_four_variants/` |
| Hopf classification | `code/verify_hopf.py` | `results/hopf/` |
| Analytic finite-N drift | `code/check_finite_population_analytic.py` | `results/finite_population_analytic/` |
| Smaller-m independent-Fermi exact orders | `code/check_exact_nash_orders.py` | `results/exact_nash_orders/` |
| Cubic-order and margin checks | `code/audit_cnsns_components.py` | `results/cnsns_component_audit/` |

Table 2 uses the cubic moment field at weak response. Table C1 instead uses exact finite sums at fixed beta = 1. These are distinct experiments. Appendix E's error comparisons are also moment-approximation studies, not exact-field asymptotic tests.

## Contents and scope

`code/`, `configs/`, `results/` and `tests/` contain implementations, settings, reference data and checks. `configs/current_manuscript_checks.json` specifies the new core calculations. See `docs/REPRODUCIBILITY.md` for the table/figure map, `docs/VALIDATION.md` for validation and precision notes, and `docs/CHANGES.md` for changed files.

Existing additional-experiment runners remain available but are not required for the submitted manuscript. No supplementary experiments PDF is part of this package. The name `automatica_model.py` is retained for import compatibility. All data are simulated matrix-game results, not empirical market measurements.

The Hopf classification applies to the two specified boundaries; other archived crossings have not all been assigned normal-form classifications. The older count-space finite-difference finite-N output remains a historical diagnostic and is not the source of Table C3.

## Citation and reuse

Publication metadata and a DOI have not been supplied. No reuse license has been selected by the authors; this update does not assign one. The Hopf convention follows Y. A. Kuznetsov, *Andronov–Hopf bifurcation*, Scholarpedia 1(10) (2006), 1858, doi:10.4249/scholarpedia.1858.
