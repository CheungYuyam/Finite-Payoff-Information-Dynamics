# Changes from the public repository

Baseline commit: `86da7c8e48d9560f899b01d751b96477610e8d81`.

## Scientific reproduction changes

- Added exact four-combination large-m calculations, with theoretical coefficients derived from payoff moments, analytic binomial derivatives and numerical audits.
- Added analytic finite-N polynomial differentiation, existing-generator comparisons, and three-step derivative audits.
- Integrated the independently verified Hopf calculation, full eigenvalue/equilibrium output, unit-norm eigenvectors, transversality and derivative/normal-form robustness checks.
- Added source/configuration hashes and runtime metadata for all three workflows.
- Added the Figure 3 integration/plotting entry point and its checked trajectory data from the manuscript workspace.
- Added seven current-manuscript regression tests, retaining all 31 existing tests.
- Replaced outdated README/reproduction statements and mapped every numerical table and figure to an entry point. Existing additional workflows are preserved but separated from the submitted paper's reproduction path.
- Updated dependency pins to the versions used in this release validation. Refreshed deterministic audit metadata; unchanged simulation data and seeds retain their original provenance.

No scientific model, payoff matrix, response normalization, sampling convention or manuscript text was changed. No repository files were removed from the baseline.

## File inventory

`A` denotes an added file and `M` a modified file. The machine-readable reference files are regenerated outputs, not manually adjusted targets.

- M `README.md`
- A `code/check_exact_four_variants.py`
- A `code/check_finite_population_analytic.py`
- A `code/plot_manuscript_dynamics.py`
- A `code/reproduce_current_manuscript.py`
- A `code/reproduction_metadata.py`
- A `code/verify_hopf.py`
- A `configs/current_manuscript_checks.json`
- M `docs/REPRODUCIBILITY.md`
- A `docs/UPLOAD_GUIDE_zh.md`
- A `docs/VALIDATION.md`
- M `requirements.txt`
- M `results/cnsns_component_audit/audit.json`
- A `results/exact_four_variants/exact_four_variants.csv`
- A `results/exact_four_variants/summary.csv`
- A `results/exact_four_variants/summary.json`
- M `results/exact_nash_orders/summary.json`
- A `results/finite_population_analytic/finite_population_analytic.csv`
- A `results/finite_population_analytic/summary.json`
- A `results/hopf/hopf_results.json`
- A `results/revision_checks/three_strategy_trajectories.csv`
- A `results/revision_checks/trajectory_verification.json`
- A `tests/test_current_manuscript.py`
- A `docs/CHANGES.md` — this inventory.
- A `SHA256SUMS.txt` — checksums of all other packaged files.
