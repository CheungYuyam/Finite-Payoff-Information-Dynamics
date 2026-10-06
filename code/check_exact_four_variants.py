"""Exact finite-sum reproduction of manuscript Table C1 (all four combinations)."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln
import automatica_model as model
from reproduction_metadata import ROOT, metadata, settings, write_json

CFG = settings()['exact_orders']
A = np.asarray(CFG['payoff'])
DELTA_SLOPE = A[0,0]-A[1,0]-A[0,1]+A[1,1]
NASH = -(A[0,1]-A[1,1])/DELTA_SLOPE
J_STAR = NASH*(1-NASH)*DELTA_SLOPE

def theoretical_coefficients(kernel, design):
    weights = np.array([NASH, 1-NASH])
    centered = A - (A @ weights)[:,None]
    if design == 'independent':
        variance = np.sum(weights * (centered[0]**2 + centered[1]**2))
        third = np.sum(weights * (centered[0]**3 - centered[1]**3))
    else:
        difference = A[0]-A[1]
        difference -= difference @ weights
        variance = weights @ difference**2
        third = weights @ difference**3
    q_third = (-0.5 if kernel == 'fermi' else -2.0)*CFG['beta']**2
    displacement = -(NASH*(1-NASH)*q_third*third/6)/J_STAR
    jacobian = q_third/2 * NASH*(1-NASH)*variance*DELTA_SLOPE
    return float(displacement), float(jacobian)

def exact_two(m, kernel, design):
    k = np.arange(m + 1, dtype=float)
    lc = gammaln(m + 1) - gammaln(k + 1) - gammaln(m - k + 1)
    z = A[0, 1] + (A[0, 0] - A[0, 1]) * k[:, None] / m - A[1, 1] - (A[1, 0] - A[1, 1]) * k[None, :] / m if design == 'independent' else A[0, 1] - A[1, 1] + DELTA_SLOPE * k / m
    response = model.normalized_revision_flux(z, CFG['beta'], kernel=kernel)

    def evaluate(q):
        p = np.exp(lc + k * np.log(q) + (m - k) * np.log1p(-q))
        p /= p.sum()
        score = k / q - (m - k) / (1 - q)
        score -= p @ score
        dp = p * score
        if design == 'independent':
            e = p @ response @ p
            de = dp @ response @ p + p @ response @ dp
        else:
            e = p @ response
            de = dp @ response
        return (q * (1 - q) * e, (1 - 2 * q) * e + q * (1 - q) * de)
    q = brentq(lambda q: evaluate(q)[0], *CFG['root_bracket'], xtol=CFG['root_xtol'])
    f, j = evaluate(q)
    fd_gap = max(abs(j - (evaluate(q+h)[0] - evaluate(q-h)[0])/(2*h)) for h in CFG['derivative_steps'])
    if fd_gap >= 2e-8:
        raise RuntimeError('Analytic binomial Jacobian failed finite-difference audit')
    return dict(m=m, kernel=kernel, sampling=design, root=q, shift=q - NASH, jacobian=j, jacobian_gap=j - J_STAR, scaled_shift=m * m * (q - NASH), scaled_jacobian=m * (j - J_STAR), residual=abs(f), derivative_check=fd_gap)

def calculate():
    rows, summaries = [], []
    for kernel in CFG['kernels']:
        for design in CFG['sampling_designs']:
            group = [exact_two(m, kernel, design) for m in CFG['sample_sizes']]
            rows.extend(group)
            limits = theoretical_coefficients(kernel, design)
            slopes = [float(np.polyfit(np.log([r['m'] for r in group]),
                       np.log([abs(r[key]) for r in group]), 1)[0])
                       for key in ('shift', 'jacobian_gap')]
            summaries.append(dict(kernel=kernel, sampling=design,
                displacement_slope=slopes[0], jacobian_slope=slopes[1],
                predicted_displacement=limits[0], predicted_jacobian=limits[1],
                scaled_displacement=group[-1]['scaled_shift'],
                scaled_jacobian=group[-1]['scaled_jacobian']))
    return rows, summaries

def main():
    rows, summaries = calculate()
    out = ROOT/'results/exact_four_variants'
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [('exact_four_variants.csv', rows), ('summary.csv', summaries)]:
        with (out/name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    report = metadata([__file__, model.__file__], CFG)
    report.update(nash=NASH, reference_jacobian=J_STAR, variants=summaries,
        max_root_residual=max(r['residual'] for r in rows),
        max_derivative_discrepancy=max(r['derivative_check'] for r in rows))
    write_json(out/'summary.json', report)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
