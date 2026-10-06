"""Analytic finite-population polynomial drift for manuscript Table C3."""
import csv
import json
import numpy as np
from scipy.optimize import root
import automatica_model as model
from reproduction_metadata import ROOT, metadata, settings, write_json

CFG = settings()['finite_population']
if (CFG['m'], CFG['kernel'], CFG['sampling_design']) != (1, 'fermi', 'independent'):
    raise ValueError('This polynomial calculation requires independent Fermi sampling with m=1.')
PAYOFF_CONFIG = ROOT / CFG['payoff_config']
A = np.asarray(json.loads(PAYOFF_CONFIG.read_text())['payoff'])
U = np.vstack([np.eye(2), -np.ones(2)])
def drift_jac(y, N=None, beta=0.5):
    x = np.r_[y, 1 - np.sum(y)]
    f = np.zeros(3)
    jac = np.zeros((3, 2))
    rate = 1.0 if N is None else N / (N - 1)
    for i in range(3):
        for j in range(i + 1, 3):
            pi = x if N is None else (N * x - np.eye(3)[i]) / (N - 1)
            pj = x if N is None else (N * x - np.eye(3)[j]) / (N - 1)
            dpool = U if N is None else rate * U
            q = 2 / beta * np.tanh(beta * (A[i, :, None] - A[j, None, :]) / 2)
            e = pi @ q @ pj
            de = dpool.T @ q @ pj + pi @ q @ dpool
            w = rate * x[i] * x[j]
            df = rate * (U[i] * x[j] + x[i] * U[j]) * e + w * de
            f[i] += w * e
            f[j] -= w * e
            jac[i] += df
            jac[j] -= df
    return (f, jac[:2])

def calculate():
    xc = np.linalg.solve(np.vstack([A[:2]-A[2], np.ones(3)]), [0,0,1.])
    beta = CFG['beta']
    sol = root(lambda y: drift_jac(y, beta=beta)[0][:2], xc[:2],
               jac=lambda y: drift_jac(y, beta=beta)[1], tol=CFG['root_tol'])
    xe = np.r_[sol.x, 1-sol.x.sum()]
    residual = np.linalg.norm(drift_jac(sol.x, beta=beta)[0])
    if residual >= 1e-12 or not np.all(xe > 0):
        raise RuntimeError('Continuum stationary root failed validation')
    rows = []
    for N in CFG['populations']:
        counts = model.initial_counts(xe, N)
        x = counts/N
        f, jac = drift_jac(x[:2], N, beta)
        generator = model.finite_population_generator_rhs(counts, A, beta, 1,
                           kernel='fermi', sampling_mode='independent')
        implementation_gap = float(np.max(np.abs(f-generator)))
        checks = []
        for h in CFG['derivative_steps']:
            fd = np.column_stack([(drift_jac(x[:2]+h*e, N, beta)[0][:2]
                                  -drift_jac(x[:2]-h*e, N, beta)[0][:2])/(2*h)
                                  for e in np.eye(2)])
            checks.append(float(np.max(np.abs(jac-fd))))
        if implementation_gap >= 1e-12 or max(checks) >= 1e-8:
            raise RuntimeError('Finite-N drift/Jacobian audit failed')
        rows.append(dict(N=N, x1=x[0], x2=x[1], x3=x[2],
            spectral_abscissa=float(np.linalg.eigvals(jac).real.max()),
            same_state_drift_error=float(np.max(np.abs(f-drift_jac(x[:2], beta=beta)[0]))),
            center_residual=float(np.linalg.norm(f)), derivative_check=max(checks),
            generator_implementation_gap=implementation_gap))
    classical_jac = (np.diag(xc) @ (A-np.ones((3,1)) @ (xc@A+A@xc)[None,:]) @ U)[:2]
    summary = dict(continuum_equilibrium=xe.tolist(), continuum_root_residual=float(residual),
        continuum_spectral_abscissa=float(np.linalg.eigvals(drift_jac(sol.x, beta=beta)[1]).real.max()),
        classical_spectral_abscissa=float(np.linalg.eigvals(classical_jac).real.max()))
    return rows, summary

def main():
    rows, summary = calculate()
    out = ROOT/'results/finite_population_analytic'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'finite_population_analytic.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report=metadata([__file__, model.__file__, PAYOFF_CONFIG], CFG)
    report.update(summary)
    report['rows']=rows
    write_json(out/'summary.json', report)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
