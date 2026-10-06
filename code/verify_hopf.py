"""Independent exact finite-sum Hopf verification; requires numpy and scipy.
Taylor jets provide analytic state derivatives through order three.
The right eigenvector has unit Euclidean norm; the left eigenvector satisfies
np.vdot(p, q) = 1. Time and coordinates match manuscript Appendix C.8.
"""
import argparse
import json
import math
import itertools
from pathlib import Path
import numpy as np
from scipy.optimize import root, brentq
import automatica_model as model
from reproduction_metadata import ROOT, metadata, settings, write_json
CFG = settings()['hopf']
if (CFG['m'], CFG['kernel'], CFG['sampling_design']) != (2, 'fermi', 'independent'):
    raise ValueError('This Hopf reproduction is specialized to independent Fermi sampling with m=2.')
PAYOFF_CONFIG = ROOT / CFG['payoff_config']
A = np.array(json.loads(PAYOFF_CONFIG.read_text())['payoff'])
counts = np.array([(i, j, 2 - i - j) for i in range(3) for j in range(3 - i)])

def mul(a, b):
    """Multiply bivariate Taylor series, retaining total degree at most three."""
    out = np.zeros((4, 4), dtype=np.result_type(a, b))
    for i, j in itertools.product(range(4), repeat=2):
        if i + j > 3:
            continue
        for k, l in itertools.product(range(4 - i), range(4 - j)):
            if i + j + k + l <= 3:
                out[i + k, j + l] += a[i, j] * b[k, l]
    return out

def jets(y, beta):
    """Return the reduced field, analytic Jacobian, and raw derivative tensors B and C."""
    x = np.r_[y, 1 - sum(y)]
    xx = []
    for i in range(3):
        v = np.zeros((4, 4))
        v[0, 0] = x[i]
        v[1, 0] = int(i == 0) - int(i == 2)
        v[0, 1] = int(i == 1) - int(i == 2)
        xx.append(v)
    probs = []
    for n in counts:
        p = np.zeros((4, 4))
        p[0, 0] = 2 / np.prod([math.factorial(int(k)) for k in n])
        for i in range(3):
            for _ in range(n[i]):
                p = mul(p, xx[i])
        probs.append(p)
    f = np.zeros((3, 4, 4))
    for i in range(3):
        for j in range(i + 1, 3):
            z = (counts @ A[i])[:, None] / 2 - (counts @ A[j])[None, :] / 2
            q = 2 / beta * np.tanh(beta * z / 2)
            pair = np.zeros((4, 4))
            for k, l in itertools.product(range(6), repeat=2):
                pair += q[k, l] * mul(probs[k], probs[l])
            pair = mul(mul(xx[i], xx[j]), pair)
            f[i] += pair
            f[j] -= pair
    value = f[:2, 0, 0]
    J = np.column_stack([f[:2, 1, 0], f[:2, 0, 1]])
    tensors = []
    for degree in [2, 3]:
        t = np.zeros((2,) + (2,) * degree)
        for idx in itertools.product(range(2), repeat=degree):
            n0 = idx.count(0)
            n1 = idx.count(1)
            t[(slice(None),) + idx] = f[:2, n0, n1] * math.factorial(n0) * math.factorial(n1)
        tensors.append(t)
    return (value, J, *tensors)

def state(beta):
    """Solve the interior stationary state in coordinates (x1, x2), checking its residual."""
    r = root(lambda y: jets(y, beta)[0], CFG['stationary_start'], jac=lambda y: jets(y, beta)[1], tol=CFG['stationary_tol'])
    residual = np.linalg.norm(jets(r.x, beta)[0])
    if not np.isfinite(residual) or residual >= CFG['stationary_residual_limit']:
        raise RuntimeError(f'Stationary residual {residual} at beta={beta}; {r.message}')
    if min(r.x[0], r.x[1], 1.0 - sum(r.x)) <= 0:
        raise RuntimeError(f'Noninterior stationary state at beta={beta}')
    return r.x

def g(beta):
    """Real part of the conjugate eigenvalue pair near the two Hopf boundaries."""
    return np.trace(jets(state(beta), beta)[1]) / 2

def verify_hopf():
    """Reproduce both Hopf points and the three stationary-branch step-size checks."""
    out = []
    for bounds in CFG['root_brackets']:
        beta = brentq(g, *bounds, xtol=CFG['root_xtol'])
        y = state(beta)
        f, J, B, C = jets(y, beta)
        ev, vec = np.linalg.eig(J)
        idx = np.argmax(ev.imag)
        omega = ev[idx].imag
        q = vec[:, idx]
        ep, vp = np.linalg.eig(J.T)
        p = vp[:, np.argmin(abs(ep + 1j * omega))]
        p /= np.conj(np.vdot(p, q))
        b = lambda u, v: np.einsum('ijk,j,k->i', B, u, v)
        c = lambda u, v, w: np.einsum('ijkl,j,k,l->i', C, u, v, w)
        l1 = np.vdot(p, c(q, q, q.conj()) - 2 * b(q, np.linalg.solve(J, b(q, q.conj()))) + b(q.conj(), np.linalg.solve(2j * omega * np.eye(2) - J, b(q, q)))).real / (2 * omega)
        slopes = [(g(beta + h) - g(beta - h)) / (2 * h) for h in CFG['branch_derivative_steps']]
        audits = []
        for h in CFG['state_derivative_audit_steps']:
            lower = [jets(y-h*e, beta) for e in np.eye(2)]
            upper = [jets(y+h*e, beta) for e in np.eye(2)]
            b_fd = np.stack([(up[1]-lo[1])/(2*h) for up,lo in zip(upper,lower)], axis=-1)
            c_fd = np.stack([(up[2]-lo[2])/(2*h) for up,lo in zip(upper,lower)], axis=-1)
            b_audit = lambda u,v: np.einsum('ijk,j,k->i', b_fd, u, v)
            c_audit = lambda u,v,w: np.einsum('ijkl,j,k,l->i', c_fd, u,v,w)
            l1_audit = np.vdot(p, c_audit(q,q,q.conj())
                -2*b_audit(q,np.linalg.solve(J,b_audit(q,q.conj())))
                +b_audit(q.conj(),np.linalg.solve(2j*omega*np.eye(2)-J,b_audit(q,q)))).real/(2*omega)
            audits.append(dict(step=h, second_derivative_max_error=float(np.max(abs(B-b_fd))),
                third_derivative_max_error=float(np.max(abs(C-c_fd))),
                l1_from_differenced_derivatives=float(l1_audit), l1_difference=float(abs(l1-l1_audit))))
        def independent_field(v):
            return model.exact_rhs(np.r_[v,1-sum(v)],A,beta,2,
                                   kernel='fermi',sampling_mode='independent')[:2]
        h=1e-5
        j_fd=np.column_stack([(independent_field(y+h*e)-independent_field(y-h*e))/(2*h) for e in np.eye(2)])
        field_gap=max(float(np.max(abs(jets(v,beta)[0]-independent_field(v))))
                      for v in [y,np.array([.52,.4])])
        if omega <= 0 or field_gap > 1e-12 or max(abs(J-j_fd).flat) > 1e-7:
            raise RuntimeError('Hopf exact-field/eigenvalue audit failed')
        if np.linalg.norm(J@q-1j*omega*q) > 1e-10 or abs(np.vdot(p,q)-1) > 1e-10:
            raise RuntimeError('Hopf eigenvector normalization failed')
        out.append(dict(beta=beta, state=y.tolist(), equilibrium=np.r_[y,1-sum(y)].tolist(),
            eigenvalues=[[float(z.real),float(z.imag)] for z in ev], omega=omega, l1=l1,
            branch_slopes=slopes, qnorm=float(np.linalg.norm(q)),
            pq=[np.vdot(p, q).real, np.vdot(p, q).imag], stationary_residual=float(np.linalg.norm(f)),
            independent_field_gap=field_gap, independent_jacobian_step=h,
            independent_jacobian_gap=float(np.max(abs(J-j_fd))),
            derivative_audits=audits, classification='subcritical' if l1>0 else 'supercritical'))
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/hopf/hopf_results.json', help='JSON output path.')
    args = parser.parse_args()
    results = verify_hopf()
    report = metadata([__file__,model.__file__,PAYOFF_CONFIG],CFG)
    report['points']=results
    report['normalization']='q^*q=1; p^*q=1; raw derivative tensors B,C; normalized time'
    write_json(args.output,report)
    print(json.dumps(report,indent=2))
if __name__ == '__main__':
    main()
