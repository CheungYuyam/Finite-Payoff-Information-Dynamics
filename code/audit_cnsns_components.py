"""Focused, deterministic checks added in the CNSNS component review.

No previous outputs are overwritten. E9 is reanalysed, not rerun.
The inverse-sample-size checks use a fixed game/state and fixed beta.
"""
from pathlib import Path
from itertools import product
import csv
import hashlib
import json
import platform
import sys

import numpy as np
import scipy
from scipy.optimize import brentq

import automatica_model as model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cnsns_component_audit"
A = np.array([[0., 3.], [2., 0.]])
BETA = 0.2
MS = [8, 16, 32, 64, 128, 256]


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def coefficients(game, x, kernel, mode):
    """Direct one-draw central moments; independent of cumulant implementation."""
    a = {"fermi": 1/12, "arctan": 1/3}[kernel]
    pi = game @ x
    centered = game - pi[:, None]
    var = centered**2 @ x
    skew = centered**3 @ x
    H, S = np.zeros(len(x)), np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                continue
            delta = pi[i] - pi[j]
            if mode == "independent":
                V, T = var[i] + var[j], skew[i] - skew[j]
            else:
                d = game[i] - game[j] - delta
                V, T = d**2 @ x, d**3 @ x
            H[i] -= 3*a*x[i]*x[j]*delta*V
            S[i] -= a*x[i]*x[j]*T
    return H, S


def enumeration_checks():
    game = np.array([[0., 1.7, -0.6], [1.2, -0.3, 0.4], [-0.2, 0.8, 1.]])
    x = np.array([.2, .3, .5])
    worst = 0.
    count = 0
    # Exhaustive ordered draws, not multinomial/cumulant code.
    for mode, mi, mj in [("independent", 1, 3), ("independent", 2, 3),
                         ("independent", 3, 1), ("common", 1, 1),
                         ("common", 2, 2), ("common", 3, 3)]:
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            raw = 0.
            draws = mi + mj if mode == "independent" else mi
            for seq in product(range(3), repeat=draws):
                p = float(np.prod(x[list(seq)]))
                if mode == "independent":
                    z = game[i, list(seq[:mi])].mean() - game[j, list(seq[mi:])].mean()
                else:
                    z = (game[i] - game[j])[list(seq)].mean()
                raw += p*z**3
            pi, pj = game[i] @ x, game[j] @ x
            d = pi-pj
            if mode == "independent":
                ci, cj = game[i]-pi, game[j]-pj
                v = ci**2 @ x / mi + cj**2 @ x / mj
                t = ci**3 @ x / mi**2 - cj**3 @ x / mj**2
            else:
                c = game[i]-game[j]-d
                v, t = c**2 @ x / mi, c**3 @ x / mi**2
            predicted = d**3+3*d*v+t
            worst = max(worst, abs(raw-predicted))
            assert abs(raw-predicted) < 2e-12
            count += 1
    return {"pairs_checked": count, "max_absolute_error": worst}


def scaling_checks():
    rows, fits = [], []
    x, star = np.array([.37, .63]), np.array([.6, .4])
    for kernel, mode in product(["fermi", "arctan"], ["independent", "common"]):
        H, S = coefficients(A, x, kernel, mode)
        Hstar, Sstar = coefficients(A, star, kernel, mode)
        assert np.linalg.norm(Hstar) < 1e-14
        # F_C'(q*) = -5 q*(1-q*) for this fixed two-strategy game.
        J = -1.2
        shift_coefficient = -BETA**2*Sstar[0]/J
        h = 1e-5
        Hp = coefficients(A, np.array([.6+h, .4-h]), kernel, mode)[0][0]
        Hm = coefficients(A, np.array([.6-h, .4+h]), kernel, mode)[0][0]
        jac_coefficient = BETA**2*(Hp-Hm)/(2*h)
        m3 = lambda state, m: model.moment_tail_rhs(
            state, A, BETA, m, order=3, kernel=kernel, sampling_mode=mode)
        baseline = m3(x, None)
        block = []
        for m in MS:
            gap = m3(x, m)-baseline
            predicted = BETA**2*(H/m+S/m**2)
            np.testing.assert_allclose(gap, predicted, rtol=1e-10, atol=2e-14)
            scalar = lambda q: m3(np.array([q, 1-q]), m)[0]
            root = brentq(scalar, .55, .65, xtol=5e-15)
            jac = (scalar(root+h)-scalar(root-h))/(2*h)
            row = dict(kernel=kernel, sampling=mode, m=m, beta=BETA,
                       field_gap=float(np.linalg.norm(gap, np.inf)),
                       field_residual_after_variance=float(np.linalg.norm(gap-BETA**2*H/m, np.inf)),
                       identity_error=float(np.linalg.norm(gap-predicted, np.inf)),
                       root=root, shift=root-.6, scaled_shift=m*m*(root-.6),
                       predicted_scaled_shift=shift_coefficient,
                       jacobian_gap=jac-J, scaled_jacobian_gap=m*(jac-J),
                       predicted_scaled_jacobian_gap=jac_coefficient)
            rows.append(row)
            block.append(row)
        slope = lambda key: float(np.polyfit(np.log(MS[-4:]),
            np.log([abs(r[key]) for r in block[-4:]]), 1)[0])
        fit = dict(kernel=kernel, sampling=mode, field_slope=slope("field_gap"),
                   remainder_slope=slope("field_residual_after_variance"),
                   shift_slope=slope("shift"), jacobian_slope=slope("jacobian_gap"),
                   shift_coefficient_relative_error=abs(block[-1]["scaled_shift"]/shift_coefficient-1),
                   jacobian_coefficient_relative_error=abs(block[-1]["scaled_jacobian_gap"]/jac_coefficient-1))
        assert abs(fit["field_slope"]+1) < .03
        assert abs(fit["remainder_slope"]+2) < .005
        assert abs(fit["shift_slope"]+2) < .03
        assert abs(fit["jacobian_slope"]+1) < .03
        assert fit["shift_coefficient_relative_error"] < .01
        assert fit["jacobian_coefficient_relative_error"] < .01
        fits.append(fit)
    write_csv("sampling_scaling.csv", rows)
    write_csv("sampling_scaling_summary.csv", fits)
    return fits


def margin_summary():
    source = ROOT / "results" / "nature_full_e9_qualitative_map.csv"
    with source.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Descriptive, post-hoc bins; retain neutral labels and failed solves separately.
    bins = [(0., .001), (.001, .01), (.01, .1), (.1, np.inf)]
    output = []
    for ensemble in sorted({r["ensemble"] for r in rows}):
        for lo, hi in bins:
            group = [r for r in rows if r["ensemble"] == ensemble and
                     lo <= abs(float(r["classical_stability"])) < hi]
            valid = [r for r in group if r["classical_valid"] == "1" and r["exact_valid"] == "1"]
            neutral = [r for r in valid if int(r["classical_label"]) == 0 or int(r["exact_label"]) == 0]
            opposite = sum(int(r["classical_label"])*int(r["exact_label"]) == -1 for r in valid)
            mismatch = sum(r["classical_label"] != r["exact_label"] for r in valid)
            output.append(dict(ensemble=ensemble, margin_min=lo, margin_max="inf" if np.isinf(hi) else hi,
                               n=len(group), valid=len(valid), failed=len(group)-len(valid),
                               neutral=len(neutral), label_mismatch=mismatch,
                               opposite_sign=opposite))
    assert sum(r["n"] for r in output) == len(rows)
    write_csv("stability_margin_reanalysis.csv", output)
    return {"source": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "rows": len(rows), "groups": output, "analysis": "post-hoc descriptive reanalysis"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = dict(game=A.tolist(), state=[.37, .63], equilibrium=[.6, .4],
                   beta=BETA, sample_sizes=MS, slope_fit_sizes=MS[-4:],
                   python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                   platform=platform.platform(),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   model_sha256=hashlib.sha256((ROOT/"code"/"automatica_model.py").read_bytes()).hexdigest())
    summary["enumeration"] = enumeration_checks()
    summary["scaling"] = scaling_checks()
    summary["margin"] = margin_summary()
    summary["status"] = "passed"
    (OUT/"audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
