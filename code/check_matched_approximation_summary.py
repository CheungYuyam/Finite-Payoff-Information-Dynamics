"""Recompute the manuscript's matched approximation table from saved cases."""

from pathlib import Path
import csv
import json
import math

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = ROOT / "results" / "extended_full_e6_random_stress.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3000, "Expected the 3,000-case matched experiment"
    assert len({row["case_id"] for row in rows}) == len(rows)
    fields = ("relative_classical", "relative_moment3_tail", "relative_moment5_tail")
    errors = np.array([[float(row[key]) for key in fields] for row in rows])
    assert np.isfinite(errors).all() and (errors >= 0).all()
    quantiles = np.quantile(errors, 0.99, axis=0, method="linear")
    frequencies = [float(np.mean(errors[:, 1] < errors[:, 0])),
                   float(np.mean(errors[:, 2] < errors[:, 1]))]
    assert math.isclose(frequencies[0], 0.913, abs_tol=1e-12)
    assert math.isclose(frequencies[1], 2614 / 3000, abs_tol=1e-12)
    assert [f"{q:.2f}" for q in quantiles] == ["0.81", "1.83", "7.60"]
    print(json.dumps({
        "cases": len(rows), "quantile_method": "linear",
        "relative_error_99th_percentiles": dict(zip(fields, quantiles.tolist())),
        "F3_improvement_percent": 100 * frequencies[0],
        "F5_improvement_percent": 100 * frequencies[1],
        "status": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
