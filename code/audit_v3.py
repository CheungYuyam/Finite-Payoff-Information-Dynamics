from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> tuple[dict, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return config, digest


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def expected_counts(config: dict) -> dict[str, int]:
    state_dimensions = sum(
        len(state)
        for game in config["games"]
        for state in game["states"]
    )
    state_count = sum(len(game["states"]) for game in config["games"])
    m_count = len(config["sample_sizes"])
    return {
        "local_drift": state_dimensions * m_count * len(config["local_betas"]),
        "error_order": state_count * m_count * len(config["order_betas"]) * 3,
        "error_order_summary": state_count * m_count * 3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "smoke.json"
    )
    args = parser.parse_args()
    config, config_hash = load_config(args.config.resolve())
    prefix = config["profile"]
    result_dir = ROOT / "results"
    files = {
        "local_drift": result_dir / f"{prefix}_local_drift.csv",
        "error_order": result_dir / f"{prefix}_error_order.csv",
        "error_order_summary": result_dir / f"{prefix}_error_order_summary.csv",
    }
    failures: list[str] = []
    rows: dict[str, list[dict[str, str]]] = {}
    for name, path in files.items():
        if not path.exists():
            failures.append(f"missing file: {path.name}")
            rows[name] = []
        else:
            rows[name] = read_rows(path)
    expected = expected_counts(config)
    for name, expected_count in expected.items():
        if len(rows[name]) != expected_count:
            failures.append(
                f"{name}: expected {expected_count} rows, found {len(rows[name])}"
            )
    for name, records in rows.items():
        if any(record.get("schema_version") != config["schema_version"] for record in records):
            failures.append(f"{name}: schema version mismatch")
        if any(record.get("config_hash") != config_hash for record in records):
            failures.append(f"{name}: config hash mismatch")
    unique_specs = {
        "local_drift": ("game", "state_id", "m", "beta", "component"),
        "error_order": ("game", "state_id", "m", "model", "beta"),
        "error_order_summary": ("game", "state_id", "m", "model"),
    }
    for name, keys in unique_specs.items():
        identities = [tuple(row[key] for key in keys) for row in rows[name]]
        if len(identities) != len(set(identities)):
            failures.append(f"{name}: duplicate logical keys")
    numeric_fields = {
        "local_drift": [
            "beta", "x", "exact", "classical", "variance_tail", "moment_tail",
            "empirical", "empirical_se", "error_classical_l2",
            "error_variance_tail_l2", "error_moment_tail_l2",
            "error_empirical_l2", "mass_exact", "mass_classical",
            "mass_variance_tail", "mass_moment_tail", "mass_empirical"
        ],
        "error_order": ["beta", "error_l2"],
        "error_order_summary": [],
    }
    for name, fields in numeric_fields.items():
        for row_number, row in enumerate(rows[name], start=2):
            for field in fields:
                try:
                    value = float(row[field])
                except (KeyError, ValueError):
                    failures.append(f"{name}:{row_number}: invalid {field}")
                    continue
                if not math.isfinite(value):
                    failures.append(f"{name}:{row_number}: non-finite {field}")
    allowed_fit_status = {"fitted", "degenerate_below_precision"}
    for row_number, row in enumerate(rows["error_order_summary"], start=2):
        status = row.get("fit_status")
        if status not in allowed_fit_status:
            failures.append(f"error_order_summary:{row_number}: invalid fit_status")
        if status == "fitted":
            for field in ("slope", "intercept"):
                try:
                    value = float(row[field])
                except (KeyError, ValueError):
                    failures.append(
                        f"error_order_summary:{row_number}: invalid fitted {field}"
                    )
                    continue
                if not math.isfinite(value):
                    failures.append(
                        f"error_order_summary:{row_number}: non-finite fitted {field}"
                    )
        elif row.get("slope") or row.get("intercept"):
            failures.append(
                f"error_order_summary:{row_number}: degenerate fit must have blank coefficients"
            )
    tolerance = float(config["audit"]["mass_tolerance"])
    mass_fields = [
        "mass_exact", "mass_classical", "mass_variance_tail",
        "mass_moment_tail", "mass_empirical"
    ]
    maximum_mass_error = max(
        (float(row[field]) for row in rows["local_drift"] for field in mass_fields),
        default=math.inf,
    )
    if maximum_mass_error > tolerance:
        failures.append(
            f"mass conservation: {maximum_mass_error:.3e} > {tolerance:.3e}"
        )
    coverage_values = [int(row["covers_exact_95"]) for row in rows["local_drift"]]
    empirical_coverage = (
        sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
    )
    minimum_coverage = float(config["audit"]["minimum_empirical_coverage"])
    if empirical_coverage < minimum_coverage:
        failures.append(
            f"empirical coverage: {empirical_coverage:.3f} < {minimum_coverage:.3f}"
        )
    slope_ranges = {
        "classical": config["audit"]["classical_slope_range"],
        "moment_tail": config["audit"]["moment_slope_range"],
    }
    slope_report: dict[str, dict[str, float]] = {}
    for model_name, bounds in slope_ranges.items():
        values = [
            float(row["slope"])
            for row in rows["error_order_summary"]
            if row["model"] == model_name and row.get("fit_status") == "fitted"
        ]
        slope_report[model_name] = {
            "minimum": min(values, default=math.nan),
            "maximum": max(values, default=math.nan),
            "mean": sum(values) / len(values) if values else math.nan,
        }
        outside = [value for value in values if not (float(bounds[0]) <= value <= float(bounds[1]))]
        if outside:
            failures.append(
                f"{model_name}: {len(outside)} slopes outside [{bounds[0]}, {bounds[1]}]"
            )
    report = {
        "schema_version": config["schema_version"],
        "profile": prefix,
        "config_hash": config_hash,
        "passed": not failures,
        "row_counts": {name: len(records) for name, records in rows.items()},
        "expected_row_counts": expected,
        "maximum_mass_error": maximum_mass_error,
        "empirical_component_coverage_95": empirical_coverage,
        "slope_report": slope_report,
        "degenerate_order_fits": sum(
            row.get("fit_status") == "degenerate_below_precision"
            for row in rows["error_order_summary"]
        ),
        "failures": failures,
    }
    report_path = result_dir / f"{prefix}_integrity.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
