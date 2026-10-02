from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from automatica_model import classical_rhs, empirical_local_rhs, exact_rhs, tail_rhs


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> tuple[dict, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return config, digest


def m_label(value: int | None) -> str:
    return "inf" if value is None else str(value)


def write_csv(path: Path, fields: list[str], rows) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
            if count % 100 == 0:
                handle.flush()
    temporary.replace(path)
    return count


def local_rows(config: dict, config_hash: str):
    base_seed = int(config["empirical_seed"])
    samples = int(config["empirical_samples_per_pair"])
    case_number = 0
    for game in config["games"]:
        A = np.asarray(game["payoff"], dtype=float)
        for state_id, state_values in enumerate(game["states"]):
            x = np.asarray(state_values, dtype=float)
            classical = classical_rhs(x, A)
            for m in config["sample_sizes"]:
                for beta in config["local_betas"]:
                    case_number += 1
                    beta = float(beta)
                    exact = exact_rhs(x, A, beta, m)
                    variance = tail_rhs(
                        x, A, beta, m, include_skewness=False
                    )
                    moment = tail_rhs(x, A, beta, m, include_skewness=True)
                    empirical, standard_error = empirical_local_rhs(
                        x,
                        A,
                        beta,
                        m,
                        samples_per_pair=samples,
                        seed=base_seed + case_number,
                    )
                    errors = {
                        "classical": float(np.linalg.norm(classical - exact)),
                        "variance_tail": float(np.linalg.norm(variance - exact)),
                        "moment_tail": float(np.linalg.norm(moment - exact)),
                        "empirical": float(np.linalg.norm(empirical - exact)),
                    }
                    mass = {
                        "exact": abs(float(exact.sum())),
                        "classical": abs(float(classical.sum())),
                        "variance_tail": abs(float(variance.sum())),
                        "moment_tail": abs(float(moment.sum())),
                        "empirical": abs(float(empirical.sum())),
                    }
                    for component in range(len(x)):
                        yield {
                            "schema_version": config["schema_version"],
                            "config_hash": config_hash,
                            "profile": config["profile"],
                            "case_id": case_number,
                            "game": game["name"],
                            "state_id": state_id,
                            "component": component,
                            "m": m_label(m),
                            "beta": f"{beta:.12g}",
                            "x": f"{x[component]:.17g}",
                            "exact": f"{exact[component]:.17g}",
                            "classical": f"{classical[component]:.17g}",
                            "variance_tail": f"{variance[component]:.17g}",
                            "moment_tail": f"{moment[component]:.17g}",
                            "empirical": f"{empirical[component]:.17g}",
                            "empirical_se": f"{standard_error[component]:.17g}",
                            "covers_exact_95": int(
                                abs(empirical[component] - exact[component])
                                <= 1.96 * standard_error[component]
                            ),
                            "error_classical_l2": f"{errors['classical']:.17g}",
                            "error_variance_tail_l2": f"{errors['variance_tail']:.17g}",
                            "error_moment_tail_l2": f"{errors['moment_tail']:.17g}",
                            "error_empirical_l2": f"{errors['empirical']:.17g}",
                            "mass_exact": f"{mass['exact']:.17g}",
                            "mass_classical": f"{mass['classical']:.17g}",
                            "mass_variance_tail": f"{mass['variance_tail']:.17g}",
                            "mass_moment_tail": f"{mass['moment_tail']:.17g}",
                            "mass_empirical": f"{mass['empirical']:.17g}",
                        }
                    del exact, variance, moment, empirical, standard_error
                    gc.collect()


def order_data(config: dict, config_hash: str) -> tuple[list[dict], list[dict]]:
    raw_rows: list[dict] = []
    summary_rows: list[dict] = []
    fit_last = int(config["order_fit_last"])
    error_floor = float(config["order_error_floor"])
    for game in config["games"]:
        A = np.asarray(game["payoff"], dtype=float)
        for state_id, state_values in enumerate(game["states"]):
            x = np.asarray(state_values, dtype=float)
            classical = classical_rhs(x, A)
            for m in config["sample_sizes"]:
                by_model = {"classical": [], "variance_tail": [], "moment_tail": []}
                for beta_value in config["order_betas"]:
                    beta = float(beta_value)
                    exact = exact_rhs(x, A, beta, m)
                    predictions = {
                        "classical": classical,
                        "variance_tail": tail_rhs(
                            x, A, beta, m, include_skewness=False
                        ),
                        "moment_tail": tail_rhs(
                            x, A, beta, m, include_skewness=True
                        ),
                    }
                    for name, prediction in predictions.items():
                        error = float(np.linalg.norm(exact - prediction))
                        by_model[name].append((beta, error))
                        raw_rows.append(
                            {
                                "schema_version": config["schema_version"],
                                "config_hash": config_hash,
                                "profile": config["profile"],
                                "game": game["name"],
                                "state_id": state_id,
                                "m": m_label(m),
                                "model": name,
                                "beta": f"{beta:.12g}",
                                "error_l2": f"{error:.17g}",
                            }
                        )
                for name, pairs in by_model.items():
                    selected = sorted(pairs, reverse=True)[-fit_last:]
                    betas = np.asarray([pair[0] for pair in selected])
                    errors = np.asarray([pair[1] for pair in selected])
                    if float(np.max(errors)) <= error_floor or np.any(errors <= 0.0):
                        slope = ""
                        intercept = ""
                        fit_status = "degenerate_below_precision"
                    else:
                        fitted_slope, fitted_intercept = np.polyfit(
                            np.log(betas), np.log(errors), 1
                        )
                        slope = f"{float(fitted_slope):.17g}"
                        intercept = f"{float(fitted_intercept):.17g}"
                        fit_status = "fitted"
                    summary_rows.append(
                        {
                            "schema_version": config["schema_version"],
                            "config_hash": config_hash,
                            "profile": config["profile"],
                            "game": game["name"],
                            "state_id": state_id,
                            "m": m_label(m),
                            "model": name,
                            "slope": slope,
                            "intercept": intercept,
                            "fit_points": len(selected),
                            "fit_status": fit_status,
                        }
                    )
                gc.collect()
    return raw_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "smoke.json"
    )
    args = parser.parse_args()
    config, config_hash = load_config(args.config.resolve())
    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    prefix = config["profile"]
    started = time.perf_counter()
    local_fields = [
        "schema_version", "config_hash", "profile", "case_id", "game",
        "state_id", "component", "m", "beta", "x", "exact", "classical",
        "variance_tail", "moment_tail", "empirical", "empirical_se",
        "covers_exact_95", "error_classical_l2", "error_variance_tail_l2",
        "error_moment_tail_l2", "error_empirical_l2", "mass_exact",
        "mass_classical", "mass_variance_tail", "mass_moment_tail",
        "mass_empirical"
    ]
    local_count = write_csv(
        output / f"{prefix}_local_drift.csv",
        local_fields,
        local_rows(config, config_hash),
    )
    raw, summary = order_data(config, config_hash)
    raw_count = write_csv(
        output / f"{prefix}_error_order.csv",
        ["schema_version", "config_hash", "profile", "game", "state_id", "m", "model", "beta", "error_l2"],
        raw,
    )
    summary_count = write_csv(
        output / f"{prefix}_error_order_summary.csv",
        ["schema_version", "config_hash", "profile", "game", "state_id", "m", "model", "slope", "intercept", "fit_points", "fit_status"],
        summary,
    )
    manifest = {
        "schema_version": config["schema_version"],
        "profile": prefix,
        "config_hash": config_hash,
        "config_path": str(args.config.resolve()),
        "rows": {"local_drift": local_count, "error_order": raw_count, "error_order_summary": summary_count},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / f"{prefix}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
