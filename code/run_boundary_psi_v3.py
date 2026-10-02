"""High-density, pre-specified scan of the psi≈1 applicability boundary."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import automatica_model as model


def load_config(path: Path) -> tuple[dict, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return config, hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def relative(pred: np.ndarray, exact: np.ndarray) -> float:
    return float(np.linalg.norm(pred - exact) / max(np.linalg.norm(exact), 1e-14))


def symmetric(pred: np.ndarray, exact: np.ndarray) -> float:
    return float(2.0 * np.linalg.norm(pred - exact) / max(np.linalg.norm(pred) + np.linalg.norm(exact), 1e-14))


def psi_values(A: np.ndarray, x: np.ndarray, beta: float, m, mode: str) -> tuple[float, float]:
    second = []
    variance = []
    weights = []
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            mu, var, _ = model.payoff_difference_moments(A, x, i, j, m, m, sampling_mode=mode)
            second.append(max(float(mu * mu + var), 0.0))
            variance.append(max(float(var), 0.0))
            weights.append(float(x[i] * x[j]))
    w = np.asarray(weights, dtype=float)
    w /= max(float(w.sum()), 1e-15)
    return beta * math.sqrt(float(w @ np.asarray(second))), beta * math.sqrt(float(w @ np.asarray(variance)))


def classify_state(x: np.ndarray) -> str:
    minimum = float(np.min(x))
    if minimum < 0.05:
        return "near_boundary"
    if minimum < 0.15:
        return "moderately_sparse"
    return "interior"


def generate(config: dict, config_hash: str):
    rng = np.random.default_rng(int(config["seed"]))
    low, high = map(float, config["psi_window"])
    case_id = 0
    for dimension in map(int, config["dimensions"]):
        for scale in map(float, config["scales"]):
            for alpha in map(float, config["dirichlet_alphas"]):
                for _ in range(int(config["games_per_dimension_per_scale_alpha"])):
                    A = rng.normal(size=(dimension, dimension))
                    np.fill_diagonal(A, 0.0)
                    A -= A.mean()
                    A *= scale / max(float(A.std()), 1e-12)
                    x = rng.dirichlet(np.full(dimension, alpha))
                    game_hash = hashlib.sha256(A.tobytes()).hexdigest()[:12]
                    state_class = classify_state(x)
                    entropy = float(-np.sum(x * np.log(np.maximum(x, 1e-300))))
                    for beta in map(float, config["betas"]):
                        for raw_m in config["sample_sizes"]:
                            m = None if raw_m is None else int(raw_m)
                            for kernel in config["kernels"]:
                                for mode in config["sampling_modes"]:
                                    started = time.perf_counter()
                                    exact = model.exact_rhs(x, A, beta, m, kernel=kernel, sampling_mode=mode)
                                    exact_seconds = time.perf_counter() - started
                                    classical = model.classical_rhs(x, A)
                                    moment3 = model.moment_tail_rhs(x, A, beta, m, order=3, kernel=kernel, sampling_mode=mode)
                                    moment5 = model.moment_tail_rhs(x, A, beta, m, order=5, kernel=kernel, sampling_mode=mode)
                                    bounded3 = model.bounded_moment_tail_rhs(x, A, beta, m, order=3, kernel=kernel, sampling_mode=mode)
                                    bounded5 = model.bounded_moment_tail_rhs(x, A, beta, m, order=5, kernel=kernel, sampling_mode=mode)
                                    psi, chi = psi_values(A, x, beta, m, mode)
                                    # Keep the boundary audit pre-specified: retain only diagnostic psi∈[0.65,1.35].
                                    if not (low <= psi <= high):
                                        del exact, classical, moment3, moment5, bounded3, bounded5
                                        gc.collect()
                                        continue
                                    rho5 = float(np.linalg.norm(moment5 - moment3) / max(np.linalg.norm(moment3), 1e-14))
                                    bounded_rho5 = float(np.linalg.norm(bounded5 - bounded3) / max(np.linalg.norm(bounded3), 1e-14))
                                    bounded3_active = int(np.linalg.norm(bounded3 - moment3) > 1e-12)
                                    bounded5_active = int(np.linalg.norm(bounded5 - moment5) > 1e-12)
                                    gated = moment5 if psi <= float(config["psi_main_max"]) else moment3
                                    bounded_gate = bounded5 if (psi <= float(config["psi_main_max"]) and rho5 <= float(config["rho5_max"])) else bounded3
                                    predictions = {
                                        "classical": classical,
                                        "moment3_tail": moment3,
                                        "moment5_tail": moment5,
                                        "bounded_moment3_tail": bounded3,
                                        "bounded_moment5_tail": bounded5,
                                        "bounded_rho_gate": bounded_gate,
                                        "psi_gate": gated,
                                    }
                                    base = {
                                        "schema_version": config["schema_version"],
                                        "config_hash": config_hash,
                                        "case_id": case_id,
                                        "game_hash": game_hash,
                                        "dimension": dimension,
                                        "scale": scale,
                                        "dirichlet_alpha": alpha,
                                        "state_class": state_class,
                                        "state_min_share": float(np.min(x)),
                                        "state_entropy": entropy,
                                        "beta": beta,
                                        "m": "inf" if m is None else str(m),
                                        "kernel": kernel,
                                        "sampling_mode": mode,
                                        "psi": psi,
                                        "psi_distance_to_one": abs(psi - 1.0),
                                        "chi": chi,
                                        "fifth_relative_to_moment3": rho5,
                                        "bounded_fifth_relative_to_moment3": bounded_rho5,
                                        "bounded3_projection_active": bounded3_active,
                                        "bounded5_projection_active": bounded5_active,
                                        "psi_in_main_region": int(psi <= float(config["psi_main_max"])),
                                        "exact_norm": float(np.linalg.norm(exact)),
                                        "exact_seconds": exact_seconds,
                                    }
                                    for name, prediction in predictions.items():
                                        yield {
                                            **base,
                                            "model": name,
                                            "relative_error": relative(prediction, exact),
                                            "symmetric_relative_error": symmetric(prediction, exact),
                                            "l2_error": float(np.linalg.norm(prediction - exact)),
                                            "mass_error": float(abs(float(prediction.sum()))),
                                        }
                                    del exact, classical, moment3, moment5, bounded3, bounded5, gated, bounded_gate, predictions
                                    gc.collect()
                    case_id += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, config_hash = load_config(args.config.resolve())
    output_dir = ROOT / "results" / "boundary_psi"
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "schema_version", "config_hash", "case_id", "game_hash", "dimension", "scale", "dirichlet_alpha",
        "state_class", "state_min_share", "state_entropy", "beta", "m", "kernel", "sampling_mode", "psi",
        "psi_distance_to_one", "chi", "fifth_relative_to_moment3", "bounded_fifth_relative_to_moment3",
        "bounded3_projection_active", "bounded5_projection_active", "psi_in_main_region", "exact_norm",
        "exact_seconds", "model", "relative_error", "symmetric_relative_error", "l2_error", "mass_error",
    ]
    path = output_dir / f"{config['profile']}_rows.csv"
    temporary = path.with_suffix(path.suffix + ".tmp")
    started = time.perf_counter()
    count = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in generate(config, config_hash):
            writer.writerow(row)
            count += 1
            if count % 500 == 0:
                handle.flush()
    temporary.replace(path)
    manifest = {
        "schema_version": config["schema_version"],
        "profile": config["profile"],
        "config_hash": config_hash,
        "config_path": str(args.config.resolve()),
        "rows": count,
        "elapsed_seconds": time.perf_counter() - started,
        "selection_rule": "retain diagnostic psi in pre-specified [0.65,1.35] window",
    }
    (output_dir / f"{config['profile']}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
