"""Cross-condition non-equivalence test for the v3 finite-feedback dynamics.

An effective payoff matrix is fitted once under an anchor feedback condition.
The same fixed matrix is then evaluated under changes in sample size, selection
strength, revision kernel, and feedback coupling.  Condition-specific refits
are retained only as an intentionally permissive oracle baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

import automatica_model as model
from run_nature_v3 import (
    atomic_csv,
    fit_effective_game,
    fit_time_scale,
    load_config,
    named_rhs,
    relative_rmse,
)


ROOT = Path(__file__).resolve().parents[1]


def condition_tuple(payload: dict) -> tuple[float, int, str, str]:
    return float(payload["beta"]), int(payload["m"]), payload["kernel"], payload["sampling_mode"]


def condition_name(payload: dict) -> str:
    return payload.get(
        "name",
        f"{payload['kernel']}_{payload['sampling_mode']}_b{payload['beta']}_m{payload['m']}",
    )


def rows(config: dict, config_hash: str):
    rng = np.random.default_rng(int(config["seed"]) + 1501)
    base_A = np.asarray(
        [[0.0, -0.37061410, 2.77335386], [0.24483140, 0.0, -2.70698466], [-2.08439346, 3.52730968, 0.0]],
        dtype=float,
    )
    anchor = config["anchor"]
    anchor_beta, anchor_m, anchor_kernel, anchor_mode = condition_tuple(anchor)
    alpha = float(config["dirichlet_alpha"])
    for case_id in range(int(config["cases"])):
        perturbation = rng.normal(scale=float(config["payoff_perturbation_scale"]), size=(3, 3))
        np.fill_diagonal(perturbation, 0.0)
        A = base_A + perturbation
        train = rng.dirichlet(np.full(3, alpha), size=int(config["train_states"]))
        test = rng.dirichlet(np.full(3, alpha), size=int(config["test_states"]))

        anchor_rhs = named_rhs(
            "exact", A, anchor_beta, anchor_m, kernel=anchor_kernel, sampling_mode=anchor_mode
        )
        anchor_train_targets = np.asarray([anchor_rhs(state) for state in train])
        anchor_A = fit_effective_game(train, anchor_train_targets)
        anchor_classical_train = np.asarray([model.classical_rhs(state, A) for state in train])
        anchor_scale = fit_time_scale(anchor_classical_train, anchor_train_targets)

        for target in config["targets"]:
            beta, sample_size, kernel, sampling_mode = condition_tuple(target)
            target_rhs = named_rhs("exact", A, beta, sample_size, kernel=kernel, sampling_mode=sampling_mode)
            train_targets = np.asarray([target_rhs(state) for state in train])
            test_targets = np.asarray([target_rhs(state) for state in test])
            classical_train = np.asarray([model.classical_rhs(state, A) for state in train])
            classical_test = np.asarray([model.classical_rhs(state, A) for state in test])
            oracle_A = fit_effective_game(train, train_targets)
            oracle_scale = fit_time_scale(classical_train, train_targets)
            predictions = {
                "original_classical": classical_test,
                "anchor_time_scale": anchor_scale * classical_test,
                "anchor_effective_game": np.asarray([model.classical_rhs(state, anchor_A) for state in test]),
                "oracle_time_scale": oracle_scale * classical_test,
                "oracle_effective_game": np.asarray([model.classical_rhs(state, oracle_A) for state in test]),
                "moment3_tail": np.asarray([
                    model.moment_tail_rhs(
                        state, A, beta, sample_size, order=3, kernel=kernel, sampling_mode=sampling_mode
                    ) for state in test
                ]),
                "moment5_tail": np.asarray([
                    model.moment_tail_rhs(
                        state, A, beta, sample_size, order=5, kernel=kernel, sampling_mode=sampling_mode
                    ) for state in test
                ]),
            }
            for name, prediction in predictions.items():
                yield {
                    "schema_version": config["schema_version"],
                    "config_hash": config_hash,
                    "case_id": case_id,
                    "anchor_condition": condition_name(anchor),
                    "target_condition": condition_name(target),
                    "is_anchor": int(condition_name(target) == condition_name(anchor)),
                    "beta": beta,
                    "m": sample_size,
                    "kernel": kernel,
                    "sampling_mode": sampling_mode,
                    "model": name,
                    "test_relative_rmse": f"{relative_rmse(prediction, test_targets):.17g}",
                    "anchor_time_scale_value": f"{anchor_scale:.17g}" if name == "anchor_time_scale" else "",
                    "oracle_time_scale_value": f"{oracle_scale:.17g}" if name == "oracle_time_scale" else "",
                }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "nature_cross_smoke.json")
    args = parser.parse_args()
    config, config_hash = load_config(args.config.resolve())
    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    profile = config["profile"]
    started = time.perf_counter()
    count = atomic_csv(
        output / f"{profile}_e10_cross_condition.csv",
        [
            "schema_version", "config_hash", "case_id", "anchor_condition", "target_condition",
            "is_anchor", "beta", "m", "kernel", "sampling_mode", "model", "test_relative_rmse",
            "anchor_time_scale_value", "oracle_time_scale_value",
        ],
        rows(config, config_hash),
    )
    manifest = {
        "profile": profile,
        "config_hash": config_hash,
        "rows": count,
        "elapsed_seconds": time.perf_counter() - started,
    }
    path = output / f"{profile}_e10_cross_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
