"""Holdout and threshold-sensitivity audit for the bounded gate refinement.

This is an audit, not a post-hoc model selector.  The original rho5=0.1
policy remains the preregistered reference.  Candidate thresholds are shown
for transparency, and the 0.2 candidate is evaluated on a case-level
odd/even holdout so that its apparent gain is not reported as universal.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "realistic_sensitivity"

KEYS = [
    "case_id", "dimension", "scale", "beta", "m", "kernel",
    "sampling_mode", "psi", "main_work_region", "fifth_relative_to_moment3",
]


def bootstrap_mean(values: np.ndarray, seed: int, reps: int = 4000) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def record(delta: np.ndarray, seed: int) -> dict:
    delta = np.asarray(delta, dtype=float)
    return {
        "n": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "mean_delta_bootstrap_95": bootstrap_mean(delta, seed),
        "median_delta": float(np.median(delta)),
        "q99_delta": float(np.quantile(delta, 0.99)),
        "win_rate": float(np.mean(delta < 0.0)),
        "overshoot_rate": float(np.mean(delta > 0.0)),
    }


def policy(frame: pd.DataFrame, rho5_max: float, psi_max: float = 1.0) -> np.ndarray:
    use_m5 = (frame["psi"].to_numpy(float) <= psi_max) & (
        frame["fifth_relative_to_moment3"].to_numpy(float) <= rho5_max
    )
    return np.where(use_m5, frame["bounded_moment5_tail"], frame["bounded_moment3_tail"])


def main() -> None:
    path = OUT / "realistic_sensitivity_bounded_rows.csv"
    frame = pd.read_csv(path)
    pivot = frame.pivot_table(
        index=KEYS, columns="model", values="relative_error", aggfunc="first"
    ).reset_index()
    pivot.columns.name = None

    thresholds = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    rows = []
    for tau in thresholds:
        pred = policy(pivot, tau)
        safe = np.where(pivot["psi"].to_numpy(float) <= 1.0, pred, pivot["classical"])
        for split_name, split in (
            ("all", np.ones(len(pivot), dtype=bool)),
            ("train_even_case", pivot["case_id"].to_numpy(int) % 2 == 0),
            ("validation_odd_case", pivot["case_id"].to_numpy(int) % 2 == 1),
        ):
            main = split & (pivot["main_work_region"].to_numpy(int) == 1)
            d_main = pred[main] - pivot.loc[main, "classical"].to_numpy(float)
            d_safe = safe[split] - pivot.loc[split, "classical"].to_numpy(float)
            rows.append({
                "rho5_max": tau,
                "split": split_name,
                "n": int(split.sum()),
                "main_n": int(main.sum()),
                "main_mean_delta_vs_classical": float(d_main.mean()),
                "main_q99_delta_vs_classical": float(np.quantile(d_main, 0.99)),
                "main_win_rate_vs_classical": float(np.mean(d_main < 0.0)),
                "safe_mean_delta_vs_classical": float(d_safe.mean()),
                "safe_win_rate_vs_classical": float(np.mean(d_safe < 0.0)),
                "safe_overshoot_rate": float(np.mean(d_safe > 0.0)),
            })
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(OUT / "bounded_gate_refinement_threshold_sensitivity_v3.csv", index=False)

    pred01 = policy(pivot, 0.1)
    pred02 = policy(pivot, 0.2)
    safe01 = np.where(pivot.psi.to_numpy(float) <= 1.0, pred01, pivot.classical)
    safe02 = np.where(pivot.psi.to_numpy(float) <= 1.0, pred02, pivot.classical)
    report = {
        "schema_version": "v3-bounded-gate-refinement-audit-1",
        "reference_policy": "rho5_max=0.1; psi_max=1; classical fallback for psi>1",
        "candidate_policy": "rho5_max=0.2; psi_max=1; classical fallback for psi>1",
        "data": {"source": str(path.resolve()), "conditions": int(len(pivot))},
        "candidate_vs_reference": {},
        "candidate_vs_classical": {},
        "holdout": {},
        "threshold_sensitivity_csv": str((OUT / "bounded_gate_refinement_threshold_sensitivity_v3.csv").resolve()),
        "guardrails": [
            "rho5=0.1 remains the preregistered reference and is not overwritten.",
            "All thresholds are reported; no candidate is called globally superior.",
            "Strong feedback psi>1 is evaluated only through the declared classical fallback.",
        ],
    }
    for name, mask in (
        ("all", np.ones(len(pivot), dtype=bool)),
        ("main_work_region", pivot.main_work_region.to_numpy(int) == 1),
        ("outside_main_work_region", pivot.main_work_region.to_numpy(int) == 0),
    ):
        report["candidate_vs_reference"][name] = record((safe02 - safe01)[mask], 20260911 + len(name))
        report["candidate_vs_classical"][name] = record((safe02 - pivot.classical.to_numpy(float))[mask], 20260921 + len(name))

    for split_name, split in (
        ("train_even_case", pivot.case_id.to_numpy(int) % 2 == 0),
        ("validation_odd_case", pivot.case_id.to_numpy(int) % 2 == 1),
    ):
        main = split & (pivot.main_work_region.to_numpy(int) == 1)
        report["holdout"][split_name] = {
            "candidate_vs_reference_main": record((pred02 - pred01)[main], 20260931 + len(split_name)),
            "candidate_vs_classical_main": record((pred02 - pivot.classical.to_numpy(float))[main], 20260941 + len(split_name)),
            "candidate_safe_vs_classical_all": record((safe02 - pivot.classical.to_numpy(float))[split], 20260951 + len(split_name)),
        }

    (OUT / "bounded_gate_refinement_audit_v3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# bounded-rho gate 0.2 候选复核",
        "",
        "原 0.1 规则保留为预注册参考；本报告只做透明的阈值敏感性和 case-level odd/even 留出复核。",
        "",
        "## 候选相对参考规则",
        "",
    ]
    for region, item in report["candidate_vs_reference"].items():
        lines.append(
            f"- {region}: Δ均值={item['mean_delta']:.6g}, 95% CI=[{item['mean_delta_bootstrap_95'][0]:.6g}, {item['mean_delta_bootstrap_95'][1]:.6g}], "
            f"胜率={item['win_rate']:.3f}, q99={item['q99_delta']:.6g}"
        )
    lines += ["", "## 留出复核", ""]
    for split, item in report["holdout"].items():
        x = item["candidate_vs_classical_main"]
        lines.append(
            f"- {split}: 主区候选−Classical Δ均值={x['mean_delta']:.6g}, "
            f"95% CI=[{x['mean_delta_bootstrap_95'][0]:.6g}, {x['mean_delta_bootstrap_95'][1]:.6g}], "
            f"胜率={x['win_rate']:.3f}"
        )
    lines += [
        "", "## 解释边界", "",
        "0.2 候选若在留出集保持方向一致，只能说明稳健性较好；它不能推出全域无条件优越。",
        "强反馈区仍由 Classical 回退处理，bounded 近似本身的 psi>1 失败结果不被删除。",
    ]
    (OUT / "bounded_gate_refinement_audit_v3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
