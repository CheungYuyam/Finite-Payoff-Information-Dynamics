"""Transparent sensitivity audit for the psi applicability gate.

This script does not fit parameters and does not use Exact error to select a
configuration. It reuses the completed paired-error table and reports how the
declared fallback policy changes when the weak-feedback boundary is moved.
The preregistered psi<=1 rule remains the reference; all alternatives are
sensitivity checks only.
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
        "q90_delta": float(np.quantile(delta, 0.90)),
        "q99_delta": float(np.quantile(delta, 0.99)),
        "win_rate": float(np.mean(delta < 0.0)),
        "overshoot_rate": float(np.mean(delta > 0.0)),
    }


def make_pivot(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot_table(index=KEYS, columns="model", values="relative_error", aggfunc="first").reset_index()
    pivot.columns.name = None
    return pivot


def evaluate(pivot: pd.DataFrame, psi_max: float, rho5_max: float, rho_col: str) -> dict:
    psi = pivot["psi"].to_numpy(float)
    rho = pivot[rho_col].to_numpy(float)
    use_m5 = (psi <= psi_max) & (rho <= rho5_max)
    pred = np.where(use_m5, pivot["bounded_moment5_tail"], pivot["bounded_moment3_tail"])
    safe = np.where(psi <= psi_max, pred, pivot["classical"].to_numpy(float))
    result = {"psi_max": psi_max, "rho5_max": rho5_max, "rho_column": rho_col, "regions": {}}
    for name, mask in (
        ("all", np.ones(len(pivot), dtype=bool)),
        ("main_work_region_psi_le_1", psi <= 1.0),
        ("outside_main_work_region_psi_gt_1", psi > 1.0),
        ("policy_active_region", psi <= psi_max),
    ):
        d = pred[mask] - pivot.loc[mask, "classical"].to_numpy(float)
        ds = safe[mask] - pivot.loc[mask, "classical"].to_numpy(float)
        result["regions"][name] = {
            "candidate_vs_classical": record(d, 2026090400 + int(100 * psi_max) + int(1000 * rho5_max)),
            "safe_policy_vs_classical": record(ds, 2026090500 + int(100 * psi_max) + int(1000 * rho5_max)),
        }
    for split_name, split in (
        ("train_even_case", pivot["case_id"].to_numpy(int) % 2 == 0),
        ("validation_odd_case", pivot["case_id"].to_numpy(int) % 2 == 1),
    ):
        main = split & (psi <= 1.0)
        d = pred[main] - pivot.loc[main, "classical"].to_numpy(float)
        ds = safe[split] - pivot.loc[split, "classical"].to_numpy(float)
        result.setdefault("holdout", {})[split_name] = {
            "candidate_main_vs_classical": record(d, 2026090600 + int(100 * psi_max)),
            "safe_all_vs_classical": record(ds, 2026090700 + int(100 * psi_max)),
        }
    return result


def main() -> None:
    frame = pd.read_csv(OUT / "realistic_sensitivity_bounded_rows.csv")
    pivot = make_pivot(frame)
    psi_values = [0.50, 0.75, 1.00, 1.25, 1.50]
    rho_values = [0.05, 0.10, 0.20]
    results = []
    rho_col = "fifth_relative_to_moment3"
    for psi_max in psi_values:
        for rho5_max in rho_values:
            results.append(evaluate(pivot, psi_max, rho5_max, rho_col))

    report = {
        "schema_version": "v3-psi-gate-sensitivity-1",
        "reference_policy": "psi_max=1.0, rho5_max=0.1, classical fallback outside psi gate",
        "source": str((OUT / "realistic_sensitivity_bounded_rows.csv").resolve()),
        "conditions": int(len(pivot)),
        "psi_values": psi_values,
        "rho_values": rho_values,
        "results": results,
        "guardrails": [
            "This is a transparent sensitivity audit, not post-hoc Exact-error tuning.",
            "psi<=1 remains the preregistered main work region.",
            "Strong-feedback failures of the bounded approximation remain visible.",
            "Case-level odd/even results are within-scan holdout robustness, not external validation.",
        ],
    }
    out_json = OUT / "psi_gate_sensitivity_audit_v3.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for item in results:
        main = item["regions"]["main_work_region_psi_le_1"]["candidate_vs_classical"]
        safe = item["regions"]["all"]["safe_policy_vs_classical"]
        val = item["holdout"]["validation_odd_case"]["candidate_main_vs_classical"]
        rows.append({
            "psi_max": item["psi_max"],
            "rho5_max": item["rho5_max"],
            "rho_column": item["rho_column"],
            "main_mean_delta": main["mean_delta"],
            "main_q99_delta": main["q99_delta"],
            "main_win_rate": main["win_rate"],
            "safe_all_mean_delta": safe["mean_delta"],
            "safe_all_win_rate": safe["win_rate"],
            "safe_all_overshoot_rate": safe["overshoot_rate"],
            "validation_main_mean_delta": val["mean_delta"],
            "validation_main_win_rate": val["win_rate"],
        })
    pd.DataFrame(rows).to_csv(OUT / "psi_gate_sensitivity_summary_v3.csv", index=False)

    lines = [
        "# psi 适用域阈值敏感性复核",
        "",
        "本审计固定已有模型和数据，只扫描理论上可解释的 psi 边界与 rho5 门控阈值；不根据 Exact 误差事后选模。",
        "预注册主规则仍为 psi<=1.0、rho5<=0.1；其余组合仅用于检查优势是否依赖孤立阈值。",
        "",
        "## 主要观察",
        "",
    ]
    lines.append(f"### 门控列：`{rho_col}`")
    for psi_max in psi_values:
        selected = [x for x in results if x["psi_max"] == psi_max and x["rho5_max"] == 0.1][0]
        m = selected["regions"]["main_work_region_psi_le_1"]["candidate_vs_classical"]
        s = selected["regions"]["all"]["safe_policy_vs_classical"]
        v = selected["holdout"]["validation_odd_case"]["candidate_main_vs_classical"]
        lines.append(
            f"- psi_max={psi_max:.2f}: 主区均值Δ={m['mean_delta']:.6g}，主区胜率={m['win_rate']:.3f}，"
            f"全域安全策略均值Δ={s['mean_delta']:.6g}，过冲率={s['overshoot_rate']:.3f}，"
            f"奇数 case 留出主区均值Δ={v['mean_delta']:.6g}，胜率={v['win_rate']:.3f}"
        )
    lines.append("")
    lines += [
        "## 解释边界",
        "",
        "psi 边界增大到强反馈区会把已知失效区域纳入近似，安全策略的尾部风险随之上升；因此不能为了扩大覆盖率而把 psi_max 提高到 1 以上。",
        "较低边界会更保守但牺牲覆盖率；psi=1.0 在当前扫描中兼顾主区收益、留出一致性与回退安全性，作为机制边界而非经验最优点。",
    ]
    (OUT / "psi_gate_sensitivity_audit_v3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(out_json), "summary_rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
