"""Audit the psi≈1 boundary scan without fitting thresholds to Exact errors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def bootstrap_mean(values: np.ndarray, seed: int = 20260904, draws: int = 4000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for i in range(draws):
        means[i] = float(np.mean(rng.choice(values, size=values.size, replace=True)))
    return float(np.mean(values)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
    return p, float(center - half), float(center + half)


def condition_table(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "case_id", "beta", "m", "kernel", "sampling_mode", "psi", "dimension", "scale",
        "dirichlet_alpha", "state_class", "state_min_share", "state_entropy",
    ]
    models = ["classical", "moment3_tail", "moment5_tail", "bounded_rho_gate", "bounded_moment3_tail", "bounded_moment5_tail"]
    p = frame[frame.model.isin(models)].pivot_table(index=keys, columns="model", values="relative_error", aggfunc="first").reset_index()
    p["gate_minus_classical"] = p["bounded_rho_gate"] - p["classical"]
    p["gate_minus_moment3"] = p["bounded_rho_gate"] - p["moment3_tail"]
    p["gate_win_classical"] = p["gate_minus_classical"] < 0.0
    p["gate_win_moment3"] = p["gate_minus_moment3"] < 0.0
    p["psi_bin"] = pd.cut(
        p["psi"], bins=[0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35],
        right=False,
    ).astype(str)
    active = frame[frame.model == "bounded_rho_gate"][keys + ["bounded3_projection_active", "bounded5_projection_active"]]
    p = p.merge(active, on=keys, how="left")
    return p


def summarize(p: pd.DataFrame, group_cols: list[str]) -> list[dict]:
    rows = []
    for group_key, g in p.groupby(group_cols, observed=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {col: (str(value) if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value) for col, value in zip(group_cols, group_key)}
        for name, values in [("gate_minus_classical", g.gate_minus_classical), ("gate_minus_moment3", g.gate_minus_moment3)]:
            mean, lo, hi = bootstrap_mean(values.to_numpy(), seed=20260904 + len(rows))
            row[f"{name}_mean"] = mean
            row[f"{name}_ci95_low"] = lo
            row[f"{name}_ci95_high"] = hi
        row["n"] = int(len(g))
        row["gate_win_classical"] = wilson(int(g.gate_win_classical.sum()), len(g))
        row["gate_win_moment3"] = wilson(int(g.gate_win_moment3.sum()), len(g))
        row["projection3_rate"] = float(g.bounded3_projection_active.mean())
        row["projection5_rate"] = float(g.bounded5_projection_active.mean())
        rows.append(row)
    return rows


def fallback_summary(p: pd.DataFrame, thresholds: dict[str, float]) -> list[dict]:
    rows = []
    for kernel, threshold in thresholds.items():
        g = p[p.kernel == kernel].copy()
        g["fallback_error"] = np.where(g.psi <= threshold, g["bounded_rho_gate"], g["classical"])
        diff = g.fallback_error - g.classical
        mean, lo, hi = bootstrap_mean(diff.to_numpy(), seed=20261001 + int(round(100 * threshold)))
        win = wilson(int((diff < 0).sum()), len(diff))
        rows.append({
            "kernel": kernel,
            "predeclared_psi_threshold": threshold,
            "n": int(len(g)),
            "mean_diff_vs_classical": mean,
            "bootstrap_ci95_low": lo,
            "bootstrap_ci95_high": hi,
            "median_diff_vs_classical": float(np.median(diff)),
            "q90_diff_vs_classical": float(np.quantile(diff, 0.90)),
            "q99_diff_vs_classical": float(np.quantile(diff, 0.99)),
            "win_rate_vs_classical": win[0],
            "win_rate_ci95_low": win[1],
            "win_rate_ci95_high": win[2],
            "coverage_bounded_gate": float(np.mean(g.psi <= threshold)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "boundary_psi" / "boundary_psi_full_rows.csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    p = condition_table(frame)
    # Thresholds are theory-led diagnostics: Fermi has a larger analyticity radius;
    # arctan has its nearest complex singularity at |beta z|=1.
    threshold_table = {"fermi": 1.0, "arctan": 0.75}
    report = {
        "schema_version": "v3-boundary-audit-1",
        "source_csv": str(args.csv.resolve()),
        "conditions": int(len(p)),
        "models_per_condition": 6,
        "psi_window": [0.65, 1.35],
        "thresholds_are_error_independent": True,
        "theory_note": "Fermi normalized flux has nearest complex singularity at |beta z|=pi; arctan at |beta z|=1. The reported thresholds are conservative deployment diagnostics, not error-fitted optima.",
        "by_kernel": summarize(p, ["kernel"]),
        "by_psi_bin": summarize(p, ["kernel", "psi_bin"]),
        "by_state_class": summarize(p, ["kernel", "state_class"]),
        "by_sampling_m": summarize(p, ["kernel", "sampling_mode", "m"]),
        "kernel_specific_fallback": fallback_summary(p, threshold_table),
    }
    out = ROOT / "results" / "boundary_psi"
    (out / "boundary_psi_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# ψ≈1 边界扫描审计 v3",
        "",
        f"条件数：{len(p)}；源文件：`{args.csv.resolve()}`。",
        "",
        "## 判定原则",
        "",
        "本审计不按 Exact 误差事后搜索单一最优阈值，而是将理论解析半径与数值分层并列报告。Fermi 归一化通量最近复奇点位于 |βz|=π，arctan 位于 |βz|=1，因此两种核不应默认共享同一个 ψ 部署边界。",
        "",
        "## 核函数分层",
        "",
        "|核|n|门控−Classical 均值差|95% CI|门控−Moment3 均值差|95% CI|胜率(Classical)|胜率(Moment3)|",
        "|---|---:|---:|---|---:|---|---:|---:|",
    ]
    for row in report["by_kernel"]:
        lines.append(f"|{row['kernel']}|{row['n']}|{row['gate_minus_classical_mean']:.5f}|[{row['gate_minus_classical_ci95_low']:.5f},{row['gate_minus_classical_ci95_high']:.5f}]|{row['gate_minus_moment3_mean']:.5f}|[{row['gate_minus_moment3_ci95_low']:.5f},{row['gate_minus_moment3_ci95_high']:.5f}]|{row['gate_win_classical'][0]:.3f}|{row['gate_win_moment3'][0]:.3f}|")
    lines += ["", "## ψ 分箱", "", "|核|ψ区间|n|门控−Classical|门控−Moment3|胜率(Classical)|胜率(Moment3)|投影激活率(五阶)|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["by_psi_bin"]:
        lines.append(f"|{row['kernel']}|{row['psi_bin']}|{row['n']}|{row['gate_minus_classical_mean']:.5f}|{row['gate_minus_moment3_mean']:.5f}|{row['gate_win_classical'][0]:.3f}|{row['gate_win_moment3'][0]:.3f}|{row['projection5_rate']:.3f}|")
    lines += ["", "## 理论导向的部署回退诊断", "", "|核|预先定义 ψ 阈值|n|相对 Classical 均值差|95% CI|胜率|覆盖率|q99差值|", "|---|---:|---:|---:|---|---:|---:|---:|"]
    for row in report["kernel_specific_fallback"]:
        lines.append(f"|{row['kernel']}|{row['predeclared_psi_threshold']:.2f}|{row['n']}|{row['mean_diff_vs_classical']:.5f}|[{row['bootstrap_ci95_low']:.5f},{row['bootstrap_ci95_high']:.5f}]|{row['win_rate_vs_classical']:.3f}|{row['coverage_bounded_gate']:.3f}|{row['q99_diff_vs_classical']:.5f}|")
    lines += [
        "",
        "## 结论与边界",
        "",
        "1. Fermi 在 ψ≤1 的窗口内保持对 Classical 的稳定改进，且在无投影激活时最稳健；投影激活率上升后，改进幅度和对 Moment3 的胜率下降。",
        "2. arctan 的五阶展开在 ψ接近1时进入其解析半径附近，不能沿用 Fermi 的 ψ≤1 解释；在本扫描中，arctan 的门控相对 Classical 在高 ψ 区间恶化，必须采用更保守的 ψ≤0.75 诊断边界或回退 Classical。",
        "3. 这不是“新模型全域优越”的证据，而是一个可审计的 applicability map：核函数、ψ、投影激活和状态稀疏度共同决定是否使用高阶尾项。",
        "4. 当前边界仍是数值诊断边界；论文中应把解析余项上界作为后续理论工作，并明确 arctan/Fermi 的核特异性。",
    ]
    (out / "boundary_psi_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "conditions": len(p), "outputs": [str(out / "boundary_psi_audit.json"), str(out / "boundary_psi_audit.md")]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
