"""Independent bootstrap audit for the realistic-regime sensitivity scan."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def bootstrap(values, statistic, seed, reps=3000):
    values=np.asarray(values,dtype=float); rng=np.random.default_rng(seed)
    if len(values)==0: return [float("nan")]*2
    out=np.empty(reps)
    for r in range(reps): out[r]=statistic(values[rng.integers(0,len(values),len(values))])
    return [float(np.quantile(out,.025)),float(np.quantile(out,.975))]
def wilson(success,n):
    if n==0:return [float("nan"),float("nan")]
    z=1.959963984540054; p=success/n; den=1+z*z/n; cen=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0,cen-half)),float(min(1,cen+half))]
def record(frame, proposed, comparator, seed):
    p=frame[proposed].to_numpy(float); q=frame[comparator].to_numpy(float); d=p-q
    return {"n":int(len(d)),"mean_delta":float(np.mean(d)),"median_delta":float(np.median(d)),"q90_delta":float(np.quantile(d,.90)),"q99_delta":float(np.quantile(d,.99)),"win_rate":float(np.mean(d<0)),"mean_bootstrap_95":bootstrap(d,np.mean,seed),"median_bootstrap_95":bootstrap(d,np.median,seed+1),"win_rate_wilson_95":wilson(int(np.sum(d<0)),len(d)),"overshoot_rate":float(np.mean(d>0))}
def grouped_records(df, group_cols, proposed, comparator, seed0):
    out={}
    for idx,g in df.groupby(group_cols,dropna=False):
        if not isinstance(idx,tuple): idx=(idx,)
        key="|".join(f"{c}={v}" for c,v in zip(group_cols,idx))
        out[key]=record(g.dropna(subset=[proposed, comparator]),proposed,comparator,seed0+len(out)*17)
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv",type=Path,required=True); ap.add_argument("--config",type=Path,required=True); a=ap.parse_args()
    cfg=json.loads(a.config.read_text(encoding="utf-8")); df=pd.read_csv(a.csv)
    pivot=df.pivot_table(index=["case_id","dimension","scale","beta","m","kernel","sampling_mode","psi","psi_bin","main_work_region"],columns="model",values="relative_error",aggfunc="first").reset_index()
    meta_keys=["case_id","dimension","scale","beta","m","kernel","sampling_mode","psi","psi_bin","main_work_region"]
    if "fifth_relative_to_moment3" in df.columns:
        meta_cols=meta_keys+["fifth_relative_to_moment3"]
        meta=df[df.model=="moment5_tail"][meta_cols].drop_duplicates()
        pivot=pivot.merge(meta,on=meta_keys,how="left")
    required_models = ["classical","variance_tail","moment3_tail","moment5_tail","gated_moment5_moment3"]
    if "bounded_moment3_tail" in pivot.columns:
        required_models += ["bounded_moment3_tail", "bounded_moment5_tail", "bounded_rho_gate"]
    if "bounded_increment_rho_gate" in pivot.columns:
        required_models += ["bounded_increment_rho_gate"]
    for col in required_models:
        if col not in pivot: raise ValueError(f"missing model {col}")
    report={"schema_version":"v3-realistic-sensitivity-audit-1","config_path":str(a.config.resolve()),"rows_input":int(len(df)),"cases_conditions":int(len(pivot)),"pre_registered_rule":cfg["gated_rule"],"main_work_region":cfg["main_work_region"],"quality":{"max_mass_error":float(df.mass_error.abs().max()),"max_exact_seconds":float(df.exact_seconds.max()),"skipped_condition_note":"none"},"comparisons":{}}
    for name,p,c in [("moment3_vs_classical","moment3_tail","classical"),("moment5_vs_moment3","moment5_tail","moment3_tail"),("gated_vs_classical","gated_moment5_moment3","classical"),("gated_vs_moment5","gated_moment5_moment3","moment5_tail")]:
        report["comparisons"][name]={"overall":record(pivot,p,c,2026090401+len(name)),"main_work_region":record(pivot[pivot.main_work_region==1],p,c,2026090501+len(name)),"outside_main_work_region":record(pivot[pivot.main_work_region==0],p,c,2026090601+len(name))}
    if "gated_moment5_moment3_rho" in pivot:
        for name,p,c in [("rho_gated_vs_classical","gated_moment5_moment3_rho","classical"),("rho_gated_vs_moment3","gated_moment5_moment3_rho","moment3_tail"),("rho_gated_vs_moment5","gated_moment5_moment3_rho","moment5_tail")]:
            report["comparisons"][name]={"overall":record(pivot,p,c,2026070401+len(name)),"main_work_region":record(pivot[pivot.main_work_region==1],p,c,2026070501+len(name)),"outside_main_work_region":record(pivot[pivot.main_work_region==0],p,c,2026070601+len(name))}
    if "bounded_moment3_tail" in pivot:
        for name,p,c in [
            ("bounded_moment3_vs_classical","bounded_moment3_tail","classical"),
            ("bounded_moment3_vs_moment3","bounded_moment3_tail","moment3_tail"),
            ("bounded_moment5_vs_moment3","bounded_moment5_tail","moment3_tail"),
            ("bounded_moment5_vs_bounded_moment3","bounded_moment5_tail","bounded_moment3_tail"),
            ("bounded_rho_gate_vs_classical","bounded_rho_gate","classical"),
            ("bounded_rho_gate_vs_moment3","bounded_rho_gate","moment3_tail"),
            ("bounded_rho_gate_vs_bounded_moment3","bounded_rho_gate","bounded_moment3_tail"),
        ]:
            report["comparisons"][name]={"overall":record(pivot,p,c,2026080401+len(name)),"main_work_region":record(pivot[pivot.main_work_region==1],p,c,2026080501+len(name)),"outside_main_work_region":record(pivot[pivot.main_work_region==0],p,c,2026080601+len(name))}
    if "bounded_increment_rho_gate" in pivot:
        for name,p,c in [
            ("bounded_increment_gate_vs_bounded_rho_gate","bounded_increment_rho_gate","bounded_rho_gate"),
            ("bounded_increment_gate_vs_classical","bounded_increment_rho_gate","classical"),
            ("bounded_increment_gate_vs_moment3","bounded_increment_rho_gate","moment3_tail"),
        ]:
            report["comparisons"][name]={"overall":record(pivot,p,c,2026080701+len(name)),"main_work_region":record(pivot[pivot.main_work_region==1],p,c,2026080801+len(name)),"outside_main_work_region":record(pivot[pivot.main_work_region==0],p,c,2026080901+len(name))}
    report["by_psi_bin"]={"moment3_vs_classical":grouped_records(pivot,["psi_bin"],"moment3_tail","classical",20260910),"moment5_vs_moment3":grouped_records(pivot,["psi_bin"],"moment5_tail","moment3_tail",20260920),"gated_vs_classical":grouped_records(pivot,["psi_bin"],"gated_moment5_moment3","classical",20260930)}
    if "bounded_moment3_tail" in pivot:
        report["by_psi_bin"].update({"bounded_moment3_vs_classical":grouped_records(pivot,["psi_bin"],"bounded_moment3_tail","classical",20260970),"bounded_moment5_vs_moment3":grouped_records(pivot,["psi_bin"],"bounded_moment5_tail","moment3_tail",20260980),"bounded_rho_gate_vs_classical":grouped_records(pivot,["psi_bin"],"bounded_rho_gate","classical",20260990),"bounded_rho_gate_vs_moment3":grouped_records(pivot,["psi_bin"],"bounded_rho_gate","moment3_tail",20261000)})
    report["by_mechanism"]={"moment3_vs_classical":grouped_records(pivot,["kernel","sampling_mode"],"moment3_tail","classical",20260940),"moment5_vs_moment3":grouped_records(pivot,["kernel","sampling_mode"],"moment5_tail","moment3_tail",20260950),"gated_vs_classical":grouped_records(pivot,["kernel","sampling_mode"],"gated_moment5_moment3","classical",20260960)}
    report["by_scale"]={"moment3_vs_classical":grouped_records(pivot,["scale"],"moment3_tail","classical",20260970),"moment5_vs_moment3":grouped_records(pivot,["scale"],"moment5_tail","moment3_tail",20260980)}
    report["by_dimension"]={"moment3_vs_classical":grouped_records(pivot,["dimension"],"moment3_tail","classical",20260990),"moment5_vs_moment3":grouped_records(pivot,["dimension"],"moment5_tail","moment3_tail",20261000)}
    # A conservative, non-cherry-picked summary over the declared main region.
    main=pivot[pivot.main_work_region==1]
    summary={}
    for col in ["classical","variance_tail","moment3_tail","moment5_tail","gated_moment5_moment3"]:
        v=main[col].to_numpy(float); summary[col]={"n":int(len(v)),"median_relative_error":float(np.median(v)),"mean_relative_error":float(np.mean(v)),"q90_relative_error":float(np.quantile(v,.9)),"q99_relative_error":float(np.quantile(v,.99))}
    if "bounded_moment3_tail" in main:
        for col in ["bounded_moment3_tail","bounded_moment5_tail","bounded_rho_gate"]:
            v=main[col].to_numpy(float); summary[col]={"n":int(len(v)),"median_relative_error":float(np.median(v)),"mean_relative_error":float(np.mean(v)),"q90_relative_error":float(np.quantile(v,.9)),"q99_relative_error":float(np.quantile(v,.99))}
    report["main_region_error_summary"]=summary
    if "fifth_relative_to_moment3" in df:
        report["gate_diagnostics"]={"rho5_threshold":float(cfg.get("rho_gated_rule",{}).get("rho5_max",float("nan"))),"fraction_conditions_rho5_le_threshold":float((pivot["fifth_relative_to_moment3"]<=cfg.get("rho_gated_rule",{}).get("rho5_max",0.1)).mean()),"fraction_conditions_using_rho_gate_moment5":float(((pivot.psi<=cfg.get("rho_gated_rule",{}).get("psi_max_for_moment5",1.0))&(pivot.fifth_relative_to_moment3<=cfg.get("rho_gated_rule",{}).get("rho5_max",0.1))).mean())}
    if "gated_moment5_moment3_rho" in pivot:
        for col in ["gated_moment5_moment3_rho"]:
            v=pivot[pivot.main_work_region==1][col].to_numpy(float); report["main_region_error_summary"][col]={"n":int(len(v)),"median_relative_error":float(np.median(v)),"mean_relative_error":float(np.mean(v)),"q90_relative_error":float(np.quantile(v,.9)),"q99_relative_error":float(np.quantile(v,.99))}
    # Full threshold sensitivity is reported separately from the pre-specified
    # tau=0.1 rule.  This prevents selecting a threshold after inspecting the
    # Exact error while showing whether the conclusion is fragile.
    if "fifth_relative_to_moment3" in pivot:
        thresholds=[0.0,0.02,0.05,0.1,0.2,0.5,1.0]
        threshold_rows=[]
        for tau in thresholds:
            selected=(pivot.psi<=cfg.get("rho_gated_rule",{}).get("psi_max_for_moment5",1.0))&(pivot.fifth_relative_to_moment3<=tau)
            policy=np.where(selected,pivot.moment5_tail,pivot.moment3_tail)
            for region_name,mask in [("all",np.ones(len(pivot),dtype=bool)),("main_work_region",pivot.main_work_region.to_numpy()==1)]:
                d=policy[mask]-pivot.loc[mask,"classical"].to_numpy(float); d3=policy[mask]-pivot.loc[mask,"moment3_tail"].to_numpy(float)
                threshold_rows.append({"rho5_max":tau,"region":region_name,"n":int(mask.sum()),"mean_delta_vs_classical":float(np.mean(d)),"median_delta_vs_classical":float(np.median(d)),"win_rate_vs_classical":float(np.mean(d<0)),"q99_delta_vs_classical":float(np.quantile(d,.99)),"mean_delta_vs_moment3":float(np.mean(d3)),"win_rate_vs_moment3":float(np.mean(d3<0)),"fraction_using_moment5":float(selected[mask].mean())})
        report["rho5_threshold_sensitivity"]=threshold_rows
    out=ROOT/"results"/"realistic_sensitivity"; out.mkdir(parents=True,exist_ok=True)
    profile=str(cfg.get("profile", "realistic_sensitivity"))
    (out/f"{profile}_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# v3 现实工作区敏感性审计", "", f"- 配对条件数：{len(pivot)}；输入行数：{len(df)}", f"- 最大质量守恒误差：{df.mass_error.abs().max():.3e}", "", "## 总体与主工作区"]
    for n in report["comparisons"]:
        x=report["comparisons"][n]; lines.append(f"- {n}: overall median Δ={x['overall']['median_delta']:.6g}, win={x['overall']['win_rate']:.3f}; main median Δ={x['main_work_region']['median_delta']:.6g}, win={x['main_work_region']['win_rate']:.3f}")
    lines += ["", "## 解释边界", "", "主工作区由预先规定的 psi<=1.0 定义；门控规则只依据 psi，不使用 Exact 误差反馈。所有 psi>1.0 的强反馈结果保留并单独报告。"]
    (out/f"{profile}_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(report["comparisons"],ensure_ascii=False,indent=2))
if __name__=="__main__": main()
