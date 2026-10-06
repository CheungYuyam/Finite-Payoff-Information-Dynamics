"""Pre-registered realistic-regime sensitivity scan for v3."""
from __future__ import annotations
import argparse, csv, gc, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import automatica_model as model

def load_config(path):
    c=json.loads(path.read_text(encoding="utf-8")); s=json.dumps(c,ensure_ascii=False,sort_keys=True,separators=(",",":"))
    return c, hashlib.sha256(s.encode()).hexdigest()[:16]
def atomic_csv(path, fields, rows):
    tmp=path.with_suffix(path.suffix+".tmp"); n=0
    with tmp.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow(r); n+=1
            if n%250==0: h.flush()
    tmp.replace(path); return n
def psi_bin(v, edges):
    for a,b in zip(edges[:-1],edges[1:]):
        if a<=v<b: return f"[{a:g},{b:g})"
    return f"[{edges[-2]:g},{edges[-1]:g})"
def mlabel(m): return "inf" if m is None else str(int(m))
def psi_values(A,x,beta,m,mode):
    vals=[]; vars_=[]; ws=[]; k=len(x)
    for i in range(k-1):
        for j in range(i+1,k):
            mu,v,_=model.payoff_difference_moments(A,x,i,j,m,m,sampling_mode=mode)
            vals.append(max(float(mu*mu+v),0.0)); vars_.append(max(float(v),0.0)); ws.append(float(x[i]*x[j]))
    w=np.asarray(ws); w/=max(float(w.sum()),1e-15)
    return beta*math.sqrt(float(w@np.asarray(vals))), beta*math.sqrt(float(w@np.asarray(vars_)))
def rel(p,e): return float(np.linalg.norm(p-e)/max(np.linalg.norm(e),1e-14))
def sym(p,e): return float(2*np.linalg.norm(p-e)/max(np.linalg.norm(p)+np.linalg.norm(e),1e-14))
def generate(c,h):
    rng=np.random.default_rng(int(c["seed"])); case=0; edges=list(map(float,c["psi_bins"]))
    gate=float(c["gated_rule"]["psi_max_for_moment5"]); main=float(c["main_work_region"]["psi_max"])
    rho_gate=float(c.get("rho_gated_rule", {}).get("rho5_max", 0.1))
    include_bounded=bool(c.get("include_bounded_models", False))
    include_bounded_increment_gate=bool(c.get("include_bounded_increment_gate", False))
    if include_bounded_increment_gate and not include_bounded:
        raise ValueError("include_bounded_increment_gate requires include_bounded_models")
    for k in map(int,c["dimensions"]):
      for scale in map(float,c["scales"]):
       for _ in range(int(c["games_per_dimension_per_scale"])):
        A=rng.normal(size=(k,k)); np.fill_diagonal(A,0); A-=A.mean(); A=A/max(float(A.std()),1e-12)*scale
        x=rng.dirichlet(np.full(k,float(c["dirichlet_alpha"]))); gh=hashlib.sha256(A.tobytes()).hexdigest()[:12]
        for beta in map(float,c["betas"]):
         for mm in c["sample_sizes"]:
          m=None if mm is None else int(mm)
          for kernel in c["kernels"]:
           for mode in c["sampling_modes"]:
            t=time.perf_counter(); exact=model.exact_rhs(x,A,beta,m,kernel=kernel,sampling_mode=mode); sec=time.perf_counter()-t
            classical=model.classical_rhs(x,A)
            variance=model.tail_rhs(x,A,beta,m,include_skewness=False,kernel=kernel,sampling_mode=mode)
            m3=model.moment_tail_rhs(x,A,beta,m,order=3,kernel=kernel,sampling_mode=mode)
            m5=model.moment_tail_rhs(x,A,beta,m,order=5,kernel=kernel,sampling_mode=mode)
            if include_bounded:
                b3=model.bounded_moment_tail_rhs(x,A,beta,m,order=3,kernel=kernel,sampling_mode=mode)
                b5=model.bounded_moment_tail_rhs(x,A,beta,m,order=5,kernel=kernel,sampling_mode=mode)
            else:
                b3=b5=None
            psi,chi=psi_values(A,x,beta,m,mode); fifth_norm=float(np.linalg.norm(m5-m3)); m3_norm=float(np.linalg.norm(m3)); rho5=fifth_norm/max(m3_norm,1e-14)
            bounded_fifth_norm=float(np.linalg.norm(b5-b3)) if include_bounded else float("nan")
            bounded_m3_norm=float(np.linalg.norm(b3)) if include_bounded else float("nan")
            bounded_rho5=bounded_fifth_norm/max(bounded_m3_norm,1e-14) if include_bounded else float("nan")
            gated=m5 if psi<=gate else m3; gated_rho=m5 if (psi<=gate and rho5<=rho_gate) else m3
            preds={"classical":classical,"variance_tail":variance,"moment3_tail":m3,"moment5_tail":m5,"gated_moment5_moment3":gated,"gated_moment5_moment3_rho":gated_rho}
            if include_bounded:
                preds.update({"bounded_moment3_tail":b3,"bounded_moment5_tail":b5,"bounded_rho_gate":b5 if (psi<=gate and rho5<=rho_gate) else b3})
                if include_bounded_increment_gate:
                    preds["bounded_increment_rho_gate"]=b5 if (psi<=gate and bounded_rho5<=rho_gate) else b3
            base={"schema_version":c["schema_version"],"config_hash":h,"case_id":case,"game_hash":gh,"dimension":k,"scale":scale,"beta":beta,"m":mlabel(m),"kernel":kernel,"sampling_mode":mode,"psi":psi,"chi":chi,"psi_bin":psi_bin(psi,edges),"main_work_region":int(psi<=main),"exact_norm":float(np.linalg.norm(exact)),"exact_seconds":sec,"moment3_norm":m3_norm,"fifth_increment_norm":fifth_norm,"fifth_relative_to_moment3":rho5,"bounded_moment3_norm":bounded_m3_norm,"bounded_fifth_increment_norm":bounded_fifth_norm,"bounded_fifth_relative_to_moment3":bounded_rho5,"gate_used":"moment5" if psi<=gate else "moment3","rho_gate_used":"moment5" if (psi<=gate and rho5<=rho_gate) else "moment3","bounded_rho_gate_used":"moment5" if (psi<=gate and bounded_rho5<=rho_gate) else "moment3"}
            for name,p in preds.items():
                yield {**base,"model":name,"relative_error":rel(p,exact),"symmetric_relative_error":sym(p,exact),"l2_error":float(np.linalg.norm(p-exact)),"mass_error":float(abs(float(p.sum())))}
            del exact,classical,variance,m3,m5,gated,preds,b3,b5; gc.collect()
        case+=1
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",type=Path,required=True); a=ap.parse_args(); c,h=load_config(a.config.resolve())
    out=ROOT/"results"/"realistic_sensitivity"; out.mkdir(parents=True,exist_ok=True)
    fields=["schema_version","config_hash","case_id","game_hash","dimension","scale","beta","m","kernel","sampling_mode","psi","chi","psi_bin","main_work_region","exact_norm","exact_seconds","moment3_norm","fifth_increment_norm","fifth_relative_to_moment3","bounded_moment3_norm","bounded_fifth_increment_norm","bounded_fifth_relative_to_moment3","gate_used","rho_gate_used","bounded_rho_gate_used","model","relative_error","symmetric_relative_error","l2_error","mass_error"]
    t=time.perf_counter(); n=atomic_csv(out/f"{c['profile']}_rows.csv",fields,generate(c,h)); manifest={"schema_version":c["schema_version"],"profile":c["profile"],"config_hash":h,"config_path":str(a.config.resolve()),"rows":n,"elapsed_seconds":time.perf_counter()-t,"declared_work_guard":"all configured conditions evaluated"}; (out/f"{c['profile']}_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
