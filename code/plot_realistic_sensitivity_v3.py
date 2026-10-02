"""Diagnostic plots for the v3 realistic-regime audit."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figure_style import PALETTE, setup_style

ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--audit",type=Path,required=True); a=ap.parse_args()
    setup_style()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42})
    r=json.loads(a.audit.read_text(encoding="utf-8")); out=ROOT/"figures"/"realistic_sensitivity"; out.mkdir(parents=True,exist_ok=True)
    prof=a.audit.stem.replace("_audit","")
    ts=pd.DataFrame(r.get("rho5_threshold_sensitivity",[]))
    if not ts.empty:
        main=ts[ts.region=="main_work_region"]
        fig,ax=plt.subplots(figsize=(7.2,4.5)); ax.plot(main.rho5_max,main.mean_delta_vs_classical,"o-",color=PALETTE["blue"],label="均值差：门控−Classical"); ax.plot(main.rho5_max,main.mean_delta_vs_moment3,"s-",color=PALETTE["green"],label="均值差：门控−Moment3"); ax.axhline(0,color=PALETTE["ink"],lw=.8); ax.axvline(.1,color=PALETTE["burgundy"],ls="--",lw=1,label="预注册阈值 0.1"); ax.set_xlabel(r"$\rho_5$ threshold"); ax.set_ylabel("paired relative-error difference"); ax.legend(frameon=False); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/f"{prof}_threshold_sensitivity.svg",bbox_inches="tight"); fig.savefig(out/f"{prof}_threshold_sensitivity.pdf",bbox_inches="tight"); fig.savefig(out/f"{prof}_threshold_sensitivity.png",dpi=300,bbox_inches="tight"); fig.savefig(out/f"{prof}_threshold_sensitivity.tiff",dpi=600,bbox_inches="tight"); plt.close(fig)
    bins=r.get("by_psi_bin",{}).get("moment3_vs_classical",{})
    if bins:
        labels=list(bins); vals=[bins[k]["median_delta"] for k in labels]; wins=[bins[k]["win_rate"] for k in labels]
        fig,ax1=plt.subplots(figsize=(8,4.5)); x=np.arange(len(labels)); ax1.bar(x,vals,color=PALETTE["blue"],alpha=.85); ax1.axhline(0,color=PALETTE["ink"],lw=.8); ax1.set_ylabel("median Δ (Moment3 − Classical)"); ax1.set_xticks(x,labels,rotation=35,ha="right"); ax2=ax1.twinx(); ax2.plot(x,wins,"o-",color=PALETTE["burgundy"],lw=2); ax2.axhline(.5,color=PALETTE["burgundy"],ls=":",lw=1); ax2.set_ylabel("paired win rate"); ax1.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(out/f"{prof}_psi_bins.svg",bbox_inches="tight"); fig.savefig(out/f"{prof}_psi_bins.pdf",bbox_inches="tight"); fig.savefig(out/f"{prof}_psi_bins.png",dpi=300,bbox_inches="tight"); fig.savefig(out/f"{prof}_psi_bins.tiff",dpi=600,bbox_inches="tight"); plt.close(fig)
    print(json.dumps({"profile":prof,"output_dir":str(out)},ensure_ascii=False))
if __name__=="__main__": main()
