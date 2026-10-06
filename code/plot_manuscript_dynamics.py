"""Reproduce Figure 3 from the defining game, with integration checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'figures').mkdir(parents=True,exist_ok=True)
(ROOT/'results/revision_checks').mkdir(parents=True,exist_ok=True)
case=json.loads((ROOT/'configs/extended_full.json').read_text())['constructive_case']
A=np.array(case['payoff']);beta=case['beta'];x0=np.array(case['initial_state'])
Q=2/beta*np.tanh(beta*(A[:,None,:,None]-A[None,:,None,:])/2)
def rhs(kind,x):
 if kind=='classical': return x*(A@x-x@A@x)
 if kind=='full_feedback':
  pay=A@x; pair=2/beta*np.tanh(beta*(pay[:,None]-pay[None,:])/2)
 else: pair=np.einsum('l,ijkl,k->ij',x,Q,x)
 return x*(pair@x)
xc=np.linalg.solve(np.vstack([A[:2]-A[2],np.ones(3)]),[0,0,1])
r=root(lambda y:rhs('finite_feedback',np.r_[y,1-sum(y)])[:2],xc[:2],tol=1e-11)
xe=np.r_[r.x,1-sum(r.x)];assert max(abs(rhs('finite_feedback',xe)))<1e-12
names=['classical','full_feedback','finite_feedback'];ts=np.linspace(0,40,801)
rows=[];paths={};checks={}
old=pd.read_csv(ROOT/'results/extended_full_e3_trajectories.csv')
for name in names:
 def integrate(tol):
  sol=solve_ivp(lambda t,x:rhs(name,x),(0,40),x0,method='DOP853',rtol=tol,atol=tol*.01,t_eval=ts)
  assert sol.success;return sol.y.T
 values=integrate(1e-10);fine=integrate(1e-12)
 error=float(np.max(abs(values-fine)));assert error<1e-8
 assert np.min(values)>0 and np.max(abs(values.sum(axis=1)-1))<1e-12
 checks[name]={'tolerance_comparison':error};paths[name]=fine
 if name!='full_feedback':
  ref=old[old.model==('exact' if name=='finite_feedback' else 'classical')].pivot(index='time',columns='strategy',values='share').to_numpy()
  diff=float(np.max(abs(ref-fine))); assert diff<1e-8
  checks[name]['archived_RK4_difference']=diff
 for t,x in zip(ts,fine):rows.append(dict(model=name,time=t,x1=x[0],x2=x[1],x3=x[2]))
pd.DataFrame(rows).to_csv(ROOT/'results/revision_checks/three_strategy_trajectories.csv',index=False)
(ROOT/'results/revision_checks/trajectory_verification.json').write_text(json.dumps(checks,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':10,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(7.5,3.65),gridspec_kw={'wspace':.30})
styles=[('#68717D','--','Classical'),('#237A96','-.','Matched full feedback'),('#9C2944','-','Finite feedback')]
for name,(color,style,label) in zip(names,styles):
 x=paths[name];axs[0].plot(x[:,0],x[:,2],color=color,ls=style,lw=1.25,label=label)
 axs[1].plot(ts,x[:,0],color=color,ls=style,lw=1.25)
 if name!='classical':
  eq=xe if name=='finite_feedback' else xc
  axs[0].scatter(eq[0],eq[2],s=30,color=color,edgecolor='white',lw=.6,zorder=6)
 for start,end in [(80,92)]:axs[0].annotate('',xy=x[end,[0,2]],xytext=x[start,[0,2]],arrowprops=dict(arrowstyle='->',color=color,lw=1.0))
axs[0].scatter(x0[0],x0[2],marker='D',s=24,color='#26313D',zorder=7)
axs[0].set(xlabel=r'Strategy share $x_1$',ylabel=r'Strategy share $x_3$');axs[0].set_title('(a) Trajectories in the simplex',loc='left')
axs[1].set(xlabel=r'Normalized time $\tau$',ylabel=r'Strategy share $x_1$',xlim=(0,40));axs[1].set_title('(b) Share evolution',loc='left')
for ax in axs:ax.grid(color='#E8EBEE',lw=.6);ax.set_axisbelow(True)
h,l=axs[0].get_legend_handles_labels()
h.extend([Line2D([],[],color='#26313D',marker='o',ls='',markersize=4),Line2D([],[],color='#26313D',marker='D',ls='',markersize=4)])
l.extend(['Stationary point','Initial state'])
fig.legend(h,l,loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.53,.0),columnspacing=1.3)
fig.subplots_adjust(left=.09,right=.99,top=.89,bottom=.28)
fig.savefig(ROOT/'figures/three_strategy_dynamics.pdf',bbox_inches='tight')
print(json.dumps(checks))
