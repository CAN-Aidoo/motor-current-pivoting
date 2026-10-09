import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=json.load(open('results/mj_results.json')); B=R['expB']; floor=R['expA']['0.15']['ls_err_rms']
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,"axes.edgecolor":"#8a8a86","axes.labelcolor":"#3b3b38",
 "xtick.color":"#5c5c58","ytick.color":"#5c5c58","axes.spines.top":False,"axes.spines.right":False})
def get(**kw): return [g for g in B if all(abs(g[k]-v)<1e-9 for k,v in kw.items())]
# Fig 3: error vs bits for noise levels (tau_c = 0.03)
fig,ax=plt.subplots(figsize=(6.0,3.5),dpi=200)
cols={0.0:"#86b6ef",1.0:"#3987e5",5.0:"#1c5cab",20.0:"#0d366b"}
for sn,c in cols.items():
    gs=sorted(get(sigma_n_mA=sn,tau_c=0.03),key=lambda g:g['bits'])
    x=[g['bits'] for g in gs]; y=[g['chain_rms'] for g in gs]
    ax.plot(x,y,color=c,lw=2,marker="o",ms=5,mec="white",mew=1,label=f"{sn:g} mA")
    ax.text(14.25,y[-1],f"{sn:g} mA",va="center",fontsize=8,color="#3b3b38")
ax.axhline(floor,color="#9a9994",ls="--",lw=1); ax.text(6.05,floor*1.12,"physics floor (ideal sensing, Exp A)",fontsize=7.5,color="#76766f")
ax.set_yscale("log"); ax.set_xticks([6,8,10,12,14]); ax.set_xlim(5.7,15.3)
ax.set_xlabel("Current ADC resolution (bits, ±2 A full scale)"); ax.set_ylabel("Angle error from sensing chain\n(RMS, rad, log scale)")
ax.grid(axis="y",color="#ebeae6",lw=0.8,which="both"); ax.legend(title="Current noise σ",frameon=False,fontsize=7.5,title_fontsize=7.5,ncol=4,loc="upper right")
fig.tight_layout(); fig.savefig("figures/fig_bits.png")
# Fig 4: friction robustness, sweep vs naive at 12-bit, 1 mA
fig,ax=plt.subplots(figsize=(5.2,3.3),dpi=200)
gs=sorted(get(bits=12,sigma_n_mA=1.0),key=lambda g:g['tau_c'])
x=[g['tau_c'] for g in gs]; xs=[0.003 if v==0 else v for v in x]
ax.plot(xs,[g['total_rms'] for g in gs],color="#2a78d6",lw=2,marker="o",ms=5,mec="white",mew=1,label="Wrist sweep + least squares")
ax.plot(xs,[g['naive_rms'] for g in gs],color="#eb6834",lw=2,marker="s",ms=5,mec="white",mew=1,label="Single static reading")
ax.axvline(0.0392*1,color="#c9c8c3",lw=1,ls=":"); ax.text(0.042,0.012,"≈ typical gravity\ntorque m·g·d",fontsize=7.5,color="#76766f")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels(["0","0.01","0.03","0.1"])
ax.set_xlabel("Gearbox Coulomb friction at the output (N·m)"); ax.set_ylabel("Total angle error (RMS, rad)")
ax.grid(axis="y",color="#ebeae6",lw=0.8,which="both"); ax.legend(frameon=False,fontsize=8,loc="center left",bbox_to_anchor=(0.0,0.62))
fig.tight_layout(); fig.savefig("figures/fig_friction.png")
# Fig 5: predicted vs measured chain error
fig,ax=plt.subplots(figsize=(4.4,3.8),dpi=200)
d=[g for g in B if g['dithered']]; u=[g for g in B if not g['dithered']]
ax.plot([1e-4,1],[1e-4,1],color="#c9c8c3",ls="--",lw=1)
ax.scatter([g['predicted_rms'] for g in d],[g['chain_rms'] for g in d],s=26,color="#2a78d6",edgecolor="white",lw=1,label="noise ≥ LSB/2 (dithered)",zorder=3)
ax.scatter([g['predicted_rms'] for g in u],[g['chain_rms'] for g in u],s=26,color="#eb6834",marker="s",edgecolor="white",lw=1,label="noise < LSB/2",zorder=3)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(5e-5,0.5); ax.set_ylim(2e-4,1.5)
ax.set_xlabel("Predicted error, linearized LS (rad)"); ax.set_ylabel("Measured error (rad)")
ax.text(1.2e-4,1.6e-4*1.0+0,"",fontsize=1); ax.text(0.02,0.012,"prediction = measurement",rotation=37,fontsize=7.5,color="#76766f")
ax.grid(color="#ebeae6",lw=0.8,which="major"); ax.legend(frameon=False,fontsize=7.5,loc="upper left")
fig.tight_layout(); fig.savefig("figures/fig_predicted.png")
# agreement stats for dithered
r=np.array([g['chain_rms']/g['predicted_rms'] for g in d]); print("dithered ratio measured/pred: median %.2f, range %.2f-%.2f, n=%d"%(np.median(r),r.min(),r.max(),len(r)))
ru=np.array([g['chain_rms']/g['predicted_rms'] for g in u]); print("undithered ratio median %.1f, range %.1f-%.1f n=%d"%(np.median(ru),ru.min(),ru.max(),len(ru)))
