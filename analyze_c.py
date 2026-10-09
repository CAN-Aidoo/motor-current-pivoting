import json, glob, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = sum([json.load(open(f)) for f in sorted(glob.glob("results/expC_part*.json"))], [])
rows.sort(key=lambda r: r["i"])
M = ["open_nominal", "open_oracle", "closed_oracle", "closed_current"]
LBL = {"open_nominal": "Open loop, nominal object", "open_oracle": "Open loop, true object parameters",
       "closed_oracle": "Closed loop, true angle (bound)", "closed_current": "Closed loop, motor current only"}
res = {"n": len(rows)}
for m in M:
    e = np.array([r[m]["final"] - r["target"] for r in rows]); held = np.array([r[m]["held"] for r in rows])
    er = np.array([r[m]["rod"] - r["target"] for r in rows])
    d = dict(rms=float(np.sqrt(np.mean(e[held] ** 2))), median_abs=float(np.median(np.abs(e[held]))),
             p95_abs=float(np.percentile(np.abs(e[held]), 95)), mean_signed=float(np.mean(e[held])),
             within_0p05=float(np.mean(np.abs(e) <= 0.05)), within_0p1=float(np.mean(np.abs(e) <= 0.1)),
             dropped=int((~held).sum()), rod_angle_rms=float(np.sqrt(np.mean(er[held] ** 2))))
    if m.startswith("closed"):
        d["probes_mean"] = float(np.mean([r[m]["probes"] for r in rows]))
        d["probes_max"] = int(np.max([r[m]["probes"] for r in rows]))
        d["t_sim_mean_s"] = float(np.mean([r[m]["t_sim"] for r in rows]))
    if m == "closed_current":
        se = np.array([r[m]["self_est"] - r[m]["final"] for r in rows])
        d["self_estimate_rms"] = float(np.sqrt(np.mean(se ** 2)))
    res[m] = d
json.dump(res, open("results/expC_results.json", "w"), indent=1)
print(json.dumps(res, indent=1))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": "#8a8a86", "axes.labelcolor": "#3b3b38",
                     "xtick.color": "#5c5c58", "ytick.color": "#5c5c58", "axes.spines.top": False, "axes.spines.right": False})
col = {"open_nominal": "#eda100", "open_oracle": "#eb6834", "closed_oracle": "#1baf7a", "closed_current": "#2a78d6"}
fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=200)
for m in M:
    e = np.sort(np.abs([r[m]["final"] - r["target"] for r in rows]))
    y = np.arange(1, len(e) + 1) / len(e)
    ax.step(np.maximum(e, 1e-3), y, where="post", color=col[m], lw=2, label=LBL[m])
ax.axvline(0.05, color="#c9c8c3", lw=1, ls=":"); ax.text(0.05, 1.04, "0.05 rad (2.9°)", fontsize=7.5, color="#76766f", ha="center", transform=ax.get_xaxis_transform())
ax.set_xscale("log"); ax.set_xlim(1e-3, 2.0); ax.set_ylim(0, 1.02)
ax.set_xlabel("Final angle error |θ − θ*| (rad, log scale)"); ax.set_ylabel("Fraction of trials")
ax.grid(color="#ebeae6", lw=0.8, which="major"); ax.legend(frameon=False, fontsize=7.8, loc="upper left")
fig.tight_layout(); fig.savefig("figures/fig_closed_loop.png")
