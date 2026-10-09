"""Generate numbers.tex (macros) and table snippets from the experiment result files."""
import json, numpy as np
R = json.load(open("../results/mj_results.json")); C = json.load(open("../results/expC_results.json"))
A = R["expA"]["0.15"]; B = R["expB"]
f3 = lambda x: f"{x:.3f}"; f4 = lambda x: f"{x:.4f}"
mac = {}
def m(name, val): mac[name] = val
# Exp A
m("Aheld", str(A["held"])); m("An", str(A["n"])); m("Apiv", str(A["pivoted"]))
m("AlsRms", f3(A["ls_err_rms"])); m("AlsMed", f3(A["ls_err_median"])); m("AlsPnf", f3(A["ls_err_p95"]))
m("AlsHor", f3(A["ls_err_rms_near_horizontal"])); m("AlsTilt", f3(A["ls_err_rms_tilted"])); m("AnHor", str(A["n_near_horizontal"]))
m("AlsRod", f3(A["ls_vs_rod_rms"])); m("AnaiveRod", f3(A["naive_err_rms"])); m("Agap", f3(A["comdir_vs_rod_gap_rms"])); m("AgapPnf", f3(A["comdir_vs_rod_gap_p95"]))
m("ArOverDhi", f"{A['comr_over_d_p5_p95'][1]:.2f}")
amps = {k: R["expA"][k]["ls_err_rms"] for k in ("0.075", "0.15", "0.3")}
m("AampLo", f4(min(amps.values()))); m("AampHi", f4(max(amps.values())))
# Exp B
def g(**kw): return [x for x in B if all(abs(x[k] - v) < 1e-9 for k, v in kw.items())]
d = [x for x in B if x["dithered"]]; u = [x for x in B if not x["dithered"]]
rd = np.array([x["chain_rms"] / x["predicted_rms"] for x in d]); ru = np.array([x["chain_rms"] / x["predicted_rms"] for x in u])
m("Bnd", str(len(d))); m("Bnu", str(len(u))); m("BratLo", f"{rd.min():.2f}"); m("BratHi", f"{rd.max():.2f}")
m("BuMed", f"{np.median(ru):.1f}"); m("BuHi", f"{ru.max():.1f}"); m("Bconfigs", str(len(B)))
s12 = sorted(g(bits=12, sigma_n_mA=1.0), key=lambda x: x["tau_c"])
m("BsweepLo", f4(min(x["total_rms"] for x in s12))); m("BsweepHi", f4(max(x["total_rms"] for x in s12)))
m("BnaiveZero", f3(s12[0]["naive_rms"])); m("BnaiveMax", f"{s12[-1]['naive_rms']:.2f}")
e8 = {x["sigma_n_mA"]: x["chain_rms"] for x in g(bits=8, tau_c=0.0)}
m("BeightZero", f3(e8[0.0])); m("BeightFive", f3(e8[5.0])); m("BditherGain", f"{e8[0.0]/e8[5.0]:.0f}")
m("BchainTwelve", f4(g(bits=12, sigma_n_mA=1.0, tau_c=0.03)[0]["chain_rms"]))
# Exp C
for key, short in [("open_nominal", "On"), ("open_oracle", "Oo"), ("closed_oracle", "Co"), ("closed_current", "Cc")]:
    c = C[key]
    m(f"C{short}Rms", f3(c["rms"])); m(f"C{short}Med", f3(c["median_abs"])); m(f"C{short}Pnf", f3(c["p95_abs"]))
    m(f"C{short}Five", f"{100*c['within_0p05']:.1f}"); m(f"C{short}Ten", f"{100*c['within_0p1']:.1f}")
    m(f"C{short}Rod", f3(c["rod_angle_rms"])); m(f"C{short}Drop", str(c["dropped"]))
    if "probes_mean" in c:
        m(f"C{short}Probes", f"{c['probes_mean']:.1f}"); m(f"C{short}ProbesMax", str(c["probes_max"])); m(f"C{short}Tsim", f"{c['t_sim_mean_s']/60:.1f}")
m("CccSelf", f3(C["closed_current"]["self_estimate_rms"])); m("Cn", str(C["n"]))
with open("numbers.tex", "w") as fh:
    for k, v in mac.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
# Table: Exp B bits x noise at tau_c = 0.03
rows = []
for sn in (0.0, 1.0, 5.0, 20.0):
    vals = [g(bits=b, sigma_n_mA=sn, tau_c=0.03)[0]["chain_rms"] for b in (6, 8, 10, 12, 14)]
    rows.append(f"{sn:g} & " + " & ".join(f"{v:.4f}" for v in vals) + r" \\")
open("tab_bits.tex", "w").write("\n".join(rows) + "\n")
rows = []
for x in s12:
    rows.append(f"{x['tau_c']:g} & {x['total_rms']:.4f} & {x['naive_rms']:.3f}" + r" \\")
open("tab_fric.tex", "w").write("\n".join(rows) + "\n")
print(open("numbers.tex").read())
