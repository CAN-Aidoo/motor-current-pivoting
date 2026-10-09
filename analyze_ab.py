"""Analysis for Exp A (estimator accuracy, ideal sensing) and Exp B (sensing budget).
Writes mj_results.json. Every number in the write-up comes from here."""
import json, sys, numpy as np
from motor_model import DEFAULT, current_trace, static_current, sweep_ls, predicted_sd, naive_static

rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else None
pct = lambda x, q: float(np.percentile(x, q)) if len(x) else None
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
COLS = "i m d mu mut F Fstar held rod comdir comr tau0 taurest thetarest slip".split()


def load(amp):
    z = np.load(f"results/mjA_amp{amp}.npz")
    R = {c: z["rows"][:, k] for k, c in enumerate(COLS)}
    return R, z["traces"].astype(float)


def ideal_fit(T, P):
    psi, dpsi, ddcmd, tau, _ = T
    th, c, rs, X = sweep_ls(psi, dpsi, ddcmd, tau, P, friction=True, comp_inertia=False)
    return th, c, rs


def exp_a(amps):
    P = dict(DEFAULT); P.update(J_r=0.0, fs=1000.0)
    out = {}
    for amp in amps:
        R, TR = load(amp)
        held = R["held"] > 0
        est = np.array([ideal_fit(T, P)[0] for T in TR])
        A = np.array([np.hypot(*ideal_fit(T, P)[1][:2]) for T in TR])
        err = np.abs(wrap(est - R["comdir"]))
        naive = -np.arccos(np.clip(R["taurest"] / R["tau0"], -1, 1))
        # naive is read at rest before re-clamp; compare it with the rod angle at rest
        naive_err = np.abs(wrap(naive - R["thetarest"]))
        gap = np.abs(wrap(R["comdir"] - R["rod"]))
        piv = held & (np.abs(R["rod"]) > 0.1)
        s = np.abs(np.sin(R["comdir"]))
        o = dict(n=int(len(est)), held=int(held.sum()), pivoted=int(piv.sum()),
                 ls_err_rms=rms(err[held]), ls_err_median=pct(err[held], 50), ls_err_p95=pct(err[held], 95),
                 ls_err_rms_pivoted=rms(err[piv]),
                 ls_vs_rod_rms=rms(np.abs(wrap(est - R['rod']))[held]), ls_vs_rod_rms_pivoted=rms(np.abs(wrap(est - R['rod']))[piv]),
                 ls_err_rms_near_horizontal=rms(err[held & (s < 0.3)]), n_near_horizontal=int((held & (s < 0.3)).sum()),
                 ls_err_rms_tilted=rms(err[held & (s >= 0.3)]),
                 naive_err_rms=rms(naive_err[held]), naive_err_rms_pivoted=rms(naive_err[piv]),
                 comdir_vs_rod_gap_rms=rms(gap[held]), comdir_vs_rod_gap_p95=pct(gap[held], 95),
                 comr_over_d_p5_p95=[pct((R["comr"] / R["d"])[held], 5), pct((R["comr"] / R["d"])[held], 95)])
        out[str(amp)] = o
    return out


def exp_b(amp, draws=4):
    R, TR = load(amp)
    held = np.where(R["held"] > 0)[0]
    P0 = dict(DEFAULT); P0.update(J_r=0.0)
    ideal = {k: ideal_fit(TR[k], P0)[0] for k in held}
    grid = []
    bits_list = [6, 8, 10, 12, 14]; noise_list = [0.0, 0.001, 0.005, 0.02]; fric_list = [0.0, 0.01, 0.03, 0.1]
    rng = np.random.default_rng(7)
    for bits in bits_list:
        for sn in noise_list:
            for tc in fric_list:
                P = dict(DEFAULT); P.update(bits=bits, sigma_n=sn, tau_c=tc, b_visc=0.002)
                lsb = 2 * P["I_fs"] / 2 ** bits
                sig_eff = np.sqrt(sn ** 2 + lsb ** 2 / 12)
                e_chain, e_total, e_naive, pred = [], [], [], []
                for k in held:
                    psi, dpsi, ddcmd, tau, ddact = TR[k]
                    for _ in range(draws):
                        ps, dps, dd, im = current_trace(psi, dpsi, ddact, tau, P, rng)
                        _, _, ddc, _ = current_trace(psi, dpsi, ddcmd, tau, P, rng)
                        th, c, rs, X = sweep_ls(ps, dps, ddc, im, P, friction=True, comp_inertia=True)
                        e_chain.append(wrap(th - ideal[k])); e_total.append(wrap(th - R["comdir"][k]))
                        i0 = static_current(R["tau0"][k], P, rng); ir = static_current(R["taurest"][k], P, rng)
                        e_naive.append(wrap(naive_static(i0, ir) - R["thetarest"][k]))
                    pred.append(predicted_sd(X, c, sig_eff))
                grid.append(dict(bits=bits, sigma_n_mA=sn * 1e3, tau_c=tc, lsb_mA=lsb * 1e3,
                                 chain_rms=rms(e_chain), total_rms=rms(e_total), total_p95=pct(np.abs(e_total), 95),
                                 naive_rms=rms(e_naive), predicted_rms=rms(np.array(pred)),
                                 dithered=bool(sn >= lsb / 2)))
        print("bits", bits, "done", flush=True)
    return grid


if __name__ == "__main__":
    amps = [a for a in ["0.075", "0.15", "0.3"]]
    res = dict(expA=exp_a(amps))
    print(json.dumps(res["expA"], indent=1), flush=True)
    res["expB_amp"] = sys.argv[1] if len(sys.argv) > 1 else "0.15"
    res["expB"] = exp_b(res["expB_amp"])
    json.dump(res, open("results/mj_results.json", "w"), indent=1)
    print("saved")
