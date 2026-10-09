"""Exp C: pivot a grasped rod to a target centre-of-mass angle.

All methods start from the same clamped, horizontal grasp and release grip slowly
(RATE N/s, quasi-static pivoting), then re-clamp with a smooth force ramp.
  open_nominal  : one release to F = A_nom*cos(target)/(2*mut_nom)   (nominal object)
  open_oracle   : same formula with the TRUE m, d, mu_t              (perfect object knowledge)
  closed_oracle : stepped loop fed the TRUE angle                    (sensing-free bound)
  closed_current: stepped loop fed the motor-current estimate from a full 10 s downhill probe,
                  through the motor + sensing chain (12-bit, 1 mA noise, 0.03 N.m Coulomb
                  friction, +/-10% motor-constant error)
Stepped loop: release -> clamp -> estimate. Grip search (x0.94) until motion is seen, then
steps covering 70% of the remaining angle using cos(theta) ~ c*F fitted at the last
equilibrium; stop within TOL of the target or once past it. The wrist is never moved while
the grip is low: an earlier rolling-window design that dithered the wrist during release
ratcheted the rod past the target (kept below as est_from/release for reference).
Scored on the true COM direction after the final clamp.
"""
import sys, json, numpy as np, mujoco
from mj_ctrl import Rig
from motor_model import DEFAULT, current_trace, sweep_ls

NOM = dict(m=0.125, d=0.04, mut=0.004); MUT_MIN = 0.002
SENS = dict(DEFAULT); SENS.update(bits=12, sigma_n=0.001, tau_c=0.03, b_visc=0.002)
KT_NOM = DEFAULT["N"] * DEFAULT["Kt"]
RATE = 0.3; AMP = 0.06; HZ = 0.25; WIN = 4.0; EVERY = 0.2; LATENCY = WIN / 2


def est_from(seg, rng):
    ps, dps, _, im = current_trace(seg["psi"], seg["dpsi"], seg["ddact"], seg["tau"], SENS, rng)
    th, c, _, _ = sweep_ls(ps, dps, seg["ddcmd"], im, SENS, friction=True, comp_inertia=True, skip=0.0)
    return th, np.hypot(c[0], c[1])


def initial_probe(rig, rng):
    psi, dpsi, ddcmd, tau, ddact = rig.probe()
    seg = dict(psi=psi, dpsi=dpsi, ddcmd=ddcmd, tau=tau, ddact=ddact)
    return est_from(seg, rng)


def release(rig, F_start, stop_rule, mode, rng, t_max=200.0):
    M, D = rig.model, rig.data; dt = M.opt.timestep; w = 2 * np.pi * HZ
    nwin = int(WIN / dt); nev = int(EVERY / dt)
    keys = ("psi", "dpsi", "ddcmd", "tau")
    buf = {k: np.zeros(int(t_max / dt) + 1) for k in keys}
    hist = []; k = 0; F = F_start
    dither = mode in ("oracle", "current")
    while k * dt < t_max:
        t = k * dt
        D.ctrl[0] = -AMP * (1 - np.cos(w * t)) if dither else 0.0
        F = max(0.3, F_start - RATE * t); D.ctrl[1:] = F
        mujoco.mj_step(M, D)
        buf["psi"][k] = D.qpos[rig.jid]; buf["dpsi"][k] = D.qvel[rig.vid]
        buf["ddcmd"][k] = -AMP * w * w * np.cos(w * t); buf["tau"][k] = D.actuator_force[0]
        k += 1
        if not rig.held(): break
        if mode == "open":
            if F <= stop_rule: break
        elif k >= nwin and k % nev == 0:
            if mode == "oracle":
                th = rig.com_dir() + float(D.qpos[rig.jid])   # true angle in the palm frame at psi = 0
            else:
                seg = {kk: buf[kk][k - nwin:k] for kk in keys}
                seg["ddact"] = np.gradient(seg["dpsi"], dt)
                th, _ = est_from(seg, rng)
            hist.append((t, th))
            # lead: extrapolate the recent trend over the estimator latency
            recent = [h for h in hist if h[0] >= t - 1.0]
            slope = np.polyfit([h[0] for h in recent], [h[1] for h in recent], 1)[0] if len(recent) >= 3 else 0.0
            lat = LATENCY if mode == "current" else 0.0
            if th + min(0.0, slope) * lat <= stop_rule:
                break
    rig.t_sim += k * dt; rig.grip = F
    D.ctrl[0] = 0.0; rig.clamp(); rig.step(0.5)
    return F


TOL = 0.03; SEARCH = 0.94; GAIN = 0.7; MAX_PROBES = 30


def stepped_loop(r, target, mode, rng, A_est):
    """Release slowly -> smooth clamp -> estimate -> decide. No wrist motion while the grip
    is low (a dither at the slip edge ratchets the rod), so all sensing happens clamped."""
    def measure():
        if mode == "oracle":
            return r.com_dir()
        return initial_probe(r, rng)[0]
    th0 = measure(); th = th0; probes = 1
    F = min(30.0, A_est / (2 * MUT_MIN)); moved = False
    log = [(None, th)]
    F_last = None
    while probes < MAX_PROBES and r.held():
        # fast down to just above the previous release force (the rod held there), then slow
        if F_last is not None and 1.05 * F_last < r.grip:
            r.ramp_grip(1.05 * F_last, secs=0.5, hold=0.0)
        r.ramp_grip(F, secs=max(0.5, (r.grip - F) / RATE), hold=1.0)
        F_last = F
        r.clamp(); th = measure(); probes += 1; log.append((F_last, th))
        if th <= target + TOL:
            break
        if not moved and th - th0 > -0.03:
            F = F_last * SEARCH; continue
        moved = True
        c = np.cos(th) / F_last                     # quasi-static soft-finger: cos(theta) ~ c*F
        aim = th + GAIN * (target - th)
        F = min(F_last * 0.995, np.cos(aim) / c)
    return probes, log


def run_method(o, method, rng):
    r = Rig(o["m"], o["d"], o["mu"], o["mut"])
    probes, self_est = 0, None
    if method.startswith("open"):
        p = NOM if method == "open_nominal" else o
        F_stop = p["m"] * 9.81 * p["d"] * np.cos(o["target"]) / (2 * p["mut"])
        A_est = o["m"] * 9.81 * o["d"] if method == "open_oracle" else NOM["m"] * 9.81 * NOM["d"]
        F_start = min(30.0, max(A_est / (2 * MUT_MIN), F_stop + 1.0))
        r.ramp_grip(F_start, secs=0.5, hold=0.5)
        r.ramp_grip(F_stop, secs=max(0.5, (F_start - F_stop) / RATE), hold=1.0)
        r.clamp()
    else:
        mode = "oracle" if method == "closed_oracle" else "current"
        if mode == "oracle":
            A_est = o["m"] * 9.81 * o["d"]
        else:
            _, A_amp = initial_probe(r, rng)
            A_est = A_amp * KT_NOM * rng.uniform(0.9, 1.1)    # nominal motor constant, +/-10%
        probes, log = stepped_loop(r, o["target"], mode, rng, A_est)
        self_est = log[-1][1]
    return dict(final=r.com_dir(), rod=r.rod(), held=r.held(), self_est=self_est, probes=probes, t_sim=r.t_sim)


if __name__ == "__main__":
    part, nparts, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    g = np.random.default_rng(99); objs = []
    for i in range(N):
        objs.append(dict(i=i, m=g.uniform(0.05, 0.2), d=g.uniform(0.02, 0.06), mu=g.uniform(0.4, 0.8),
                         mut=g.uniform(0.002, 0.006), target=g.uniform(-1.3, -0.3)))
    out = []
    for o in objs[part::nparts]:
        row = dict(o)
        for meth in ("open_nominal", "open_oracle", "closed_oracle", "closed_current"):
            row[meth] = run_method(o, meth, np.random.default_rng(1000 + o["i"]))
        out.append(row)
        print(o["i"], {m: round(row[m]["final"] - o["target"], 3) for m in ("open_nominal", "open_oracle", "closed_oracle", "closed_current")}, flush=True)
        json.dump(out, open(f"results/expC_part{part}.json", "w"), indent=1, default=float)
    print("done", flush=True)
