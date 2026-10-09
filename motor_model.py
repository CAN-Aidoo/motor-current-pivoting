"""Wrist actuator + current-sensing chain, applied to simulated wrist load torque.

Actuator (brushed DC gearmotor, output-referred):
    N*Kt*i = tau_load + tau_c*sgn(w) + b*w + J_r*alpha          (moving)
    at rest, gearbox stiction takes an unknown value in [-tau_c, +tau_c]
Sensing: i_meas = clip(Q_b(i + n), -I_fs, I_fs),  n ~ N(0, sigma_n^2),  Q_b = b-bit uniform quantizer.
Defaults are representative of a small 12 V brushed motor with a 100:1 gearbox; every
value is a swept parameter, not a claim about one specific part."""
import numpy as np

DEFAULT = dict(N=100.0, Kt=0.0107, tau_c=0.0, b_visc=0.0, J_r=2e-7*100**2,
               bits=12, I_fs=2.0, sigma_n=0.0, fs=1000.0, sim_fs=1000.0)

def lsb(P): return 2*P["I_fs"]/2**P["bits"]

def quantize(i, P):
    q = lsb(P)
    return np.clip(np.round(i/q)*q, -P["I_fs"], P["I_fs"])

def current_trace(psi, dpsi, ddpsi, tau_load, P, rng):
    """Measured motor current during the probe, decimated to fs."""
    step = int(round(P["sim_fs"]/P["fs"]))
    sl = slice(None, None, step)
    psi, dpsi, ddpsi, tau_load = psi[sl], dpsi[sl], ddpsi[sl], tau_load[sl]
    tau_m = tau_load + P["tau_c"]*np.sign(dpsi) + P["b_visc"]*dpsi + P["J_r"]*ddpsi
    i = tau_m/(P["N"]*P["Kt"])
    i_meas = quantize(i + rng.normal(0, P["sigma_n"], i.shape) if P["sigma_n"] > 0 else i, P)
    return psi, dpsi, ddpsi, i_meas

def static_current(tau_load, P, rng, n_avg=100):
    """Averaged current reading with the joint at rest (stiction value unknown)."""
    tau_f = rng.uniform(-P["tau_c"], P["tau_c"])
    i = (tau_load + tau_f)/(P["N"]*P["Kt"])
    reads = quantize(i + (rng.normal(0, P["sigma_n"], n_avg) if P["sigma_n"] > 0 else np.zeros(n_avg)), P)
    return reads.mean()

# ---------------- estimators ----------------
def regressors(psi, dpsi, friction=True):
    cols = [np.cos(psi), np.sin(psi)]
    if friction: cols += [np.sign(dpsi), dpsi]
    return np.stack(cols, 1)

def sweep_ls(psi, dpsi, ddpsi, i_meas, P, friction=True, comp_inertia=True, skip=0.2):
    """theta = atan2(-b, a) from i ~ a cos(psi) + b sin(psi) [+ c sgn(w) + v w].
    Kt and N cancel in the ratio, so no motor calibration is needed."""
    k0 = int(skip*P["fs"])
    y = i_meas.copy()
    if comp_inertia: y = y - P["J_r"]*ddpsi/(P["N"]*P["Kt"])   # datasheet rotor inertia, known
    X = regressors(psi, dpsi, friction)[k0:]; y = y[k0:]
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X@c
    return float(np.arctan2(-c[1], c[0])), c, float(resid.std()), X

def predicted_sd(X, c, sigma_eff):
    """Linearized LS prediction of the angle standard deviation."""
    cov = sigma_eff**2*np.linalg.inv(X.T@X)
    a, b = c[0], c[1]; g = np.zeros(len(c)); g[0] = b/(a*a+b*b); g[1] = -a/(a*a+b*b)
    return float(np.sqrt(g@cov@g))

def naive_static(i0, i_rest):
    """Single-read estimate: needs a horizontal calibration read i0 and assumes the offset d is unchanged."""
    return float(-np.arccos(np.clip(i_rest/i0, -1, 1)))
