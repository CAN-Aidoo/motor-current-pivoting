"""Exp A (MuJoCo): randomized objects and grips, three probe amplitudes, ideal sensing.
Saves ground truth + full probe traces (psi, dpsi, ddpsi_cmd, tau) for Exp B."""
import sys, numpy as np
from mj_env import trial
amp=float(sys.argv[1]); N=int(sys.argv[2]); HZ=0.1
rng=np.random.default_rng(2026)                 # identical object/grip draws for every amplitude
rows=[]; traces=[]
for i in range(N):
    m=rng.uniform(0.05,0.2); d=rng.uniform(0.02,0.06); mu=rng.uniform(0.4,0.8); mut=rng.uniform(0.002,0.006)
    Fstar=m*9.81*d/(2*mut)                        # soft-finger pivot threshold
    F=Fstar*rng.uniform(0.3,1.3)
    r=trial(m,d,mu,mut,F,probe_amp=amp,probe_hz=HZ)
    n=len(r['psi']); w=2*np.pi*HZ; t=np.arange(n)*1e-3
    rows.append([i,m,d,mu,mut,F,Fstar,r['held'],r['theta_true'],r['com_dir'],r['com_r'],r['tau0'],r['tau_rest'],r['theta_rest'],r['probe_slip']])
    traces.append(np.stack([r['psi'],r['dpsi'],-amp*w*w*np.cos(w*t),r['tau'],r['ddpsi']]).astype(np.float32))
np.savez_compressed(f"results/mjA_amp{amp}.npz",rows=np.array(rows,dtype=float),traces=np.array(traces))
print("done",amp,flush=True)
