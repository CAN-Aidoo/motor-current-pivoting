"""Stateful pivoting rig for closed-loop experiments (same MJCF as mj_env.py)."""
import numpy as np, mujoco
from mj_env import build, rod_angle


class Rig:
    def __init__(self, m, d, mu, mut, setup=30.0):
        self.model, self.data = build(m, d, mu, mut)
        M, D = self.model, self.data
        for j in ("lj", "rj"):
            D.qpos[M.joint(j).qposadr[0]] = 0.03 - 0.01 - 0.006 + 0.0002
        D.ctrl[:] = [0, setup, setup]
        self.grip = setup
        self.t_sim = 0.0
        self.step(0.8)
        self.jid = M.joint("wrist").qposadr[0]; self.vid = M.joint("wrist").dofadr[0]

    def step(self, secs):
        for _ in range(int(round(secs / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)
        self.t_sim += secs

    def held(self):
        return bool(self.data.body("rod").xpos[2] > 0.13)

    def com_dir(self):
        cp = self.data.body("rod").xpos - np.array([0, 0, 0.25])
        return float(np.arctan2(cp[2], cp[1])) - float(self.data.qpos[self.jid])

    def rod(self):
        return rod_angle(self.data) - float(self.data.qpos[self.jid])

    def ramp_grip(self, F_to, secs=2.0, hold=1.0):
        """Lower (or raise) grip force linearly: quasi-static release."""
        F0 = self.grip; n = int(secs / self.model.opt.timestep)
        for k in range(n):
            self.data.ctrl[1:] = F0 + (F_to - F0) * (k + 1) / n
            mujoco.mj_step(self.model, self.data)
        self.grip = F_to; self.t_sim += secs
        self.step(hold)

    def clamp(self, F=30.0, ramp_s=0.5):
        """Re-clamp with a smooth force ramp (a step change excites pad/contact chatter)."""
        F0 = self.grip; n = int(ramp_s / self.model.opt.timestep)
        for k in range(n):
            self.data.ctrl[1:] = F0 + (F - F0) * (k + 1) / n
            mujoco.mj_step(self.model, self.data)
        self.t_sim += ramp_s
        self.grip = F; self.step(0.5)

    def probe(self, amp=0.15, hz=0.1):
        M, D = self.model, self.data
        w = 2 * np.pi * hz; n = int(1 / hz / M.opt.timestep)
        psi = np.zeros(n); dpsi = np.zeros(n); tau = np.zeros(n)
        for k in range(n):
            D.ctrl[0] = -amp * (1 - np.cos(w * k * M.opt.timestep))
            mujoco.mj_step(M, D)
            psi[k] = D.qpos[self.jid]; dpsi[k] = D.qvel[self.vid]; tau[k] = D.actuator_force[0]
        D.ctrl[0] = 0.0; self.step(0.3)
        self.t_sim += n * M.opt.timestep
        t = np.arange(n) * M.opt.timestep
        return psi, dpsi, -amp * w * w * np.cos(w * t), tau, np.gradient(dpsi, M.opt.timestep)


def dither_release(rig, F_from, rate, target, est_fn, lead=0.0, amp=0.04, hz=1.0, win_s=1.0,
                   every_s=0.1, F_min=0.5, t_max=60.0, log=None):
    """Lower grip at `rate` N/s while dithering the wrist downhill; re-estimate the angle
    over a rolling window and clamp as soon as the estimate reaches target + lead."""
    M, D = rig.model, rig.data
    dt = M.opt.timestep; w = 2 * np.pi * hz; nwin = int(win_s / dt); nev = int(every_s / dt)
    buf = {k: [] for k in ("psi", "dpsi", "ddcmd", "tau", "ddact")}
    F = F_from; k = 0; th_hat = None
    while k * dt < t_max:
        t = k * dt
        D.ctrl[0] = -amp * (1 - np.cos(w * t))
        F = max(F_min, F_from - rate * t); D.ctrl[1:] = F
        mujoco.mj_step(M, D); k += 1
        buf["psi"].append(D.qpos[rig.jid]); buf["dpsi"].append(D.qvel[rig.vid])
        buf["ddcmd"].append(-amp * w * w * np.cos(w * t)); buf["tau"].append(D.actuator_force[0])
        buf["ddact"].append(0.0)
        if k >= nwin and k % nev == 0:
            seg = {kk: np.array(v[-nwin:]) for kk, v in buf.items()}
            seg["ddact"] = np.gradient(seg["dpsi"], dt)
            th_hat = est_fn(seg)
            if log is not None: log.append((t, F, th_hat, rig.com_dir()))
            if th_hat <= target + lead:
                break
        if not rig.held(): break
    rig.t_sim += k * dt
    rig.grip = F
    rig.clamp()
    D.ctrl[0] = 0.0; rig.step(0.5)
    return th_hat, F
