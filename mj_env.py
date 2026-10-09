"""MuJoCo gravity-pivoting testbed (soft-finger contacts).

Wrist hinge (axis x) carries a palm with two sliding finger pads. Each pad is a small
sphere, so each pad makes one contact with condim=4: Coulomb sliding friction mu and
torsional friction mu_t (units: m), i.e. the standard soft-finger model, giving a pivot
holding torque of 2*mu_t*F_grip. The rod is grasped a distance d from its centre of mass.
"""
import numpy as np, mujoco

XML = """
<mujoco>
  <option timestep="0.001" integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="20" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="palm" pos="0 0 0.25">
      <joint name="wrist" type="hinge" axis="1 0 0" damping="0.02"/>
      <geom type="box" size="0.05 0.008 0.008" pos="0 0 0" mass="0.3" contype="0" conaffinity="0"/>
      <body name="lf" pos="-0.03 0 0">
        <joint name="lj" type="slide" axis="1 0 0" damping="300" range="0 0.03" limited="true"/>
        <geom name="lpad" type="sphere" size="0.006" mass="0.02" condim="4" friction="{mu} {mut} 0.0001" solref="0.003 1" solimp="0.95 0.99 0.0005"/>
      </body>
      <body name="rf" pos="0.03 0 0">
        <joint name="rj" type="slide" axis="-1 0 0" damping="300" range="0 0.03" limited="true"/>
        <geom name="rpad" type="sphere" size="0.006" mass="0.02" condim="4" friction="{mu} {mut} 0.0001" solref="0.003 1" solimp="0.95 0.99 0.0005"/>
      </body>
    </body>
    <body name="rod" pos="0 {d} 0.25">
      <freejoint/>
      <geom name="rodg" type="box" size="0.01 0.08 0.01" mass="{m}" condim="4" friction="{mu} {mut} 0.0001" solref="0.003 1" solimp="0.95 0.99 0.0005"/>
    </body>
  </worldbody>
  <contact><exclude body1="lf" body2="rf"/></contact>
  <actuator>
    <position name="wrist_servo" joint="wrist" kp="20" kv="1" forcerange="-5 5"/>
    <motor name="lgrip" joint="lj" gear="1" ctrlrange="0 40"/>
    <motor name="rgrip" joint="rj" gear="1" ctrlrange="0 40"/>
  </actuator>
</mujoco>"""


def rod_angle(d):
    R = d.body("rod").xmat.reshape(3, 3)
    return float(np.arctan2(R[2, 1], R[1, 1]))


def build(m, d, mu, mut):
    model = mujoco.MjModel.from_xml_string(XML.format(m=m, d=d, mu=mu, mut=mut))
    return model, mujoco.MjData(model)


def run(model, data, secs):
    for _ in range(int(round(secs / model.opt.timestep))):
        mujoco.mj_step(model, data)


def trial(m, d, mu, mut, grip, probe_amp=0.15, probe_hz=0.2, clamp=30.0, setup=30.0, settle_s=2.0):
    model, data = build(m, d, mu, mut)
    for j in ("lj", "rj"):                              # start with pads already on the rod faces
        data.qpos[model.joint(j).qposadr[0]] = 0.03 - 0.01 - 0.006 + 0.0002
    data.ctrl[:] = [0, setup, setup]
    run(model, data, 0.8)                               # held by friction only from t = 0
    tau0 = float(data.actuator_force[0]); ang0 = rod_angle(data)
    data.ctrl[1:] = grip                                # loosen: gravity pivots the rod
    run(model, data, settle_s)
    held = data.body("rod").xpos[2] > 0.13
    tau_rest = float(data.actuator_force[0]); theta_rest = rod_angle(data)
    data.ctrl[1:] = clamp                               # re-clamp
    run(model, data, 0.5)
    theta_true = rod_angle(data)
    cp = data.body("rod").xpos - np.array([0, 0, 0.25])
    com_dir = float(np.arctan2(cp[2], cp[1])); com_r = float(np.hypot(cp[1], cp[2]))
    w = 2 * np.pi * probe_hz; n = int(1 / probe_hz / model.opt.timestep)
    psi = np.zeros(n); dpsi = np.zeros(n); tau = np.zeros(n); rel = np.zeros(n)
    jid = model.joint("wrist").qposadr[0]; vid = model.joint("wrist").dofadr[0]
    for k in range(n):
        t = k * model.opt.timestep
        data.ctrl[0] = -probe_amp * (1 - np.cos(w * t))     # one-sided 'downhill' sweep
        mujoco.mj_step(model, data)
        psi[k] = data.qpos[jid]; dpsi[k] = data.qvel[vid]; tau[k] = data.actuator_force[0]
        rel[k] = rod_angle(data) - psi[k]
    ddpsi = np.gradient(dpsi, model.opt.timestep)
    return dict(m=m, d=d, mu=mu, mut=mut, grip=grip, held=bool(held), tau0=tau0, ang0=ang0,
                tau_rest=tau_rest, theta_rest=theta_rest, theta_true=theta_true, com_dir=com_dir, com_r=com_r,
                probe_slip=float(np.max(np.abs(rel - rel[0]))), psi=psi, dpsi=dpsi, ddpsi=ddpsi, tau=tau)
