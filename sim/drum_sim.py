"""OpenArm v2 drumming in MuJoCo: does joint stiffness (EMG-style k) change the stroke outcome?

  python sim/drum_sim.py                      # sweep k, plots + metrics -> sim/out/
  python sim/drum_sim.py --video              # also render free (k=0) vs dead (k=1) side by side

Robot: official OpenArm v2 MJCF (enactic/openarm_mujoco, cloned next to this repo). Every joint is
torque-driven and controlled like the real Damiao MIT mode at 1 kHz:
    tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_g(q)        (clipped to the motor's peak torque)
A 40 cm, 50 g stick is fixed in the right gripper, extending the forearm line; a 13 cm pad with an
elastic head sits under its tip. One stroke: the wrist (J6) and elbow (J4) swing down along a
minimum-jerk path that aims `aim` metres below the head, hold, then return up.
Stiffness k in [0, 1] scales kp of the striking joints between kp_lo and kp_hi (kd keeps a constant
damping ratio, kd ~ sqrt(kp)); trajectory, aim and timing are identical for every k.

Outputs per stroke: peak contact force (~ MIDI velocity), first-contact duration, number of
contacts in 300 ms (bounces), rebound height of the tip after the first contact, and the force still
pressing the head 150 ms after impact (dead stroke = stick held on the head).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

ROOT = Path(__file__).resolve().parent
MJCF = Path(os.environ.get("OPENARM_MUJOCO", ROOT.parent.parent / "openarm_mujoco")) / "v2" / "openarm_bimanual.xml"
OUT = ROOT / "out"

ARM = [f"openarm_right_joint{i}" for i in range(1, 8)]
# default MIT gains (= the position-actuator gains in the official model) and peak torques
KP0 = np.array([230, 230, 190, 190, 30, 30, 30.0])
KD0 = np.array([2.7, 2.7, 2.2, 2.2, 1.5, 1.5, 1.5])
TAU_MAX = np.array([40, 40, 27, 27, 7, 7, 7.0])
STRIKE = [3, 5]                     # indices in ARM: J4 elbow, J6 wrist pitch (positive = tip down)
POSE = np.array([0.35, 0.08, 0.0, 1.15, 0.0, 0.0, 0.0])   # upper arm slightly forward, forearm ~horizontal
STICK_L, STICK_R, STICK_M = 0.40, 0.007, 0.05
GRIP_Z = -0.068                     # gripper pinch point in the ee frame (finger joint offset)
# model-specific: body holding the stick, stick start/direction in that body's z, wrist pitch joint
EE, STICK_Z0, STICK_DIR = "openarm_right_ee_base_link", GRIP_Z, -1
WRIST, WRIST_DOWN = 5, 1            # index in ARM of the wrist pitch joint; sign of "tip down"


def use_model(version="v2", side="right"):
    """Switch to the v1.0 hardware model (wrist pitch = J7) or back to v2, for either arm.

    The left arm mirrors the right: same J4, negated J1-J3, J5-J7 (see robot/sim2real.py MIRROR), so the
    planned (right-arm) angles are mirrored by the caller; here only names and the wrist joint change."""
    global MJCF, ARM, EE, STICK_Z0, STICK_DIR, WRIST, WRIST_DOWN, STRIKE
    MJCF = MJCF.parent.parent / version / "openarm_bimanual.xml"
    ARM = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
    if version == "v1":
        EE, STICK_Z0, STICK_DIR = f"openarm_{side}_hand_tcp", 0.0, 1    # tcp = pinch point, z points out
        WRIST, WRIST_DOWN = 6, -1                                          # right J7: +q lifts the tip
    else:
        EE, STICK_Z0, STICK_DIR = f"openarm_{side}_ee_base_link", GRIP_Z, -1
        WRIST, WRIST_DOWN = 5, 1
    STRIKE = [3, WRIST]


def build(head_k=3.0e4, head_b=8.0, contact_j6=0.20, dt=0.0005, contact_q=None, pad_raise=0.0, as_spec=False,
          kit="pad"):
    """Model with torque motors on every joint, a stick in the right gripper and a pad under its tip.

    The head is placed where the tip is at contact_q (default: POSE with J6 = contact_j6), raised by
    pad_raise metres (> 0: the commanded path goes that far "through" the head, like aiming below it).
    as_spec: return the MjSpec instead of the compiled model (to write it out as XML).
    kit: "pad" (a single 13 cm pad on a stand, as in the W1-W3 sims) or "ded70" (the Donner DED-70 kit around
    a 6.5" snare, sim/ded70.py)."""
    spec = mujoco.MjSpec.from_file(str(MJCF))
    spec.option.timestep = dt
    for act in list(spec.actuators):
        spec.delete(act)
    joints = [j.name for j in spec.joints]
    for jn in joints:
        spec.add_actuator(name=f"tau_{jn}", target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          ctrllimited=True, ctrlrange=[-50, 50])
    ee = spec.body(EE)
    tip_z = STICK_Z0 + STICK_DIR * STICK_L
    ee.add_geom(name="stick", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[STICK_R, 0, 0], mass=STICK_M,
                fromto=[0, 0, STICK_Z0, 0, 0, tip_z], rgba=[0.85, 0.7, 0.45, 1])
    ee.add_site(name="tip", pos=[0, 0, tip_z])
    # place the pad so the tip touches the head when J6 = contact_j6 (rest of the arm at POSE)
    m = spec.compile()
    d = mujoco.MjData(m)
    if contact_q is None:
        q = POSE.copy(); q[WRIST] = WRIST_DOWN * contact_j6
    else:
        q = np.asarray(contact_q, float)
    set_arm(m, d, q)
    mujoco.mj_forward(m, d)
    tip = d.site_xpos[m.site("tip").id].copy()
    tip[2] += pad_raise
    if kit == "ded70":
        from ded70 import add_ded70
        add_ded70(spec, [tip[0], tip[1], tip[2] - STICK_R], head_solref=[-head_k, -head_b])
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], pos=[0, 0, -0.5],
                                rgba=[0.3, 0.32, 0.35, 1], contype=0, conaffinity=0)
        spec.worldbody.add_light(pos=[0.5, -0.5, 1.0], dir=[-0.4, 0.4, -1], diffuse=[0.8, 0.8, 0.8])
        return spec if as_spec else spec.compile()
    pad = spec.worldbody.add_body(name="pad", pos=[tip[0], tip[1], tip[2] - STICK_R - 0.01])
    # direct stiffness/damping contact (solref < 0): head_k N/m, head_b N s/m
    pad.add_geom(name="head", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.13, 0.01, 0],
                 rgba=[0.92, 0.92, 0.92, 1], solref=[-head_k, -head_b], solimp=[0.95, 0.99, 0.001, 0.5, 2])
    pad.add_geom(name="rim", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.14, 0.012, 0], pos=[0, 0, -0.004],
                 rgba=[0.1, 0.1, 0.1, 1], contype=0, conaffinity=0)
    stand = spec.worldbody.add_body(name="stand", pos=[tip[0], tip[1], (tip[2] - 0.5) / 2])
    stand.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012, abs(tip[2] + 0.5) / 2 - 0.02, 0],
                   rgba=[0.2, 0.2, 0.2, 1], contype=0, conaffinity=0)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], pos=[0, 0, -0.5],
                            rgba=[0.3, 0.32, 0.35, 1], contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[0.5, -0.5, 1.0], dir=[-0.4, 0.4, -1], diffuse=[0.8, 0.8, 0.8])
    return spec if as_spec else spec.compile()


def arm_idx(m):
    qi = np.array([m.jnt_qposadr[m.joint(n).id] for n in ARM])
    vi = np.array([m.jnt_dofadr[m.joint(n).id] for n in ARM])
    return qi, vi


def set_arm(m, d, q):
    qi, _ = arm_idx(m)
    d.qpos[qi] = q


def min_jerk(t, t0, t1, a, b):
    """position, velocity, acceleration of a minimum-jerk move a -> b over [t0, t1]."""
    T = t1 - t0
    s = np.clip((t - t0) / T, 0, 1)
    sh = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    inside = 0 < s < 1
    dsh = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T if inside else 0.0
    ddsh = (60 * s - 180 * s ** 2 + 120 * s ** 3) / T ** 2 if inside else 0.0
    return a + (b - a) * sh, (b - a) * dsh, (b - a) * ddsh


def stroke_plan(contact_j6, aim_rad, up=-0.45, t_down=0.16, t_hold=0.20, t_up=0.25, t_start=0.3):
    """Desired (q, dq, ddq) of J6 and J4 over time for one down-hold-up stroke."""
    j6_low = contact_j6 + aim_rad
    j4_up, j4_low = POSE[3] + 0.10, POSE[3]       # forearm raised in the preparation, back to POSE at the hit

    def at(t):
        if t < t_start + t_down:
            return min_jerk(t, t_start, t_start + t_down, up, j6_low), min_jerk(t, t_start, t_start + t_down, j4_up, j4_low)
        t2 = t_start + t_down + t_hold
        return min_jerk(t, t2, t2 + t_up, j6_low, up), min_jerk(t, t2, t2 + t_up, j4_low, j4_up)
    return at, t_start + t_down + t_hold + t_up + 0.2


def gains(k, kp_lo_frac=0.08, kp_hi_frac=1.0):
    """k in [0,1] -> per-joint kp, kd. Striking joints are scaled, the rest keep the default gains."""
    kp, kd = KP0.copy(), KD0.copy()
    frac = kp_lo_frac + k * (kp_hi_frac - kp_lo_frac)
    kp[STRIKE] = KP0[STRIKE] * frac
    kd[STRIKE] = KD0[STRIKE] * np.sqrt(frac)
    return kp, kd


def run(m, k, aim_m=0.03, contact_j6=0.20, record=False, render=None, feedforward=True):
    """One stroke from stroke_plan. feedforward=True adds inverse-dynamics torque of the planned motion
    (MIT tau_ff), so the free-space swing is the same for every k and kp only shapes the contact response."""
    aim_rad = aim_m / (STICK_L - 0.02)        # aim depth at the tip -> wrist angle
    plan, T = stroke_plan(contact_j6, aim_rad)

    def ref(t):
        (q6, d6, a6), (q4, d4, a4) = plan(t)
        qd, dqd, ddqd = POSE.copy(), np.zeros(7), np.zeros(7)
        w, sg = WRIST, WRIST_DOWN
        qd[w], dqd[w], ddqd[w], qd[3], dqd[3], ddqd[3] = sg * q6, sg * d6, sg * a6, q4, d4, a4
        return qd, dqd, ddqd
    r, frames = simulate(m, ref, T, lambda t: k, render=render, feedforward=feedforward)
    return metrics(r), (r if record else None), frames


def simulate(m, ref, T, k_of_t, render=None, feedforward=True, fps=60, passive=(), on_step=None):
    """Track ref(t) -> (q, dq, ddq) of the 7 right-arm joints with the 1 kHz MIT controller; k_of_t(t) sets
    the stiffness. Returns time series (t, tip_z above head, contact force, q, q_des, tau, k) and frames.
    passive: joints without a motor (e.g. a stick hinge); on_step(m, d, rec): called every physics step."""
    d = mujoco.MjData(m)
    m_ff = _no_contact(m)     # the commanded path may pass below the head; tau_ff must not see the contact
    dff = mujoco.MjData(m_ff)
    qi, vi = arm_idx(m)
    ai = np.array([m.actuator(f"tau_{n}").id for n in ARM])
    others = [j for j in range(m.njnt) if m.joint(j).name not in ARM and m.joint(j).name not in passive]
    o_q = np.array([m.jnt_qposadr[j] for j in others])
    o_v = np.array([m.jnt_dofadr[j] for j in others])
    o_a = np.array([m.actuator(f"tau_{m.joint(j).name}").id for j in others])
    o_target = np.zeros(len(others))          # left arm hangs, right gripper closed on the stick
    d.qpos[qi] = ref(0.0)[0]
    mujoco.mj_forward(m, d)
    head = m.geom("head").id
    stick = m.geom("stick").id
    tip_site = m.site("tip").id
    head_top = d.geom_xpos[head][2] + m.geom_size[head][1]
    ctrl_dt, next_ctrl = 0.001, 0.0
    rec = {"t": [], "tip_z": [], "tip_z_des": [], "force": [], "q": [], "q_des": [], "tau": [], "k": []}
    dk = mujoco.MjData(m_ff)                  # forward kinematics of the commanded pose
    z_des = 0.0
    frames, fr_dt, next_fr = [], 1 / fps, 0.0
    tau, qd, k = np.zeros(7), ref(0.0)[0], k_of_t(0.0)
    while d.time < T:
        if d.time >= next_ctrl - 1e-9:      # 1 kHz MIT-mode controller
            qd, dqd, ddqd = ref(d.time)
            k = k_of_t(d.time)
            kp, kd = gains(k)
            if feedforward:   # tau_ff = M(qd) ddqd + C(qd, dqd) + g(qd)
                dff.qpos[:] = d.qpos; dff.qvel[:] = 0; dff.qacc[:] = 0
                dff.qpos[qi], dff.qvel[vi], dff.qacc[vi] = qd, dqd, ddqd
                mujoco.mj_inverse(m_ff, dff)
                ff = dff.qfrc_inverse[vi]
            else:
                ff = d.qfrc_bias[vi]
            tau = np.clip(kp * (qd - d.qpos[qi]) + kd * (dqd - d.qvel[vi]) + ff, -TAU_MAX, TAU_MAX)
            d.ctrl[ai] = tau
            d.ctrl[o_a] = np.clip(60 * (o_target - d.qpos[o_q]) - 1.5 * d.qvel[o_v] + d.qfrc_bias[o_v], -20, 20)
            dk.qpos[:] = d.qpos; dk.qpos[qi] = qd
            mujoco.mj_kinematics(m_ff, dk)
            z_des = dk.site_xpos[tip_site][2] - STICK_R - head_top
            next_ctrl += ctrl_dt
        if on_step is not None:
            on_step(m, d, rec)
        mujoco.mj_step(m, d)
        f = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if {c.geom1, c.geom2} == {head, stick}:
                fc = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, fc)
                f += abs(fc[0])
        rec["t"].append(d.time); rec["tip_z"].append(d.site_xpos[tip_site][2] - STICK_R - head_top)
        rec["tip_z_des"].append(z_des)
        rec["force"].append(f); rec["q"].append(d.qpos[qi].copy()); rec["q_des"].append(qd.copy())
        rec["tau"].append(tau.copy()); rec["k"].append(k)
        if render is not None and d.time >= next_fr:
            render.update_scene(d, camera=_cam())
            frames.append(render.render().copy())
            next_fr += fr_dt
    r = {k_: np.array(v) for k_, v in rec.items()}
    return r, frames


_NC = {}


def _no_contact(m):
    if id(m) not in _NC:
        import copy
        mc = copy.copy(m)
        mc.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        _NC[id(m)] = mc
    return _NC[id(m)]


def _cam():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.32, -0.16, -0.28]
    cam.distance, cam.azimuth, cam.elevation = 1.05, 90, -8
    return cam


def metrics(r, thr=0.5):
    t, f, z = r["t"], r["force"], r["tip_z"]
    on = f > thr
    if not on.any():
        return {"hit": False}
    i0 = int(np.argmax(on))
    t0 = t[i0]
    edges = np.flatnonzero(np.diff(on.astype(int)) == 1) + 1
    w300 = (t >= t0) & (t < t0 + 0.3)
    first_end = i0 + int(np.argmax(~on[i0:])) if (~on[i0:]).any() else len(t) - 1
    after = (t >= t[first_end]) & (t < t0 + 0.15)
    hold = (t >= t0 + 0.14) & (t < t0 + 0.16)
    rel = (z - r["tip_z_des"]) if "tip_z_des" in r else None
    return {
        "hit": True,
        "peak_force_N": float(f[w300].max()),
        "first_contact_ms": float((t[first_end] - t0) * 1000),
        "n_contacts_300ms": int(1 + ((t[edges] > t0) & (t[edges] < t0 + 0.3)).sum()),
        "rebound_mm": float(max(z[after].max(), 0) * 1000) if after.any() else 0.0,
        # passive bounce: how far the tip rides above the COMMANDED tip path after the first contact
        "rebound_rel_mm": float(max(rel[after].max(), 0) * 1000) if rel is not None and after.any() else np.nan,
        "press_force_150ms_N": float(f[hold].mean()),
        "on_head_frac_150ms": float(on[(t >= t0) & (t < t0 + 0.15)].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--aim", type=float, nargs="*", default=[0.01, 0.03, 0.06], help="aim depth below head (m)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    m = build()
    ks = np.round(np.linspace(0, 1, 11), 2)
    rows = []
    traces = {}
    for aim in a.aim:
        for k in ks:
            met, rec, _ = run(m, k, aim_m=aim, record=True)
            rows.append({"aim_m": aim, "k": float(k), **met})
            if k in (0.0, 0.5, 1.0):
                traces[(aim, float(k))] = rec
            print(f"aim {aim * 100:.0f} cm  k {k:.1f}  " + "  ".join(
                f"{n} {v:.2f}" if isinstance(v, float) else f"{n} {v}" for n, v in met.items() if n != "hit"))
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "k_sweep.csv", index=False)

    # figure 1: outcome vs k
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    for aim, g in df.groupby("aim_m"):
        lab = f"aim {aim * 100:.0f} cm below head"
        ax[0].plot(g.k, g.rebound_mm, "o-", label=lab)
        ax[1].plot(g.k, g.n_contacts_300ms, "o-", label=lab)
        ax[2].plot(g.k, g.press_force_150ms_N, "o-", label=lab)
        ax[3].plot(g.k, g.peak_force_N, "o-", label=lab)
    for x, t in zip(ax, ["rebound height of tip (mm)", "contacts in 300 ms", "force still on head at 150 ms (N)",
                         "peak impact force (N)"]):
        x.set_xlabel("stiffness k"); x.set_title(t, fontsize=10); x.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("OpenArm v2 (sim): same trajectory, only joint stiffness k changes", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "k_sweep.png", dpi=130)

    # figure 2: tip height + force traces for one aim
    aim = a.aim[len(a.aim) // 2]
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    for k, col in ((0.0, "#1baf7a"), (0.5, "#d19a1f"), (1.0, "#7b4fd6")):
        r = traces[(aim, k)]
        ax[0].plot(r["t"], r["tip_z"] * 1000, color=col, label=f"k = {k:.1f}")
        ax[1].plot(r["t"], r["force"], color=col)
    ax[0].axhline(0, color="k", lw=0.8); ax[0].set_ylabel("tip height above head (mm)"); ax[0].legend()
    ax[0].set_ylim(-15, 120)
    ax[1].set_ylabel("contact force (N)"); ax[1].set_xlabel("time (s)")
    for x in ax:
        x.grid(alpha=0.3); x.set_xlim(0.35, 0.95)
    fig.suptitle(f"one stroke, aim {aim * 100:.0f} cm below the head", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "stroke_traces.png", dpi=130)
    json.dump({"model": str(MJCF), "KP0": KP0.tolist(), "KD0": KD0.tolist(), "strike_joints": ["J4", "J6"],
               "kp_frac_range": [0.08, 1.0], "pose": POSE.tolist(), "stick": [STICK_L, STICK_R, STICK_M]},
              open(OUT / "config.json", "w"), indent=2)
    print(f"saved {OUT}/k_sweep.csv, k_sweep.png, stroke_traces.png")

    if a.video:
        import cv2
        ren = mujoco.Renderer(m, 480, 640)
        vids = {}
        for k in (0.0, 1.0):
            _, _, fr = run(m, k, aim_m=aim, render=ren)
            vids[k] = fr
        ren.close()
        n = min(len(v) for v in vids.values())
        w = cv2.VideoWriter(str(OUT / "free_vs_dead.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (1280, 480))
        for i in range(n):  # 60 fps sim frames written at 15 fps = 4x slow motion
            row = []
            for k, lab in ((0.0, "k = 0 (loose)"), (1.0, "k = 1 (stiff)")):
                im = cv2.cvtColor(vids[k][i], cv2.COLOR_RGB2BGR)
                cv2.putText(im, lab, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                row.append(im)
            fr = np.hstack(row)
            cv2.putText(fr, "4x slow motion", (1080, 466), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
            w.write(fr)
        w.release()
        print(f"saved {OUT}/free_vs_dead.mp4")


if __name__ == "__main__":
    main()
