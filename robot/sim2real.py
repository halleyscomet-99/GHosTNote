"""Sim2real on one OpenArm arm (default: left arm on can_left): same pose, same stroke, same MIT law as sim/drum_sim.py.

  python robot/sim2real.py goto --frac 0.3          # 30 % of the way to the sim drumming POSE, hold, gravity check, back
  python robot/sim2real.py goto                     # all the way to POSE; writes data/robot/gravity_check_<side>.json
  python robot/sim2real.py goto --can can_right --side right
  (--model v1 is the default: OpenArm v1.0 hardware, wrist pitch = J7; v2 has it on J6)
  python robot/sim2real.py stroke --slow 4          # one stroke in the air at 1/4 speed, then the same in MuJoCo
  python robot/sim2real.py stroke --slow 2 --n 3 --k 0.2
  python robot/sim2real.py contact --stick          # hold the pose where the tip should touch the head; place the pad
  python robot/sim2real.py stroke --stick --slow 2 --aim 1 --midi   # onto the pad, drum hits scored per stroke
  python robot/sim2real.py teleop --stick --aim 1 --midi            # stiffness from the operator's Myo (live_stiffness --udp)

Control law on the robot (MIT mode): tau = kp (q_des - q) + kd (dq_des - dq) + tau_ff, where tau_ff is the
MuJoCo inverse dynamics of the planned motion (gravity + inertia), sent only once the gravity check has
shown that every joint's sign agrees with the model. kp/kd are the real arm's defaults (openarm_zmq), with
J4 and the wrist pitch joint scaled by k exactly like drum_sim.gains(). Every run starts and ends at the pose the arm rests in.
Outputs: data/robot/<time>_<cmd>.csv and, for stroke, data/robot/<time>_stroke_vs_sim.png
"""
import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "robot"))
sys.path.insert(0, str(ROOT / "scripts"))
import drum_sim as ds  # noqa: E402
import mujoco  # noqa: E402
from arm_bringup import KD, KD_DAMP, KP, NAMES, TAU_PEAK, Arm  # noqa: E402

OUT = ROOT / "data" / "robot"
# left arm = mirror image of the right one across the robot's sagittal plane: with the model's joint axes,
# a mirrored motion has the same J4 angle and the negated angle on every other joint
MIRROR = {"right": np.ones(7), "left": np.array([-1, -1, -1, 1, -1, -1, -1.0])}
HZ = 250
GRIP = {"q": None, "kp": KP[7]}          # gripper target during track(); q None = hold where it is
MAX_ERR = 0.3
ERR_TAU = 0.25 * TAU_PEAK[:7]           # a soft joint may lag further: its PD torque at the limit is still small


def err_limit(kp):
    """Tracking error that stops the arm: 0.3 rad, or more on a soft joint (k -> 0) whose static friction
    alone holds it ~friction / kp away from the target; never above 0.7 rad."""
    return np.clip(ERR_TAU / np.maximum(kp, 1e-6), MAX_ERR, 0.7)
FRICTION = 0.8                                          # N·m the gravity check tolerates
FRIC_MAX = np.array([2.0, 2.0, 1.5, 1.5, 0.5, 0.5, 0.5])   # cap on the friction compensation (N·m)
PROBE = np.radians(5.0)                                   # goto: approach the pose from +-5 deg
FF_MAX = 0.4 * np.array([40, 40, 27, 27, 7, 7, 7.0])   # feedforward never asks for more than 40 % of peak


def real_gains(k):
    """drum_sim.gains() but around the real arm's default gains."""
    kp, kd = KP[:7].copy(), KD[:7].copy()
    frac = 0.08 + k * (1.0 - 0.08)
    kp[ds.STRIKE] *= frac
    kd[ds.STRIKE] *= np.sqrt(frac)
    return kp, kd


class Model:
    """MuJoCo inverse dynamics of the arm named in ds.ARM (the other arm at zero, no contacts)."""

    def __init__(self):
        # no contacts and no joint-limit constraints: the real arm rests a few degrees outside the model's
        # J4 range (0..140 deg), and a limit force there turns into a huge feedforward torque
        self.m = ds.build(pad_raise=-1.0)
        self.m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
        # rigid-body terms only (gravity, inertia, Coriolis): the model's guessed joint damping (0.9-1 N·m·s/rad)
        # and frictionloss would add ~1.7 N·m at the wrist mid-swing; real friction is identified by `goto` instead
        self.m.dof_damping[:] = 0
        self.m.dof_frictionloss[:] = 0
        self.d = mujoco.MjData(self.m)
        self.qi, self.vi = ds.arm_idx(self.m)
        self.g_scale = np.ones(7)     # real / model gravity torque per joint, from the gravity check
        self.friction = np.zeros(7)   # Coulomb friction per joint (N·m), from the gravity check

    def calibrate(self, check):
        """Scale the gravity part of the feedforward by what the held-pose check measured (0.3x .. 2x)
        and add Coulomb friction compensation in the direction of the planned velocity."""
        c = json.load(open(check))
        g, t = np.array(c["model_g"]), np.array(c["measured_tau"])
        for i, ok in enumerate(c["sign_ok"]):
            if ok and abs(g[i]) >= 0.5:
                self.g_scale[i] = float(np.clip(t[i] / g[i], 0.3, 2.0))
        self.friction = np.clip(np.array(c.get("friction", np.zeros(7))), 0, FRIC_MAX)

    def tau(self, q, dq=None, ddq=None, scaled=True):
        d = self.d
        d.qpos[:] = 0; d.qvel[:] = 0; d.qacc[:] = 0
        d.qpos[self.qi] = q
        if dq is not None:
            d.qvel[self.vi] = dq
        if ddq is not None:
            d.qacc[self.vi] = ddq
        mujoco.mj_inverse(self.m, d)
        out = d.qfrc_inverse[self.vi].copy()
        if scaled and np.any(self.g_scale != 1):
            d.qvel[:] = 0; d.qacc[:] = 0
            mujoco.mj_inverse(self.m, d)
            out += (self.g_scale - 1) * d.qfrc_inverse[self.vi]
        if scaled and dq is not None:
            out += self.friction * np.tanh(np.asarray(dq) / 0.05)
        return np.clip(out, -FF_MAX, FF_MAX)


def track(arm, model, ref, T, kp, kd, use_ff, log, tag, dq_max=3.0):
    """Follow ref(t) -> (q, dq, ddq) for the 7 arm joints for T seconds; gripper holds. Returns the rows."""
    q_grip = arm.state()[0][7] if GRIP["q"] is None else GRIP["q"]
    rows, t0 = [], time.monotonic()
    while True:
        t = time.monotonic() - t0
        if t > T:
            break
        qd, dqd, ddqd = ref(t)
        ff = model.tau(qd, dqd, ddqd) if use_ff else np.zeros(7)
        arm.command(np.r_[kp, GRIP["kp"]], np.r_[kd, KD[7]], np.r_[qd, q_grip], np.r_[dqd, 0], np.r_[ff, 0])
        q, dq, tau, tm, _ = arm.state()
        row = [tag, t, *qd, *q[:7], *dq[:7], *tau[:7], *ff, int(tm.max())]
        rows.append(row)
        log.writerow([f"{x:.5f}" if isinstance(x, float) else x for x in row])
        err = np.abs(q[:7] - qd)
        bad = (err > err_limit(kp)) | (np.abs(dq[:7]) > dq_max) | (np.abs(tau[:7]) > 0.8 * TAU_PEAK[:7])
        if bad.any():
            i = int(np.argmax(bad))
            raise RuntimeError(f"safety stop on {NAMES[i]} during {tag}: q {np.degrees(q[i]):+.1f} deg vs "
                               f"target {np.degrees(qd[i]):+.1f}, dq {dq[i]:+.2f}, tau {tau[i]:+.2f}")
        time.sleep(max(0.0, 1.0 / HZ - (time.monotonic() - t0 - t)))
    return rows


def move_ref(a, b, T):
    def ref(t):
        out = [ds.min_jerk(t, 0, T, a[i], b[i]) for i in range(7)]
        return tuple(np.array(x) for x in zip(*out))
    return ref


def stroke_ref(k_slow, mirror, aim_m=0.03, contact_j6=0.20):
    plan, T = ds.stroke_plan(contact_j6, aim_m / (ds.STICK_L - 0.02))

    def ref(t):
        (q6, d6, a6), (q4, d4, a4) = plan(t / k_slow)
        qd, dqd, ddqd = ds.POSE.copy(), np.zeros(7), np.zeros(7)
        w, sg = ds.WRIST, ds.WRIST_DOWN
        qd[w], dqd[w], ddqd[w] = sg * q6, sg * d6 / k_slow, sg * a6 / k_slow ** 2
        qd[3], dqd[3], ddqd[3] = q4, d4 / k_slow, a4 / k_slow ** 2
        return qd * mirror, dqd * mirror, ddqd * mirror
    return ref, T * k_slow


def sim_same(ref, T, k):
    """The same reference and gains in drum_sim (no pad) -> t, q, tau."""
    m = ds.build(pad_raise=-1.0)
    kp0, kd0 = ds.KP0.copy(), ds.KD0.copy()
    ds.KP0, ds.KD0 = KP[:7].copy(), KD[:7].copy()
    try:
        r, _ = ds.simulate(m, ref, T, lambda t: k)
    finally:
        ds.KP0, ds.KD0 = kp0, kd0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["goto", "stroke", "contact", "teleop"])
    ap.add_argument("--can", default="can_left", help="can_left / can_right (robot/udev/80-openarm-can.rules)")
    ap.add_argument("--side", default="left", choices=["left", "right"], help="which arm is on --can")
    ap.add_argument("--model", default="v1", choices=["v1", "v2"], help="OpenArm hardware version (yours: v1.0)")
    ap.add_argument("--fd", action="store_true")
    ap.add_argument("--frac", type=float, default=1.0, help="goto: fraction of the way to POSE")
    ap.add_argument("--move-time", type=float, default=6.0)
    ap.add_argument("--slow", type=float, default=4.0, help="stroke: time-stretch factor (1 = sim speed)")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--ks", default="", help="stroke: several k values, e.g. 1,0.3,0.08: each block of strokes "
                    "plays every k once in shuffled order, --n blocks (separates k from drift over the run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--udp", type=int, default=9100, help="teleop: port where live_stiffness.py streams k")
    ap.add_argument("--period", type=float, default=2.0, help="teleop: seconds between strokes")
    ap.add_argument("--aim", type=float, default=1.0, help="stroke: cm the planned tip path goes below the head")
    ap.add_argument("--midi", action="store_true", help="stroke: record the drum (DED-70) and score each stroke")
    ap.add_argument("--no-ff", action="store_true")
    ap.add_argument("--skip-rest-check", action="store_true",
                    help="start even if a joint rests > 25 deg from zero (the arm then moves it back slowly)")
    ap.add_argument("--stick", action="store_true",
                    help="take a drumstick in the gripper at the start (and add it to the model)")
    ap.add_argument("--squeeze", type=float, default=2.0, help="gripper squeeze on the stick, N·m")
    a = ap.parse_args()
    if a.slow < 1:
        ap.error("--slow below 1 is faster than the sim")

    ds.use_model(a.model, a.side)
    if not a.stick:
        ds.STICK_M = 1e-4                    # nothing in the gripper yet
    mir = MIRROR[a.side]
    start = ds.POSE.copy()
    start[3] += 0.10
    start[ds.WRIST] = ds.WRIST_DOWN * -0.45
    check = OUT / f"gravity_check_{a.model}_{a.side}.json"
    model = Model()
    arm = Arm(a.can, a.fd)
    arm.refresh()
    q_rest, _, _, tm, tr = arm.state()
    if any(tm[i] == 0 and tr[i] == 0 for i in range(8)):
        raise SystemExit("some motor does not reply; run arm_bringup.py read first")
    q_rest = q_rest[:7].copy()
    if np.abs(np.degrees(q_rest[:7])).max() > 25 and not a.skip_rest_check:
        raise SystemExit(f"the arm on {a.can} does not rest near its zero pose (max |q| "
                         f"{np.abs(np.degrees(q_rest[:7])).max():.0f} deg): wrong --can, or J3 / the wrist turned by hand while off? "
                         "run arm_bringup.py read --no-fd and turn those joints back near 0")
    g_ok = json.load(open(check))["all_signs_ok"] if check.exists() else False
    use_ff = g_ok and not a.no_ff and a.cmd != "goto"       # goto measures without feedforward
    if a.cmd != "goto" and not g_ok and not a.no_ff:
        raise SystemExit("gravity check has not passed: fix it, or run the stroke without feedforward (--no-ff)")
    contact = ds.POSE.copy()
    contact[ds.WRIST] = ds.WRIST_DOWN * 0.20          # stroke_plan's contact_j6: the tip touches the head here
    target = {"goto": q_rest + a.frac * (ds.POSE * mir - q_rest), "stroke": start * mir, "teleop": start * mir,
              "contact": contact * mir}[a.cmd]
    print("rest  (deg):", " ".join(f"{NAMES[i]} {np.degrees(q_rest[i]):+6.1f}" for i in range(7)))
    print("target(deg):", " ".join(f"{NAMES[i]} {np.degrees(target[i]):+6.1f}" for i in range(7)))
    if use_ff:
        model.calibrate(check)
    print(f"torque feedforward from MuJoCo: {'ON' if use_ff else 'OFF (gravity check not passed yet)'}"
          + (f"\n  gravity scale {np.round(model.g_scale, 2).tolist()}\n  friction (N·m) "
             f"{np.round(model.friction, 2).tolist()}" if use_ff else ""))

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = OUT / f"{stamp}_{a.cmd}.csv"
    f = open(path, "w", newline="")
    log = csv.writer(f)
    log.writerow(["phase", "t"] + [f"{p}_{n}" for p in ("qcmd", "q", "dq", "tau", "ff") for n in NAMES[:7]] + ["tmos_max"])
    kp, kd = real_gains(1.0)
    if a.stick:
        GRIP["q"], GRIP["kp"] = arm.take_stick(a.squeeze)
        arm.refresh()
        q_rest = arm.state()[0][:7].copy()      # the arm moves while the stick goes in
    arm.oa.enable_all()
    time.sleep(0.1)
    arm.oa.recv_all(2000)
    try:
        # gains in at the rest pose
        track(arm, model, lambda t: (q_rest, np.zeros(7), np.zeros(7)), 1.5, kp * 0.3, kd, False, log, "ramp")
        track(arm, model, move_ref(q_rest, target, a.move_time), a.move_time, kp, kd, use_ff, log, "go")
        hold = track(arm, model, lambda t: (target, np.zeros(7), np.zeros(7)), 2.0, kp, kd, use_ff, log, "hold")
        if a.cmd == "goto":
            # approach the pose from below and from above: holding torque = gravity +- static friction
            held = {}
            for sgn, name in ((-1, "from_below"), (1, "from_above")):
                track(arm, model, move_ref(target, target + sgn * PROBE, 1.5), 1.5, kp, kd, False, log, f"probe_{name}")
                track(arm, model, move_ref(target + sgn * PROBE, target, 1.5), 1.5, kp, kd, False, log, f"back_{name}")
                h = track(arm, model, lambda t: (target, np.zeros(7), np.zeros(7)), 1.5, kp, kd, False, log, name)
                last = np.array([r[2:2 + 7 * 5] for r in h[len(h) // 2:]], float)
                held[name] = (last[:, 7:14].mean(0), last[:, 21:28].mean(0))
            q_m = (held["from_below"][0] + held["from_above"][0]) / 2
            t_up, t_dn = held["from_below"][1], held["from_above"][1]
            tau_m, fric = (t_up + t_dn) / 2, np.abs(t_up - t_dn) / 2
            g = model.tau(q_m, scaled=False)
            print(f"\ngravity check at the held pose (measured gravity = mean of the two approaches, "
                  f"friction = half their difference)")
            print(f"{'joint':6}{'q deg':>8}{'model g':>10}{'measured':>10}{'friction':>10}  check")
            ok = []
            for i in range(7):
                ratio = tau_m[i] / g[i] if abs(g[i]) >= 0.5 else np.nan
                agree = 0.5 <= ratio <= 2.0 or abs(tau_m[i] - g[i]) < FRICTION   # small loads: friction dominates
                ok.append(bool(agree) if abs(g[i]) >= 0.5 else None)
                print(f"{NAMES[i]:6}{np.degrees(q_m[i]):+8.1f}{g[i]:+10.2f}{tau_m[i]:+10.2f}{fric[i]:10.2f}  "
                      + ("too small to tell" if abs(g[i]) < 0.5 else f"OK (x{ratio:.2f}, diff {tau_m[i] - g[i]:+.2f})" if agree
                         else f"<-- MISMATCH (measured = {ratio:.2f} x model)"))
            if a.frac >= 0.99:
                tested = [o for o in ok if o is not None]
                json.dump({"time": stamp, "can": a.can, "q": q_m.tolist(), "model_g": g.tolist(),
                           "measured_tau": tau_m.tolist(), "friction": fric.tolist(), "sign_ok": ok,
                           "stick": a.stick,
                           "all_signs_ok": bool(tested) and all(tested)}, open(check, "w"), indent=1)
                print(f"saved {check}")
        elif a.cmd == "contact":
            print("\nholding the contact pose: put the pad under the stick so the head just touches the tip\n"
                  "(tip resting on the head, not pressing). Press Enter when it is placed.")
            done = threading.Event()
            threading.Thread(target=lambda: (input(), done.set()), daemon=True).start()
            while not done.is_set():
                track(arm, model, lambda t: (target, np.zeros(7), np.zeros(7)), 0.25, kp, kd, use_ff, log, "contact")
        elif a.cmd == "teleop":
            teleop(arm, model, a, mir, target, kp, kd, use_ff, log, stamp)
        else:
            ref, T = stroke_ref(a.slow, mir, aim_m=a.aim / 100)
            ks = [float(x) for x in a.ks.split(",")] if a.ks else [a.k]
            rng = np.random.default_rng(a.seed)
            seq = [float(k) for _ in range(a.n) for k in (rng.permutation(ks) if len(ks) > 1 else ks)]
            a.k = seq[0]                     # the comparison figure shows stroke 1
            dq_max = 3.0 + 1.5 * max(abs(ref(t)[1]).max() for t in np.linspace(0, T, 400))
            midi = start_midi() if a.midi else None
            starts = []
            for s, k in enumerate(seq):
                print(f"stroke {s + 1}/{len(seq)}  (k={k:g}, {a.slow}x slower than sim, aim {a.aim:.1f} cm)")
                starts.append(midi[1]() if midi else 0.0)
                kp_s, kd_s = real_gains(k)
                track(arm, model, ref, T, kp_s, kd_s, use_ff, log, f"stroke{s + 1}", dq_max=dq_max)
            if midi:
                time.sleep(0.3)
                score_midi(midi, starts, seq, T, a.slow, OUT / f"{stamp}_midi.csv")
            track(arm, model, lambda t: (target, np.zeros(7), np.zeros(7)), 0.5, kp, kd, use_ff, log, "hold2")
        q_now = arm.state()[0][:7]
        track(arm, model, move_ref(q_now, q_rest, a.move_time), a.move_time, kp, kd, use_ff, log, "back")
        track(arm, model, lambda t: (q_rest, np.zeros(7), np.zeros(7)), 1.0, kp * 0.3, kd, False, log, "release")
    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"\n!! {e or 'Ctrl-C'} -> damping 1.5 s (support the arm), then motors off")
        for _ in range(int(1.5 * 200)):
            q = arm.state()[0]
            arm.command(np.zeros(8), KD_DAMP, q)
            time.sleep(0.005)
    finally:
        if a.stick:
            try:
                arm.give_stick()
            except (KeyboardInterrupt, EOFError):
                pass
        arm.oa.disable_all()
        arm.oa.recv_all(2000)
        f.close()
        print(f"motors disabled\nlog: {path}")

    if a.cmd == "stroke":
        if a.midi and (OUT / f"{stamp}_midi.csv").exists():
            score_wrist(path, OUT / f"{stamp}_midi.csv", mir)
        compare(path, stamp, a, mir)


class KListener:
    """Latest stiffness k from live_stiffness.py --udp 127.0.0.1:PORT ('t,k,cci' lines)."""

    def __init__(self, port):
        import socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.2)
        self.buf = []                       # (arrival time, k)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                line = self.sock.recv(256).decode().strip()
                if line.startswith("imu"):          # IMU lines are for arm_teleop.py
                    continue
                self.buf.append((time.monotonic(), float(line.split(",")[1])))
                self.buf = self.buf[-400:]
            except (OSError, ValueError, IndexError):
                pass

    def k(self, window=0.2):
        """Mean k over the last `window` s, or None if nothing arrived for 0.5 s."""
        t = time.monotonic()
        recent = [k for tk, k in self.buf if t - tk < window]
        if not self.buf or t - self.buf[-1][0] > 0.5 or not recent:
            return None
        return float(np.clip(np.mean(recent), 0, 1))


def teleop(arm, model, a, mir, target, kp, kd, use_ff, log, stamp):
    """Myo stiffness teleoperation: the robot plays the same stroke every --period s; the stiffness of every
    stroke is the operator's live k (forearm co-contraction) just before the down-swing. Ctrl-C stops."""
    ref, T = stroke_ref(a.slow, mir, aim_m=a.aim / 100)
    dq_max = 3.0 + 1.5 * max(abs(ref(t)[1]).max() for t in np.linspace(0, T, 400))
    listener = KListener(a.udp)
    midi = start_midi() if a.midi else None
    hold = lambda t: (target, np.zeros(7), np.zeros(7))   # noqa: E731
    starts, seq, n = [], [], 0
    print(f"\nteleop: listening for k on udp {a.udp}; a stroke every {a.period:.1f} s; Ctrl-C to stop")
    try:
        while a.n <= 1 or n < a.n:
            track(arm, model, hold, max(0.05, a.period - T), kp, kd, use_ff, log, "wait")
            k = listener.k()
            if k is None:
                print("  no k from live_stiffness.py (is it running with --udp 127.0.0.1:%d?) - waiting" % a.udp)
                continue
            n += 1
            print(f"  stroke {n}: k = {k:.2f}  " + "#" * int(round(k * 20)))
            starts.append(midi[1]() if midi else 0.0)
            seq.append(round(k, 3))
            kp_s, kd_s = real_gains(k)
            track(arm, model, ref, T, kp_s, kd_s, use_ff, log, f"stroke{n}", dq_max=dq_max)
    except KeyboardInterrupt:
        print("\nteleop stopped")
    if midi and starts:
        time.sleep(0.3)
        score_midi(midi, starts, seq, T, a.slow, OUT / f"{stamp}_teleop_midi.csv")
    track(arm, model, hold, 0.5, kp, kd, use_ff, log, "hold2")


def start_midi():
    """DED-70 over USB-MIDI -> (queue, clock, stream)."""
    import queue
    from ghost_io import MidiStream, now
    q = queue.Queue()
    return q, now, MidiStream(q)


def score_midi(midi, starts, ks, T, slow, path):
    """Hits per stroke: when (vs the planned contact), how hard, and any re-hits (bounces)."""
    q, _, stream = midi
    stream.close()
    hits = []
    while not q.empty():
        _, t, typ, note, vel, *_ = q.get()
        if typ == "note_on":
            hits.append((t, note, vel))
    plan_contact = (0.3 + 0.16) * slow                 # stroke_plan: t_start + t_down
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stroke", "k", "t_from_stroke_start", "t_from_planned_contact", "note", "velocity"])
        per_k = {}
        for i, (t0, k) in enumerate(zip(starts, ks)):
            mine = [h for h in hits if t0 <= h[0] < t0 + T]
            for t, note, vel in mine:
                w.writerow([i + 1, k, f"{t - t0:.4f}", f"{t - t0 - plan_contact:+.4f}", note, vel])
            desc = ", ".join(f"vel {v} at {1000 * (t - t0 - plan_contact):+.0f} ms" for t, _, v in mine) or "no hit"
            print(f"  stroke {i + 1} (k={k:g}): {len(mine)} hit(s): {desc}")
            per_k.setdefault(k, []).append([v for *_, v in mine])
    if len(per_k) > 1:
        print("per k: first-hit velocity mean ± sd, strokes with >1 hit")
        for k in sorted(per_k):
            first = [v[0] for v in per_k[k] if v]
            multi = sum(len(v) > 1 for v in per_k[k])
            print(f"  k={k:<5g} vel {np.mean(first):5.1f} ± {np.std(first):4.1f}  (n={len(first)}), re-hits {multi}")
    print(f"midi: {path}")


def score_wrist(path, midi_path, mir):
    """What the wrist encoder sees around each drum hit, per k: down-swing speed just before the hit and the
    upward (rebound) speed of the wrist within 150 ms after it. rad/s, + = tip up."""
    import pandas as pd
    d, mi = pd.read_csv(path), pd.read_csv(midi_path)
    up = -ds.WRIST_DOWN * mir[ds.WRIST]                  # sign that makes + = tip up for this arm
    n = NAMES[ds.WRIST]
    rows = []
    for i, g in mi.groupby("stroke"):
        s = d[d.phase == f"stroke{i}"]
        t, dq = s.t.values, up * s[f"dq_{n}"].values
        th = g.t_from_stroke_start.values[0]
        before, after = (t > th - 0.03) & (t <= th), (t > th) & (t < th + 0.15)
        if before.any() and after.any():
            rows.append((g.k.values[0], g.velocity.values[0], dq[before].min(), dq[after].max()))
    r = pd.DataFrame(rows, columns=["k", "vel", "swing", "rebound"])
    print(f"per k, wrist {n} encoder (rad/s, + = tip up): swing just before the hit, rebound peak after it")
    for k, g in r.groupby("k"):
        print(f"  k={k:<5g} swing {g.swing.mean():+.2f} ± {g.swing.std():.2f}   rebound {g.rebound.mean():+.2f} ± "
              f"{g.rebound.std():.2f}   midi vel {g.vel.mean():.1f}")


def compare(path, stamp, a, mir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    d = pd.read_csv(path)
    d = d[d.phase == "stroke1"]
    if d.empty:
        return
    ref, T = stroke_ref(a.slow, mir, aim_m=a.aim / 100)
    r = sim_same(ref, T, a.k)
    fig, axs = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for col, (j, name) in enumerate([(3, "J4 elbow"), (ds.WRIST, f"J{ds.WRIST + 1} wrist pitch")]):
        n = NAMES[j]
        axs[0, col].plot(d.t, np.degrees(d[f"qcmd_{n}"]), "k--", lw=1, label="command")
        axs[0, col].plot(d.t, np.degrees(d[f"q_{n}"]), color="#2a78d6", lw=2, label="real")
        axs[0, col].plot(r["t"], np.degrees(r["q"][:, j]), color="#eb6834", lw=1.5, label="sim")
        axs[0, col].set_title(name); axs[0, col].set_ylabel("deg"); axs[0, col].legend(fontsize=8)
        axs[1, col].plot(d.t, d[f"tau_{n}"], color="#2a78d6", lw=2, label="real (measured)")
        axs[1, col].plot(r["t"], r["tau"][:, j], color="#eb6834", lw=1.5, label="sim (commanded)")
        axs[1, col].set_ylabel("N·m"); axs[1, col].set_xlabel("s"); axs[1, col].legend(fontsize=8)
        e_real = np.degrees(np.sqrt(np.mean((d[f"q_{n}"] - d[f"qcmd_{n}"]) ** 2)))
        e_sim = np.degrees(np.sqrt(np.mean((r["q"][:, j] - r["q_des"][:, j]) ** 2)))
        print(f"{name}: rms tracking error real {e_real:.2f} deg, sim {e_sim:.2f} deg")
    where = "onto the pad" if a.midi else "in the air"
    fig.suptitle(f"Stroke {where}, real vs MuJoCo (no pad) · {a.side} arm · k={a.k:g} (stroke 1) · "
                 f"{a.slow}x slower than sim")
    fig.tight_layout()
    out = OUT / f"{stamp}_stroke_vs_sim.png"
    fig.savefig(out, dpi=120)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
