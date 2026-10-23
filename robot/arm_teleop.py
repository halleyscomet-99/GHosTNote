"""Whole-arm Myo teleoperation of one OpenArm arm: the two Myo IMUs give the motion, forearm EMG the stiffness.

  # terminal A (scripts/): forearm Myo (EMG + IMU) + upper-arm Myo (IMU); k and both IMUs go out over UDP
  python live_stiffness.py --session 2026-10-22_S01 --ref-session 2026-10-15_S02 --udp 127.0.0.1:9100 --myo2
  # terminal B (GHOST/):
  python robot/arm_teleop.py --dry --viewer        # no robot: the MuJoCo arm follows you (try this first)
  python robot/arm_teleop.py                       # the real left arm on can1
  python robot/arm_teleop.py --stick --wrist 20    # with the drumstick, wrist pitched 20 deg down

Mapping: each Myo's quaternion gives its segment's rotation away from the "hang" pose (arm straight down,
palm on the thigh), in the operator's body frame (x forward, y left, z up). Which sensor axes lie along each
segment comes from two held poses: hang (gravity along the arm) and arm straight forward, thumb up. The
operator's right arm drives the robot's left arm as a mirror image (y -> -y), as if the robot faced you.
Inverse kinematics on the MuJoCo model finds J1-J5 whose upper-arm and forearm links have those
rotations; the Myo cannot see the wrist, so J6 / J7 hold --wrist. Stiffness: J4 and the wrist pitch joint
scaled by the live k (sim2real.real_gains); k = 1 while no EMG arrives.
Safety: targets low-passed and rate-limited (--vmax), clipped to the model's joint ranges; an IMU silent for
0.3 s freezes the target; the sim2real checks (tracking error, speed, torque) damp the arm and switch it off.
Ctrl-C returns the arm to where it rested. Log: data/robot/<time>_arm_teleop.csv
"""
import argparse
import collections
import csv
import json
import socket
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

import sim2real as s2r
from sim2real import ds, mujoco, KD, KD_DAMP, NAMES, TAU_PEAK, Arm, Model, move_ref, real_gains, track

TAGS = ("imu2", "imu")                 # upper arm, forearm
UP = np.array([0.0, 0.0, 1.0])
STALE = 0.3


def quat_mat(q):
    """Myo (w, x, y, z), sensor -> world (world z up) -> 3x3."""
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


class Stream:
    """Everything live_stiffness.py --udp sends: 't,k,cci' lines and 'imu[2],t,qw,qx,qy,qz' lines."""

    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.2)
        self.ks = collections.deque(maxlen=400)                              # (arrival, k)
        self.imu = {t: collections.deque(maxlen=200) for t in TAGS}          # (arrival, rotation matrix)
        self.n_new = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                p = self.sock.recv(256).decode().strip().split(",")
                if p[0] in TAGS:
                    self.imu[p[0]].append((time.monotonic(), quat_mat([float(v) for v in p[2:6]])))
                    self.n_new += 1
                else:
                    self.ks.append((time.monotonic(), float(p[1])))
            except (OSError, ValueError, IndexError):
                pass

    def k(self, window=0.2):
        t = time.monotonic()
        recent = [k for tk, k in self.ks if t - tk < window]
        return float(np.clip(np.mean(recent), 0, 1)) if recent and t - self.ks[-1][0] < 0.5 else None

    def latest(self, tag):
        """Newest rotation of that Myo, or None if it has been silent for STALE s."""
        b = self.imu[tag]
        return b[-1][1] if b and time.monotonic() - b[-1][0] < STALE else None

    def mean(self, tag, sec):
        t = time.monotonic()
        mats = [r for tr, r in self.imu[tag] if t - tr < sec]
        return Rotation.from_matrix(np.array(mats)).mean().as_matrix() if len(mats) >= 10 else None


def segment_axes(R_hang, R_fwd):
    """Sensor -> segment rotation M from the two poses, and the angle the segment turned between them (deg).
    Segment frame = body frame at hang: z up along the arm, x forward; raising the straight arm forward turns
    the segment's x axis to point up, so x = world-up seen by the sensor in the forward pose."""
    z = R_hang.T @ UP
    x = R_fwd.T @ UP
    turned = np.degrees(np.arccos(np.clip(x @ z, -1, 1)))
    x = x - z * (x @ z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z]), turned


def body_rot(M, R_hang, R_now):
    """The segment's rotation away from the hang pose, in the body frame."""
    return M.T @ R_hang.T @ R_now @ M


class IK:
    """J1-J5 that put the robot's upper-arm and forearm links at given rotations (damped least squares)."""

    def __init__(self, side):
        self.m = ds.build(pad_raise=-1.0)
        self.d = mujoco.MjData(self.m)
        self.qi, self.vi = ds.arm_idx(self.m)
        jid = [self.m.joint(n).id for n in ds.ARM]
        self.bodies = [self.m.jnt_bodyid[jid[2]], self.m.jnt_bodyid[jid[4]]]      # moved by J1-J3 / J1-J5
        self.lo, self.hi = self.m.jnt_range[jid].T
        self.R0 = self.fk(np.zeros(7))                                            # link rotations at hang

    def fk(self, q):
        self.d.qpos[:] = 0
        self.d.qpos[self.qi] = q
        mujoco.mj_kinematics(self.m, self.d)
        return [self.d.xmat[b].reshape(3, 3).copy() for b in self.bodies]

    def solve(self, targets, q, iters=4, lam=0.05):
        q = q.copy()
        jac = np.zeros((3, self.m.nv))
        for _ in range(iters):
            cur = self.fk(q)
            mujoco.mj_comPos(self.m, self.d)
            e, J = [], []
            for b, R_t, R_c in zip(self.bodies, targets, cur):
                e.append(Rotation.from_matrix(R_t @ R_c.T).as_rotvec())
                mujoco.mj_jacBody(self.m, self.d, None, jac, b)
                J.append(jac[:, self.vi[:5]].copy())
            e, J = np.concatenate(e), np.vstack(J)
            q[:5] += J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            q[:5] = np.clip(q[:5], self.lo[:5], self.hi[:5])
        return q, float(np.linalg.norm(e))


class Operator:
    """Myo rotations -> robot joint targets."""

    def __init__(self, stream, ik, mirror, wrist_q):
        self.s, self.ik, self.wrist_q = stream, ik, wrist_q
        self.S = np.diag([1.0, -1.0, 1.0]) if mirror else np.eye(3)
        self.q = np.r_[np.zeros(5), wrist_q]
        self.err = 0.0

    def calibrate(self):
        for tag, name in zip(TAGS, ("upper-arm", "forearm")):
            while self.s.latest(tag) is None:
                print(f"  waiting for the {name} Myo IMU over UDP (live_stiffness.py ... --myo2 running?)")
                time.sleep(1.0)
        poses = {}
        for name, cue in (("hang", "stand facing the robot, right arm hanging straight down, palm on your thigh"),
                          ("fwd", "raise the STRAIGHT arm forward to horizontal, thumb up")):
            input(f"\npose '{name}': {cue}. Hold it and press Enter ...")
            time.sleep(1.5)
            poses[name] = {t: self.s.mean(t, 1.2) for t in TAGS}
            if any(r is None for r in poses[name].values()):
                raise SystemExit("too few IMU samples during the pose; is the Myo streaming?")
        self.cal = {}
        for tag, name in zip(TAGS, ("upper arm", "forearm")):
            M, turned = segment_axes(poses["hang"][tag], poses["fwd"][tag])
            self.cal[tag] = (M, poses["hang"][tag])
            print(f"  {name}: turned {turned:.0f} deg between the poses (expect ~90)"
                  + ("  <-- redo: arm not straight down / forward, or Myo slipping" if abs(turned - 90) > 25 else ""))
        print("  put your arm back down; the robot starts from your hang pose")

    def target(self):
        """Joint targets for the current Myo rotations, or None if an IMU is stale."""
        Rs = [self.s.latest(t) for t in TAGS]
        if any(R is None for R in Rs):
            return None
        goals = []
        for tag, R, R0 in zip(TAGS, Rs, self.ik.R0):
            M, R_hang = self.cal[tag]
            D = self.S @ body_rot(M, R_hang, R) @ self.S
            goals.append(D @ R0)
        self.q, self.err = self.ik.solve(goals, self.q)
        self.q[5:] = self.wrist_q
        return self.q.copy()


def dry_run(op, stream, viewer):
    v = None
    if viewer:
        import mujoco.viewer
        v = mujoco.viewer.launch_passive(op.ik.m, op.ik.d)
    print("\ndry run: no robot. Ctrl-C stops.")
    try:
        while v is None or v.is_running():
            q = op.target()
            k = stream.k()
            if q is None:
                print("  IMU stale")
            else:
                print(" ".join(f"{NAMES[i]} {np.degrees(q[i]):+6.1f}" for i in range(5))
                      + f"   fit {np.degrees(op.err):4.1f} deg   k {'--' if k is None else f'{k:.2f}'}")
            if v is not None:
                op.ik.fk(q if q is not None else op.q)
                v.sync()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


def live(arm, model, op, stream, a, use_ff, log):
    """Follow the operator until Ctrl-C (returns normally) or a safety stop (raises)."""
    dt = 1.0 / s2r.HZ
    alpha = dt / (a.lp + dt)
    vmax = np.r_[np.full(3, a.vmax), np.full(4, a.vmax * 1.5)]
    dq_max = 1.5 * vmax.max() + 3.0
    qd = arm.state()[0][:7].copy()
    qf, q_goal, dqd = qd.copy(), qd.copy(), np.zeros(7)
    q_grip = arm.state()[0][7] if s2r.GRIP["q"] is None else s2r.GRIP["q"]
    seen, stale_since, t0, last_print = -1, None, time.monotonic(), 0.0
    print("\nlive: the arm follows you. Ctrl-C to stop (it then goes back to rest).")
    try:
        while True:
            tick = time.monotonic()
            t = tick - t0
            if stream.n_new != seen:
                seen = stream.n_new
                qt = op.target()
                if qt is None:
                    if stale_since is None:
                        stale_since = t
                        print("  IMU stale: holding")
                else:
                    stale_since, q_goal = None, qt
            qf += alpha * (q_goal - qf)
            step = np.clip(qf - qd, -vmax * dt, vmax * dt)
            qd += step
            dqd += 0.2 * (step / dt - dqd)
            k = stream.k()
            kp, kd = real_gains(1.0 if k is None else k)
            ff = model.tau(qd) if use_ff else np.zeros(7)
            arm.command(np.r_[kp, s2r.GRIP["kp"]], np.r_[kd, KD[7]], np.r_[qd, q_grip], np.r_[dqd, 0], np.r_[ff, 0])
            q, dq, tau, tm, _ = arm.state()
            log.writerow(["live", f"{t:.4f}", *(f"{x:.5f}" for x in (*qd, *q[:7], *dq[:7], *tau[:7], *ff)),
                          int(tm.max()), "" if k is None else f"{k:.3f}"])
            bad = (np.abs(q[:7] - qd) > s2r.err_limit(kp)) | (np.abs(dq[:7]) > dq_max) | (np.abs(tau[:7]) > 0.8 * TAU_PEAK[:7])
            if bad.any():
                i = int(np.argmax(bad))
                raise RuntimeError(f"safety stop on {NAMES[i]}: q {np.degrees(q[i]):+.1f} deg vs target "
                                   f"{np.degrees(qd[i]):+.1f}, dq {dq[i]:+.2f}, tau {tau[i]:+.2f}")
            if t - last_print > 1.0:
                last_print = t
                print("  " + " ".join(f"{NAMES[i]} {np.degrees(qd[i]):+6.1f}" for i in range(5))
                      + f"   k {'-- (EMG missing: stiff)' if k is None else f'{k:.2f}'}")
            time.sleep(max(0.0, dt - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        print("\nstopping")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--can", default="can_left", help="can_left / can_right (robot/udev/80-openarm-can.rules)")
    ap.add_argument("--side", default="left", choices=["left", "right"], help="which robot arm is on --can")
    ap.add_argument("--human-arm", default="right", choices=["left", "right"], help="arm wearing the Myos")
    ap.add_argument("--model", default="v1", choices=["v1", "v2"])
    ap.add_argument("--fd", action="store_true")
    ap.add_argument("--udp", type=int, default=9100)
    ap.add_argument("--wrist", type=float, default=0.0, help="wrist pitch held during teleop, deg (+ = tip down)")
    ap.add_argument("--vmax", type=float, default=2.0, help="shoulder speed limit, rad/s (elbow / wrist: 1.5x)")
    ap.add_argument("--lp", type=float, default=0.04, help="low-pass time constant on the targets, s")
    ap.add_argument("--move-time", type=float, default=6.0)
    ap.add_argument("--stick", action="store_true")
    ap.add_argument("--squeeze", type=float, default=2.0)
    ap.add_argument("--no-ff", action="store_true")
    ap.add_argument("--skip-rest-check", action="store_true",
                    help="start even if a joint rests > 25 deg from zero (the arm then moves it back slowly)")
    ap.add_argument("--dry", action="store_true", help="no robot: print the joint targets")
    ap.add_argument("--viewer", action="store_true", help="--dry: show the MuJoCo arm following you")
    a = ap.parse_args()

    ds.use_model(a.model, a.side)
    if not a.stick:
        ds.STICK_M = 1e-4
    mir = s2r.MIRROR[a.side]
    wrist_q = np.zeros(2)
    wrist_q[ds.WRIST - 5] = ds.WRIST_DOWN * np.radians(a.wrist) * mir[ds.WRIST]
    stream = Stream(a.udp)
    op = Operator(stream, IK(a.side), a.human_arm != a.side, wrist_q)
    op.calibrate()
    if a.dry:
        return dry_run(op, stream, a.viewer)

    check = s2r.OUT / f"gravity_check_{a.model}_{a.side}.json"
    model = Model()
    use_ff = not a.no_ff and check.exists() and json.load(open(check))["all_signs_ok"]
    if use_ff:
        model.calibrate(check)
    print(f"gravity feedforward: {'ON' if use_ff else 'OFF'}")
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
    s2r.OUT.mkdir(parents=True, exist_ok=True)
    path = s2r.OUT / f"{time.strftime('%Y%m%d_%H%M%S')}_arm_teleop.csv"
    f = open(path, "w", newline="")
    log = csv.writer(f)
    log.writerow(["phase", "t"] + [f"{p}_{n}" for p in ("qcmd", "q", "dq", "tau", "ff") for n in NAMES[:7]]
                 + ["tmos_max", "k"])
    kp, kd = real_gains(1.0)
    if a.stick:
        s2r.GRIP["q"], s2r.GRIP["kp"] = arm.take_stick(a.squeeze)
        arm.refresh()
        q_rest = arm.state()[0][:7].copy()      # the arm moves while the stick goes in
    arm.oa.enable_all()
    time.sleep(0.1)
    arm.oa.recv_all(2000)
    try:
        track(arm, model, lambda t: (q_rest, np.zeros(7), np.zeros(7)), 1.5, kp * 0.3, kd, False, log, "ramp")
        q0 = op.target()
        if q0 is None:
            raise RuntimeError("IMU stale before start")
        print(f"\nhold still: moving to your pose in {a.move_time:.0f} s")
        track(arm, model, move_ref(q_rest, q0, a.move_time), a.move_time, kp, kd, use_ff, log, "go")
        live(arm, model, op, stream, a, use_ff, log)
        q_now = arm.state()[0][:7]
        track(arm, model, move_ref(q_now, q_rest, a.move_time), a.move_time, kp, kd, use_ff, log, "back")
        track(arm, model, lambda t: (q_rest, np.zeros(7), np.zeros(7)), 1.0, kp * 0.3, kd, False, log, "release")
    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"\n!! {e or 'Ctrl-C'} -> damping 1.5 s (support the arm), then motors off")
        for _ in range(int(1.5 * 200)):
            arm.command(np.zeros(8), KD_DAMP, arm.state()[0])
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


if __name__ == "__main__":
    main()
