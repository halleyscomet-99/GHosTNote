"""OpenArm bring-up: read every motor, then move one joint at a time, slowly.

  python robot/arm_bringup.py read --can can0                 # motors stay OFF, only reads q/dq/tau/temp
  python robot/arm_bringup.py move --can can0 --joint 6       # wiggle J6 by ±8 deg, 2 slow cycles, back home
  python robot/arm_bringup.py move --can can0 --joint 4 --amp 5 --dir + --period 4
  python robot/arm_bringup.py move --can can0 --joint all     # J7 -> J1 one after another, then gripper
  python robot/arm_bringup.py grip --can can1 --no-fd --squeeze 2   # gripper only: take a stick, squeeze, report

Uses the official openarm_can binding (MIT mode, same ids/gains as openarm_gello's zmq driver).
Safety: gains ramp in from zero holding the pose the arm rests in; the target moves at most
`--amp` from there; any joint with tracking error > 0.3 rad, |dq| > 3 rad/s or |tau| > 80 % of the
motor's peak makes every motor switch to pure damping for 1 s and then disables. Ctrl-C does the same.
Logs every tick to data/robot/<time>_<can>_<cmd>.csv.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

try:
    import openarm_can as oa
except ImportError:          # the constants below are enough for the sim / ROS tools; only Arm needs the binding
    oa = None

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "grip"]
TYPES = ["DM8009", "DM8009", "DM4340", "DM4340", "DM4310", "DM4310", "DM4310"]
SEND = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
RECV = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
KP = np.array([70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0, 8.0])     # openarm_zmq defaults + gripper
KD = np.array([3.5, 3.0, 2.5, 2.5, 1.0, 0.8, 0.8, 0.5])
KD_DAMP = np.array([5.0, 5.0, 4.0, 4.0, 2.0, 1.5, 1.5, 0.5])
TAU_PEAK = np.array([40, 40, 27, 27, 7, 7, 7, 7], float)
HZ = 200
GRIP_OPEN, GRIP_CLOSED = -1.0, 0.0       # gripper motor rad (openarm_zmq: -1.047 = fully open, 0 = closed)
GRIP_KP = 8.0                            # squeeze stiffness: target sits squeeze/GRIP_KP past the contact
MAX_ERR, MAX_DQ, TAU_FRAC = 0.3, 3.0, 0.8


class Arm:
    def __init__(self, can, fd):
        if oa is None:
            raise SystemExit("openarm_can Python binding missing: build openarm_can/python (build.sh) into this venv")
        self.oa = oa.OpenArm(can, fd)
        self.oa.init_arm_motors([getattr(oa.MotorType, t) for t in TYPES], SEND, RECV)
        self.oa.init_gripper_motor(oa.MotorType.DM4310, 0x08, 0x18)
        self.oa.set_callback_mode_all(oa.CallbackMode.STATE)

    def refresh(self, n=5):
        for _ in range(n):
            self.oa.refresh_all()
            time.sleep(0.01)
            self.oa.recv_all(2000)

    def state(self):
        m = list(self.oa.get_arm().get_motors()) + list(self.oa.get_gripper().get_motors())  # copies: re-fetch
        return (np.array([x.get_position() for x in m]), np.array([x.get_velocity() for x in m]),
                np.array([x.get_torque() for x in m]), np.array([x.get_state_tmos() for x in m]),
                np.array([x.get_state_trotor() for x in m]))

    def grip(self, q, kp, kd=0.3):
        """Command only the gripper (MIT): kp * (q - q_now) is the squeeze once the fingers touch."""
        self.oa.get_gripper().mit_control_all([oa.MITParam(float(kp), float(kd), float(q), 0.0, 0.0)])
        self.oa.recv_all(1000)

    def grip_for(self, seconds, q, kp):
        for _ in range(int(seconds * HZ)):
            self.grip(q, kp)
            time.sleep(1.0 / HZ)

    def take_stick(self, squeeze):
        """Gripper only: open, wait for the stick, close gently until the fingers touch it, then aim
        squeeze / GRIP_KP past that contact so the fingers press with ~squeeze N·m. Returns (q_target, kp)."""
        self.oa.get_gripper().enable_all()
        time.sleep(0.05)
        self.oa.recv_all(2000)
        self.grip_for(1.0, GRIP_OPEN, 4.0)
        input("put the stick between the fingers (hold it), then press Enter to close... ")
        self.grip_for(1.5, GRIP_CLOSED, 3.0)               # gentle close: stops on the stick
        q_contact = self.state()[0][7]
        target = q_contact + squeeze / GRIP_KP
        self.grip_for(1.0, target, GRIP_KP)
        q, _, tau, *_ = self.state()
        print(f"stick touched at {np.degrees(q_contact):+.1f} deg; holding {np.degrees(q[7]):+.1f} deg, "
              f"squeeze {tau[7]:+.2f} N·m (asked {squeeze:.1f})")
        return target, GRIP_KP

    def give_stick(self):
        input("hold the stick, press Enter to open the gripper... ")
        self.grip_for(1.0, GRIP_OPEN, 4.0)
        self.oa.get_gripper().disable_all()
        self.oa.recv_all(2000)

    def command(self, kp, kd, q, dq=None, tau=None):
        dq = np.zeros(8) if dq is None else dq
        tau = np.zeros(8) if tau is None else tau
        p = [oa.MITParam(float(kp[i]), float(kd[i]), float(q[i]), float(dq[i]), float(tau[i])) for i in range(8)]
        self.oa.get_arm().mit_control_all(p[:7])
        self.oa.get_gripper().mit_control_all(p[7:])
        self.oa.recv_all(1000)


def print_state(arm):
    q, dq, tau, tm, tr = arm.state()
    print(f"{'motor':6}{'q (rad)':>10}{'q (deg)':>10}{'dq':>8}{'tau':>8}{'T_mos':>7}{'T_rot':>7}")
    for i in range(8):
        alive = tm[i] > 0 or tr[i] > 0 or q[i] != 0
        print(f"{NAMES[i]:6}{q[i]:+10.3f}{np.degrees(q[i]):+10.1f}{dq[i]:+8.2f}{tau[i]:+8.2f}{tm[i]:7.0f}{tr[i]:7.0f}"
              + ("" if alive else "   <-- no reply"))
    return q


def run_move(arm, a, log):
    joints = list(range(7))[::-1] + [7] if a.joint == "all" else [7 if a.joint == "grip" else int(a.joint) - 1]
    arm.refresh()
    q0, _, _, tm, tr = arm.state()
    dead = [NAMES[i] for i in range(8) if tm[i] == 0 and tr[i] == 0 and q0[i] == 0]
    if dead:
        raise SystemExit(f"no reply from {dead}: run `read` first; not moving (start pose would be wrong)")
    print("start pose (deg):", " ".join(f"{NAMES[i]} {np.degrees(q0[i]):+.1f}" for i in range(8)))
    amp = np.radians(a.amp)
    sign = {"both": 0, "+": 1, "-": -1}[a.dir]
    kp, kd = KP * a.gain, KD.copy()
    dt = 1.0 / HZ
    t_start = time.monotonic()

    def tick(target, kp_now, kd_now):
        arm.command(kp_now, kd_now, target)
        q, dq, tau, tm, tr = arm.state()
        t = time.monotonic() - t_start
        log.writerow([f"{t:.4f}", *np.round(target, 5), *np.round(q, 5), *np.round(dq, 4), *np.round(tau, 4),
                      *tm.astype(int)])
        err = np.abs(q - target)
        bad = (err > MAX_ERR) | (np.abs(dq) > MAX_DQ) | (np.abs(tau) > TAU_FRAC * TAU_PEAK)
        if kp_now.max() > 0.5 * kp.max() and bad.any():
            i = int(np.argmax(bad))
            raise RuntimeError(f"safety stop on {NAMES[i]}: err {err[i]:.2f} rad, dq {dq[i]:+.2f}, tau {tau[i]:+.2f}")
        return q, tau

    def hold_for(seconds, target, ramp_from=None):
        n = int(seconds * HZ)
        for k in range(n):
            s = (k + 1) / n if ramp_from is not None else 1.0
            tick(target, kp * s, kd)
            time.sleep(dt)

    arm.oa.enable_all()
    time.sleep(0.1)
    arm.oa.recv_all(2000)
    try:
        print("ramping gains in (1.5 s), holding the start pose")
        hold_for(1.5, q0, ramp_from=0)
        for j in joints:
            print(f"\n{NAMES[j]}: ±{a.amp:.0f} deg" if sign == 0 else f"\n{NAMES[j]}: {'+' if sign > 0 else '-'}{a.amp:.0f} deg",
                  f"period {a.period:.1f} s x {a.cycles}")
            n = int(a.period * a.cycles * HZ)
            peak_tau, peak_err = 0.0, 0.0
            for k in range(n):
                ph = 2 * np.pi * k / (a.period * HZ)
                off = amp * np.sin(ph) if sign == 0 else sign * amp * (1 - np.cos(ph)) / 2
                target = q0.copy()
                target[j] += off
                q, tau = tick(target, kp, kd)
                peak_tau = max(peak_tau, abs(tau[j]))
                peak_err = max(peak_err, abs(q[j] - target[j]))
                time.sleep(dt)
            hold_for(0.5, q0)
            q, *_ = arm.state()
            print(f"  done: moved back to {np.degrees(q[j]):+.1f} deg (start {np.degrees(q0[j]):+.1f}), "
                  f"peak |tau| {peak_tau:.2f} N·m, peak tracking error {np.degrees(peak_err):.1f} deg")
        print("\nreleasing gains slowly back at the start pose")
        for k in range(int(1.0 * HZ)):
            s = 1 - (k + 1) / HZ
            tick(q0, kp * s, kd)
            time.sleep(dt)
    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"\n!! {e or 'Ctrl-C'} -> damping for 1 s, then motors off")
        for _ in range(HZ):
            q, *_ = arm.state()
            arm.command(np.zeros(8), KD_DAMP, q)
            time.sleep(dt)
    finally:
        arm.oa.disable_all()
        arm.oa.recv_all(2000)
        print("motors disabled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["read", "move", "grip"])
    ap.add_argument("--can", default="can_left", help="can_left / can_right (robot/udev/80-openarm-can.rules)")
    ap.add_argument("--no-fd", action="store_true", help="classic CAN 2.0 instead of CAN-FD")
    ap.add_argument("--joint", default="6", help="1..7, 8/grip, or all")
    ap.add_argument("--amp", type=float, default=8.0, help="degrees")
    ap.add_argument("--dir", default="both", choices=["both", "+", "-"])
    ap.add_argument("--period", type=float, default=3.0, help="seconds per cycle")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--gain", type=float, default=1.0, help="scale on the default kp")
    ap.add_argument("--squeeze", type=float, default=2.0, help="grip: squeeze torque, N·m (motor peak 7)")
    a = ap.parse_args()
    if a.amp > 20:
        ap.error("--amp above 20 deg is not allowed in bring-up")

    arm = Arm(a.can, not a.no_fd)
    if a.cmd == "read":
        arm.refresh()
        print(f"{a.can} (CAN-FD {not a.no_fd}), motors OFF")
        print_state(arm)
        return
    if a.cmd == "grip":                 # gripper only; the arm stays off
        try:
            target, kp = arm.take_stick(a.squeeze)
            print("pull / twist the stick to test the grip; readings every 0.5 s, Ctrl-C to finish")
            while True:
                arm.grip_for(0.5, target, kp)
                q, _, tau, tm, _ = arm.state()
                print(f"  gripper {np.degrees(q[7]):+6.1f} deg  squeeze {tau[7]:+.2f} N·m  T {tm[7]:.0f} C")
        except KeyboardInterrupt:
            print()
        finally:
            try:
                arm.give_stick()
            finally:
                arm.oa.disable_all()
        return
    out = ROOT / "data" / "robot"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d_%H%M%S')}_{a.can}_move_{a.joint}.csv"
    with open(path, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["t"] + [f"{p}_{n}" for p in ("qcmd", "q", "dq", "tau", "tmos") for n in NAMES])
        run_move(arm, a, log)
    print(f"log: {path}")


if __name__ == "__main__":
    main()
