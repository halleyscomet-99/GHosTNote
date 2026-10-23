"""robot/sim2real.py on ROS 2: the same goto / stroke / contact runs, sent to ghost_controller.

Start the stack first (ros2 launch ghost_bringup ghost.launch.xml [hardware_type:=real]), then e.g.
  ros2 run ghost_sim2real run goto                          # to the drumming pose, Enter, back
  ros2 run ghost_sim2real run stroke --slow 4               # one stroke in the air at 1/4 speed (+ sim comparison)
  ros2 run ghost_sim2real run stroke --slow 1 --ks 1,0.5,0.2,0.08 --n 8 --aim 1 --midi --stick
  ros2 run ghost_sim2real run contact --stick               # hold the touch pose to place the pad
  ros2 run ghost_sim2real run stroke --teleop --n 20 --slow 1 --period 2 --midi --stick
                                                            # Mode 1 teleop: k live from /ghost/k (myo_bridge)
Every run starts and ends where the reference was (the controller holds the arm where it rested).
Log: data/robot/<time>_ros_<cmd>.csv (same columns as sim2real.py) and, for strokes, the sim comparison.
"""
import argparse
import threading
import time
from types import SimpleNamespace

import numpy as np

from ghost_sim2real import ROOT  # noqa: F401  (puts GHOST/sim, robot, scripts on sys.path)
import drum_sim as ds  # noqa: E402
import sim2real as s2r  # noqa: E402
from ghost_sim2real.client import ArmClient  # noqa: E402

OUT = s2r.OUT


def hold(c, q, T, tag):
    c.play(lambda t: (q, np.zeros(7), np.zeros(7)), T, tag=tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["goto", "stroke", "contact"])
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--frac", type=float, default=1.0, help="goto: fraction of the way to POSE")
    ap.add_argument("--move-time", type=float, default=6.0)
    ap.add_argument("--slow", type=float, default=4.0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--ks", default="", help="several k values: each block plays every k once, shuffled")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aim", type=float, default=1.0, help="cm the planned tip path goes below the head")
    ap.add_argument("--midi", action="store_true")
    ap.add_argument("--stick", action="store_true", help="take a stick in the real gripper first")
    ap.add_argument("--squeeze", type=float, default=2.0)
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--teleop", action="store_true",
                    help="stroke: k is the operator's live /ghost/k (launch teleop:=true + myo_bridge), not --k/--ks")
    ap.add_argument("--period", type=float, default=0.0, help="stroke: seconds from one stroke start to the next")
    a = ap.parse_args()
    ds.use_model("v1", a.side)
    mir = s2r.MIRROR[a.side]
    start = ds.POSE.copy()
    start[3] += 0.10
    start[ds.WRIST] = ds.WRIST_DOWN * -0.45
    contact = ds.POSE.copy()
    contact[ds.WRIST] = ds.WRIST_DOWN * 0.20

    c = ArmClient(a.side)
    c.wait_ready()
    q_rest = c.qd.copy()
    target = {"goto": q_rest + a.frac * (ds.POSE * mir - q_rest), "stroke": start * mir, "contact": contact * mir}[a.cmd]
    print("rest  (deg):", " ".join(f"J{i + 1} {np.degrees(q_rest[i]):+6.1f}" for i in range(7)))
    print("target(deg):", " ".join(f"J{i + 1} {np.degrees(target[i]):+6.1f}" for i in range(7)))
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = OUT / f"{stamp}_ros_{a.cmd}.csv"
    c.start_log(path)
    seq, a.k = [a.k], a.k
    try:
        if a.stick:
            c.take_stick(a.squeeze)
        if not a.teleop:
            c.set_k(1.0)
        c.play(s2r.move_ref(q_rest, target, a.move_time), a.move_time, tag="go")
        if a.cmd in ("stroke", "contact") and c.sim_pad(True):
            print("(sim: pad switched on)")
        hold(c, target, 1.0, "hold")
        if a.cmd in ("goto", "contact"):
            print("\nholding" + (": put the pad under the stick so the head just touches the tip" if a.cmd == "contact"
                                 else "") + ". Press Enter to go back.")
            done = threading.Event()
            threading.Thread(target=lambda: (input(), done.set()), daemon=True).start()
            while not done.is_set():
                hold(c, target, 0.25, a.cmd)
        else:
            ref, T = s2r.stroke_ref(a.slow, mir, aim_m=a.aim / 100)
            ks = [float(x) for x in a.ks.split(",")] if a.ks else [a.k]
            rng = np.random.default_rng(a.seed)
            seq = [float(k) for _ in range(a.n) for k in (rng.permutation(ks) if len(ks) > 1 else ks)]
            a.k = seq[0]
            midi = s2r.start_midi() if a.midi else None
            starts = []
            for s in range(len(seq)):
                t_next = time.monotonic() + max(a.period, T)
                if a.teleop:
                    seq[s] = round(float(c.k), 3)          # the operator's k as the stroke starts
                else:
                    c.set_k(seq[s])
                    time.sleep(0.02)
                print(f"stroke {s + 1}/{len(seq)}  (k={seq[s]:g}{' live' if a.teleop else ''}, "
                      f"{a.slow}x slower than sim, aim {a.aim:.1f} cm)")
                starts.append(midi[1]() if midi else 0.0)
                c.play(ref, T, tag=f"stroke{s + 1}")
                if a.period > T:
                    hold(c, target, t_next - time.monotonic(), f"wait{s + 1}")
            a.k = seq[0]
            if not a.teleop:
                c.set_k(1.0)
            if midi:
                time.sleep(0.3)
                s2r.score_midi(midi, starts, seq, T, a.slow, OUT / f"{stamp}_ros_midi.csv")
            hold(c, target, 0.5, "hold2")
        c.sim_pad(False)
        c.play(s2r.move_ref(target, q_rest, a.move_time), a.move_time, tag="back")
        hold(c, q_rest, 0.5, "rest")
    except (KeyboardInterrupt, RuntimeError) as e:
        c.debounce_ctrl_c()
        print(f"\n!! {e or 'Ctrl-C'}")
        if not c.fault():
            print("   the controller holds the last reference; ros2 service call /ghost_controller/reset to resume")
    finally:
        if a.stick:
            try:
                c.give_stick()
            except (KeyboardInterrupt, EOFError):
                pass
        c.stop_log()
        c.close()
        print(f"log: {path}")
    if a.cmd == "stroke":
        if a.midi and (OUT / f"{stamp}_ros_midi.csv").exists():
            s2r.score_wrist(path, OUT / f"{stamp}_ros_midi.csv", mir)
        if not a.no_compare:
            s2r.compare(path, stamp, SimpleNamespace(slow=a.slow, aim=a.aim, k=seq[0], midi=a.midi, side=a.side), mir)


if __name__ == "__main__":
    main()
