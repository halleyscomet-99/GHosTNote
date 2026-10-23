"""Whole-arm Myo teleop through ghost_controller: robot/arm_teleop.py's calibration, mirror and IK, streamed.

  ros2 launch ghost_bringup ghost.launch.xml teleop:=true [hardware_type:=real]
  ros2 run ghost_teleop myo_bridge
  ros2 run ghost_teleop imu_teleop                    # in a terminal: two held poses, then the arm follows

The two-pose calibration (hang, straight arm forward) aligns the armbands; the arm then moves to your pose
along a 6 s minimum-jerk path and follows /ghost/imu/* from there (the controller low-passes and
rate-limits the stream). Stiffness k comes to the controller straight from /ghost/k. Ctrl-C: the arm goes
back to where it rested.
"""
import argparse
import collections
import threading
import time

import numpy as np
from geometry_msgs.msg import QuaternionStamped
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState

import ghost_teleop  # noqa: F401
import arm_teleop as at  # noqa: E402
import sim2real as s2r  # noqa: E402
from ghost_sim2real.client import ArmClient  # noqa: E402

TOPIC = {"imu": "/ghost/imu/forearm", "imu2": "/ghost/imu/upper"}


class RosImu:
    """arm_teleop.Stream's IMU side (latest / mean rotation per Myo) fed by ROS topics."""

    def __init__(self, node):
        self.imu = {t: collections.deque(maxlen=200) for t in at.TAGS}
        self.n_new = 0
        self.lock = threading.Lock()
        for tag in at.TAGS:
            node.create_subscription(QuaternionStamped, TOPIC[tag], lambda m, tag=tag: self._on(tag, m),
                                     qos_profile_sensor_data)

    def _on(self, tag, m):
        q = m.quaternion
        with self.lock:
            self.imu[tag].append((time.monotonic(), Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()))
            self.n_new += 1

    def latest(self, tag):
        with self.lock:
            b = self.imu[tag]
            return b[-1][1] if b and time.monotonic() - b[-1][0] < at.STALE else None

    def mean(self, tag, sec):
        t = time.monotonic()
        with self.lock:
            mats = [r for tr, r in self.imu[tag] if t - tr < sec]
        return Rotation.from_matrix(np.array(mats)).mean().as_matrix() if len(mats) >= 10 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="left", choices=["left", "right"], help="robot arm")
    ap.add_argument("--human-arm", default="right", choices=["left", "right"], help="arm wearing the Myos")
    ap.add_argument("--wrist", type=float, default=0.0, help="wrist pitch held during teleop, deg (+ = tip down)")
    ap.add_argument("--move-time", type=float, default=6.0)
    a = ap.parse_args()
    at.ds.use_model("v1", a.side)
    mir = s2r.MIRROR[a.side]
    wrist_q = np.zeros(2)
    wrist_q[at.ds.WRIST - 5] = at.ds.WRIST_DOWN * np.radians(a.wrist) * mir[at.ds.WRIST]

    c = ArmClient(a.side)
    imu = RosImu(c)
    op = at.Operator(imu, at.IK(a.side), a.human_arm != a.side, wrist_q)
    c.wait_ready()
    q_rest = c.qd.copy()
    op.calibrate()
    q0 = op.target()
    if q0 is None:
        raise SystemExit("IMU stale after calibration")
    print(f"\nhold still: the arm moves to your pose in {a.move_time:.0f} s")
    msg = JointState()
    msg.name = c.joints
    try:
        c.play(s2r.move_ref(q_rest, q0, a.move_time), a.move_time, tag="go")
        print("live: the arm follows you. Ctrl-C to stop.")
        seen, last_print = -1, 0.0
        while True:
            if imu.n_new != seen:
                seen = imu.n_new
                q = op.target()
                if q is not None:
                    msg.position = [float(x) for x in q]
                    c.target_pub.publish(msg)
            if c.fault():
                raise RuntimeError(c.status)
            if time.monotonic() - last_print > 1.0:
                last_print = time.monotonic()
                print("  " + " ".join(f"J{i + 1} {np.degrees(c.qd[i]):+6.1f}" for i in range(5))
                      + f"   fit {np.degrees(op.err):4.1f} deg   k {c.k:.2f}")
            time.sleep(0.002)
    except KeyboardInterrupt:
        c.debounce_ctrl_c()
        print("\nstopping: back to rest (Ctrl-C again aborts; the controller then holds)")
        time.sleep(0.4)                     # the controller holds once the stream stops
        q_now = c.qd.copy()
        c.play(s2r.move_ref(q_now, q_rest, a.move_time), a.move_time, tag="back")
    except RuntimeError as e:
        print(f"\n!! {e}")
    finally:
        c.close()


if __name__ == "__main__":
    main()
