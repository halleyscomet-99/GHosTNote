"""ArmClient: everything a script needs from ghost_controller, plus a sim2real.py-format CSV log.

  c = ArmClient(); c.wait_ready()
  c.set_k(0.3); c.play(ref, T, tag="stroke1")       # ref(t) -> (q, dq, ddq), sampled every 4 ms
  c.close()
"""
import csv
import signal
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray, String
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "J7"]
DT = 0.004


class ArmClient(Node):
    def __init__(self, side="left", controller="ghost_controller", k_topic="/ghost/k"):
        if not rclpy.ok():      # Ctrl-C stays a KeyboardInterrupt, so a script can still send the arm back
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        super().__init__(f"ghost_client_{int(time.time() * 1000) % 100000}")
        self.joints = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
        self.q = self.dq = self.tau = None
        self.qd = self.dqd = self.ff = None
        self.k = np.nan
        self.grip = (np.nan, np.nan)
        self.status = ""
        self.log, self.tag = None, None
        self.stamp, self.t0 = 0.0, None             # controller time of the last command; phase start
        self.lock = threading.Lock()
        self.create_subscription(JointState, "/joint_states", self._on_js, qos_profile_sensor_data)
        self.create_subscription(JointState, f"/{controller}/command", self._on_cmd, 10)
        self.create_subscription(String, f"/{controller}/status", lambda m: setattr(self, "status", m.data), 10)
        self.traj_pub = self.create_publisher(JointTrajectory, f"/{controller}/trajectory", 10)
        self.target_pub = self.create_publisher(JointState, f"/{controller}/target", qos_profile_sensor_data)
        self.k_pub = self.create_publisher(Float64, k_topic, 10)
        self.grip_pub = self.create_publisher(Float64MultiArray, f"/{controller}/gripper", 10)
        self.reset_cli = self.create_client(Trigger, f"/{controller}/reset")
        self.pad_cli = self.create_client(SetBool, f"/ghost_{side}_arm/set_pad")
        self.ex = SingleThreadedExecutor()          # rclpy's multi-threaded one drops to a few msgs/s
        self.ex.add_node(self)
        threading.Thread(target=self.ex.spin, daemon=True).start()

    # ---------------- state ----------------
    def _on_js(self, m):
        idx = {n: i for i, n in enumerate(m.name)}
        if not all(j in idx for j in self.joints):
            return
        ii = [idx[j] for j in self.joints]
        with self.lock:
            self.q = np.array([m.position[i] for i in ii])
            self.dq = np.array([m.velocity[i] for i in ii]) if m.velocity else np.zeros(7)
            self.tau = np.array([m.effort[i] for i in ii]) if m.effort else np.zeros(7)

    def _on_cmd(self, m):
        stamp = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        with self.lock:
            self.stamp = stamp
            if self.t0 is None:
                self.t0 = stamp
            self.qd, self.dqd, self.ff = (np.array(m.position[:7]), np.array(m.velocity[:7]), np.array(m.effort[:7]))
            self.k = m.position[7]
            if len(m.position) > 8:
                self.grip = (m.position[8], m.effort[8])
            if self.log is not None and self.q is not None:
                row = [self.tag, stamp - self.t0, *self.qd, *self.q, *self.dq, *self.tau, *self.ff, 0]
                self.log.writerow([f"{x:.5f}" if isinstance(x, float) else x for x in row])

    def wait_ready(self, timeout=10.0):
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            # also wait for DDS discovery: a trajectory sent before the controller's subscription matched is lost
            matched = self.traj_pub.get_subscription_count() > 0 and self.target_pub.get_subscription_count() > 0
            if self.q is not None and self.qd is not None and "RAMP" not in self.status and self.status and matched:
                return
            time.sleep(0.05)
        raise RuntimeError("no /joint_states + /ghost_controller/command yet (is ghost.launch.xml running, "
                           f"controller active?) status: {self.status!r}")

    def fault(self):
        return "FAULT" in self.status

    # ---------------- logging ----------------
    def start_log(self, path):
        self.f = open(path, "w", newline="")
        self.log = csv.writer(self.f)
        self.log.writerow(["phase", "t"] + [f"{p}_{n}" for p in ("qcmd", "q", "dq", "tau", "ff") for n in NAMES]
                          + ["tmos_max"])

    def stop_log(self):
        with self.lock:
            self.log = None
        self.f.close()

    # ---------------- commands ----------------
    def set_k(self, k):
        self.k_pub.publish(Float64(data=float(k)))

    def gripper(self, q, kp=np.nan):
        self.grip_pub.publish(Float64MultiArray(data=[float(q), float(kp)]))

    def sim_pad(self, on):
        """MuJoCo twin only: let the pad collide (on) or not; the real arm has no such service."""
        if not self.pad_cli.wait_for_service(timeout_sec=0.5):
            return False
        self.pad_cli.call(SetBool.Request(data=bool(on)))
        return True

    def reset(self):
        self.reset_cli.wait_for_service(timeout_sec=2.0)
        return self.reset_cli.call(Trigger.Request())

    def play(self, ref, T, tag="move", wait=True):
        """Send ref(t) for t in [0, T] as one trajectory; with wait, block until it has played (or a fault)."""
        msg = JointTrajectory()
        msg.joint_names = self.joints
        for t in np.arange(0.0, T + 1e-9, DT):
            q, dq, ddq = ref(t)
            p = JointTrajectoryPoint()
            p.positions, p.velocities, p.accelerations = [list(map(float, x)) for x in (q, dq, ddq)]
            p.time_from_start.sec, p.time_from_start.nanosec = int(t), int((t % 1) * 1e9)
            msg.points.append(p)
        with self.lock:                             # t restarts at 0 every phase, as in sim2real.track
            self.tag, self.t0 = tag, self.stamp
        self.traj_pub.publish(msg)
        if not wait:
            return
        t_sent = time.monotonic()
        while time.monotonic() < t_sent + T + 0.05:
            if self.fault():
                raise RuntimeError(self.status)
            if "rejected" in self.status and time.monotonic() > t_sent + 0.2:   # status is published at 10 Hz
                raise RuntimeError(self.status)
            time.sleep(0.01)

    def grip_for(self, seconds, q, kp):
        self.gripper(q, kp)
        time.sleep(seconds)

    def take_stick(self, squeeze, kp_hold=8.0, open_q=-1.0, closed_q=0.0):
        """arm_bringup.Arm.take_stick through the controller: open, wait for the stick, close gently until the
        fingers stop on it, then aim squeeze / kp_hold past that contact. Returns the measured squeeze (N m)."""
        self.grip_for(1.0, open_q, 4.0)
        input("put the stick between the fingers (hold it), then press Enter to close... ")
        self.grip_for(1.5, closed_q, 3.0)
        q_contact = self.grip[0]
        self.grip_for(1.0, q_contact + squeeze / kp_hold, kp_hold)
        print(f"stick touched at {np.degrees(q_contact):+.1f} deg; squeeze {self.grip[1]:+.2f} N m (asked {squeeze:.1f})")
        return self.grip[1]

    def give_stick(self, open_q=-1.0):
        input("hold the stick, press Enter to open the gripper... ")
        self.grip_for(1.0, open_q, 4.0)

    @staticmethod
    def debounce_ctrl_c(window=0.5):
        """After a Ctrl-C: ignore the duplicate SIGINTs `ros2 run` forwards within `window` s, so the clean-up
        (sending the arm back) runs; a later Ctrl-C still interrupts it."""
        t_first = time.monotonic()

        def handler(sig, frame):
            if time.monotonic() - t_first > window:
                signal.signal(signal.SIGINT, signal.default_int_handler)
                raise KeyboardInterrupt
        signal.signal(signal.SIGINT, handler)

    def close(self):
        self.ex.shutdown()
        self.destroy_node()
        rclpy.try_shutdown()
