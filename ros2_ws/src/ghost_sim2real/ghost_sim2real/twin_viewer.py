"""MuJoCo twin of the arm: the measured pose (/joint_states) drawn solid, the controller's reference
(/ghost_controller/command) as a translucent blue ghost. Works the same for the sim and the real arm.

  ros2 run ghost_sim2real twin_viewer --ros-args -p model:=<scene xml>     (ghost.launch.xml viewer:=true)

The physics runs in the hardware plugin, not here: this window only draws /joint_states, so the viewer's
own simulation controls (pause, reset, joint / actuator sliders) do not move the arm. The panels are still
useful for rendering options, cameras and contact display; Tab / Shift+Tab hides them (param ui:=false).
"""
import os
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

GHOST_RGBA = np.array([0.25, 0.55, 1.0, 0.35], np.float32)


class Twin(Node):
    def __init__(self):
        super().__init__("twin_viewer")
        path = self.declare_parameter("model", "").value
        self.prefix = self.declare_parameter("ghost_prefix", "openarm_left").value
        self.ui = self.declare_parameter("ui", True).value
        if not os.path.exists(path):
            raise SystemExit(f"twin_viewer: model {path!r} not found")
        self.m = mujoco.MjModel.from_xml_path(path)
        self.d, self.dg = mujoco.MjData(self.m), mujoco.MjData(self.m)
        self.adr = {self.m.joint(j).name: self.m.jnt_qposadr[j] for j in range(self.m.njnt)}
        self.ghost_geoms = {g for g in range(self.m.ngeom)
                            if self.m.body(self.m.geom_bodyid[g]).name.startswith(self.prefix)}
        self.lock = threading.Lock()
        self.create_subscription(JointState, "/joint_states", self.on_state, qos_profile_sensor_data)
        self.create_subscription(JointState, "/ghost_controller/command", self.on_cmd, 10)

    def _apply(self, d, msg):
        with self.lock:
            for n, q in zip(msg.name, msg.position):
                if n in self.adr:
                    d.qpos[self.adr[n]] = q

    def on_state(self, msg):
        self._apply(self.d, msg)

    def on_cmd(self, msg):
        self._apply(self.dg, msg)

    def run(self):
        opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
        with mujoco.viewer.launch_passive(self.m, self.d, show_left_ui=self.ui, show_right_ui=self.ui) as v:
            v.cam.lookat[:] = [0.25, 0.1, 0.35]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.6, 200, -15
            while v.is_running() and rclpy.ok():
                with self.lock:
                    mujoco.mj_kinematics(self.m, self.d)
                    self.dg.qpos[:] = np.where(self.dg.qpos == 0, self.d.qpos, self.dg.qpos)
                    mujoco.mj_kinematics(self.m, self.dg)
                with v.lock():
                    scn = v.user_scn
                    scn.ngeom = 0
                    mujoco.mjv_addGeoms(self.m, self.dg, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
                    keep = 0
                    for i in range(scn.ngeom):
                        g = scn.geoms[i]
                        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid in self.ghost_geoms:
                            g.rgba[:] = GHOST_RGBA
                            if keep != i:
                                mujoco.mjv_initGeom(scn.geoms[keep], g.type, g.size, g.pos, g.mat.flatten(), g.rgba)
                                scn.geoms[keep].dataid, scn.geoms[keep].objtype = g.dataid, g.objtype
                            keep += 1
                    scn.ngeom = keep
                v.sync()
                time.sleep(1 / 60)


def main():
    rclpy.init()
    node = Twin()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
