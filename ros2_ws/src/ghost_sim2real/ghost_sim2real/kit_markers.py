"""The drum kit of the MuJoCo scene as RViz markers (/ghost/kit, visualization_msgs/MarkerArray, frame world).

  ros2 run ghost_sim2real kit_markers --ros-args -p model:=<scene xml>     (ghost.launch.xml rviz:=true)

Reads the same scene the twin simulates, so RViz always shows the kit sim/ded70_layout.yaml describes. The
robot's world frame is the URDF's "world" (both models put the arm bases at the same place).
"""
import os

import mujoco
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray


def kit_geoms(m):
    """Geoms of the drum (bodies pad* / ded70*, plus world geoms named ded70_*), planes excluded."""
    for g in range(m.ngeom):
        body, name = m.body(m.geom_bodyid[g]).name, m.geom(g).name
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if body.startswith(("pad", "ded70")) or name.startswith("ded70"):
            yield g


def markers(path):
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out = MarkerArray()
    for i, g in enumerate(kit_geoms(m)):
        mk = Marker()
        mk.header.frame_id = "world"
        mk.ns, mk.id, mk.action = "ded70", i, Marker.ADD
        size, t = m.geom_size[g], m.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            mk.type = Marker.CYLINDER
            mk.scale.x = mk.scale.y = 2 * size[0]
            mk.scale.z = 2 * size[1]
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            mk.type = Marker.CYLINDER
            mk.scale.x = mk.scale.y = 2 * size[0]
            mk.scale.z = 2 * (size[1] + size[0])
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            mk.type = Marker.CUBE
            mk.scale.x, mk.scale.y, mk.scale.z = (2 * size).tolist()
        else:
            continue
        p, q = d.geom_xpos[g], np.zeros(4)
        mujoco.mju_mat2Quat(q, d.geom_xmat[g])
        mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = map(float, p)
        mk.pose.orientation.w, mk.pose.orientation.x, mk.pose.orientation.y, mk.pose.orientation.z = map(float, q)
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = map(float, m.geom_rgba[g])
        out.markers.append(mk)
    return out


class KitMarkers(Node):
    def __init__(self):
        super().__init__("kit_markers")
        path = self.declare_parameter("model", "").value
        if not os.path.exists(path):
            raise SystemExit(f"kit_markers: model {path!r} not found")
        self.msg = markers(path)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)   # RViz started later still gets it
        self.pub = self.create_publisher(MarkerArray, "/ghost/kit", qos)
        self.pub.publish(self.msg)
        self.create_timer(2.0, lambda: self.pub.publish(self.msg))
        self.get_logger().info(f"{len(self.msg.markers)} kit markers on /ghost/kit")


def main():
    rclpy.init()
    node = KitMarkers()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
