"""scripts/live_stiffness.py --udp (k, CCI, both Myo IMUs) -> ROS topics.

  python live_stiffness.py --session 2026-10-15_S02 --udp 127.0.0.1:9100 --myo2     (scripts/, its own window)
  ros2 run ghost_teleop myo_bridge                                                  (--ros-args -p port:=9100)

Publishes /ghost/k (std_msgs/Float64, 0..1), /ghost/cci (Float64), /ghost/imu/forearm and /ghost/imu/upper
(geometry_msgs/QuaternionStamped, sensor -> world, world z up). live_stiffness keeps its L / H keys and plot.
"""
import socket

import rclpy
from geometry_msgs.msg import QuaternionStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64


class MyoBridge(Node):
    def __init__(self):
        super().__init__("myo_bridge")
        port = self.declare_parameter("port", 9100).value
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.k_pub = self.create_publisher(Float64, "/ghost/k", qos_profile_sensor_data)
        self.cci_pub = self.create_publisher(Float64, "/ghost/cci", qos_profile_sensor_data)
        self.imu_pub = {"imu": self.create_publisher(QuaternionStamped, "/ghost/imu/forearm", qos_profile_sensor_data),
                        "imu2": self.create_publisher(QuaternionStamped, "/ghost/imu/upper", qos_profile_sensor_data)}
        self.n = 0
        self.create_timer(0.002, self.poll)
        self.create_timer(5.0, self.report)
        self.get_logger().info(f"listening for live_stiffness.py on udp {port}")

    def poll(self):
        while True:
            try:
                p = self.sock.recv(256).decode().strip().split(",")
            except BlockingIOError:
                return
            try:
                if p[0] in self.imu_pub:
                    m = QuaternionStamped()
                    m.header.stamp = self.get_clock().now().to_msg()
                    m.header.frame_id = "myo_world_" + p[0]
                    m.quaternion.w, m.quaternion.x, m.quaternion.y, m.quaternion.z = map(float, p[2:6])
                    self.imu_pub[p[0]].publish(m)
                else:
                    self.k_pub.publish(Float64(data=float(p[1])))
                    self.cci_pub.publish(Float64(data=float(p[2])))
                self.n += 1
            except (ValueError, IndexError):
                pass

    def report(self):
        if self.n == 0:
            self.get_logger().warn("nothing from live_stiffness.py yet (running with --udp 127.0.0.1:<port>?)")
        self.n = 0


def main():
    rclpy.init()
    node = MyoBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
