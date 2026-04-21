import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from .base import BaseSimEnv


class ROS2Env(BaseSimEnv, Node):
    def __init__(self, node_name="yogacara_ros2"):
        BaseSimEnv.__init__(self)
        Node.__init__(self, node_name)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.latest_odom = None
        self.action_map = {"UP": (0.5, 0), "DOWN": (-0.5, 0), "LEFT": (0, 0.5), "RIGHT": (0, -0.5), "STAY": (0, 0)}

    def _odom_cb(self, msg):
        self.latest_odom = msg

    def reset(self):
        self.latest_odom = None
        return self._observe()

    def step(self, action: str):
        vx, vyaw = self.action_map.get(action, (0, 0))
        twist = Twist()
        twist.linear.x = vx
        twist.angular.z = vyaw
        self.cmd_pub.publish(twist)
        rclpy.spin_once(self, timeout_sec=0.2)
        return self._observe(), -0.1, False, {}

    def _observe(self):
        if not self.latest_odom:
            return {"grid_view": [0] * 9, "pos": (0, 0), "step": 0}
        p = self.latest_odom.pose.pose.position
        return {"grid_view": [0] * 9, "pos": (round(p.x, 1), round(p.y, 1)), "step": 0}
