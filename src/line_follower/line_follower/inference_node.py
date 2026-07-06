import os
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np

from line_follower.model import LineFollowerCNN

_WS = os.environ.get('ROS2_LF_WORKSPACE', '/workspaces/ros2-line-follower')

# --- Proportional steering gains (tune in sim) ---
# The model outputs a steering angle in degrees: 0 = straight, + = line to the
# right, - = line to the left. In ROS, +angular.z spins the robot left (CCW),
# so we steer with the NEGATED angle to turn toward the line.
KP = 1.2            # rad/s of yaw per rad of steering-angle error
V_MAX = 0.2         # m/s on a straight line
ANGLE_MAX = 35.0    # deg; error magnitude at which we slow down the most
SLOW = 0.6          # fraction of V_MAX shed at full steering angle
MAX_YAW = 1.0       # rad/s clamp

# --- Line-loss detection & search behavior ---
# The regression model ALWAYS outputs an angle, even with no line in view, so we
# gate the controller on an explicit "is a line actually present?" check.
DARK_DELTA = 60         # a line pixel is this much darker than the floor's median
LINE_MIN_FRAC = 0.004   # min fraction of dark pixels in the lower ROI to count as a line
WAIT_TICKS = 5          # brief pause (frames) before we start spinning to search
SEARCH_YAW = 0.4        # rad/s in-place spin while searching for the line


class InferenceNode(Node):

    def __init__(self):
        super().__init__('inference_node')

        self.bridge = CvBridge()

        self.model = LineFollowerCNN()
        self.model.load_state_dict(torch.load(
            os.path.join(_WS, 'line_follower_model.pth'),
            map_location='cpu',
            weights_only=True,
        ))
        self.model.eval()
        self.angle_scale = self.model.ANGLE_SCALE

        self.lost_count = 0     # consecutive frames with no line detected
        self.search_dir = 1.0   # which way to spin when searching (+CCW / -CW)

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info('Inference node started (regression), model loaded')

    def _line_present(self, bgr):
        """True if a dark line occupies enough of the lower part of the frame.

        Uses a relative-darkness test (pixels far darker than the floor's
        median) rather than Otsu, which would hallucinate a line on a uniform
        floor. Resolution-independent (fraction based).
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        h = gray.shape[0]
        roi = gray[int(h * 0.5):, :]              # lower half: where the line should be
        med = float(np.median(roi))
        dark_frac = float(np.mean(roi.astype(np.int16) < (med - DARK_DELTA)))
        return dark_frac >= LINE_MIN_FRAC

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        twist = Twist()

        if not self._line_present(cv_image):
            # Line lost: pause briefly (it may reappear), then spin in place
            # toward the side we last saw the line, to re-acquire it.
            self.lost_count += 1
            if self.lost_count <= WAIT_TICKS:
                state = 'lost-wait'
            else:
                twist.angular.z = SEARCH_YAW * self.search_dir
                state = 'lost-search'
            self.publisher.publish(twist)
            self.get_logger().info(f'{state} ({self.lost_count}) -> v=0.00 yaw={twist.angular.z:+.2f}')
            return

        # Line present: reset loss counter and steer with the CNN's angle.
        self.lost_count = 0

        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (64, 48))
        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            angle_deg = self.model(tensor).item() * self.angle_scale

        # Remember which way to search if we lose the line next: if the line is
        # to the right (angle > 0) we should later spin clockwise (yaw < 0).
        self.search_dir = -1.0 if angle_deg > 0 else 1.0

        # --- Proportional control ---
        angle_rad = math.radians(angle_deg)
        yaw = -KP * angle_rad
        yaw = max(-MAX_YAW, min(MAX_YAW, yaw))

        slowdown = min(abs(angle_deg) / ANGLE_MAX, 1.0) * SLOW
        speed = V_MAX * (1.0 - slowdown)

        twist.linear.x = speed
        twist.angular.z = yaw
        self.publisher.publish(twist)

        self.get_logger().info(
            f'angle={angle_deg:+.1f} deg -> v={speed:.2f} yaw={yaw:+.2f}')


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
