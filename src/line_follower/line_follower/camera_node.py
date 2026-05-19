import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np

class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.get_logger().info('Camera node started, waiting for images...')

    def image_callback(self,msg):
        self.get_logger().info(
            f'Received image: {msg.width}x{msg.height}, encoding: {msg.encoding}'
        )
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.get_logger().info(
            f'Converted to numpy array, shape: {cv_image.shape}, dtype: {cv_image.dtype}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()