import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2

from line_follower.model import LineFollowerCNN

_WS = os.environ.get('ROS2_LF_WORKSPACE', '/workspaces/ros2-line-follower')



class InferenceNode(Node):

    def __init__(self):
        super().__init__('inference_node')

        self.bridge = CvBridge()

        self.model = LineFollowerCNN(num_classes=3)
        self.model.load_state_dict(torch.load(
            os.path.join(_WS, 'line_follower_model.pth'),
            map_location='cpu',
            weights_only=True,
        ))
        self.model.eval()

        self.class_names = ['center', 'left', 'right']
        self.uncertain_count = 0

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info('Inference node started, model loaded')


    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        resized = cv2.resize(cv_image, (64, 48))
        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probs, dim=1)
            confidence = confidence.item()
            predicted_idx = predicted_idx.item()

        prediction = self.class_names[predicted_idx]

        twist = Twist()
        if confidence < 0.8:
            self.uncertain_count += 1
            if self.uncertain_count < 10:
                prediction = 'uncertain-wait'
                # all zeros: pause briefly, line may reappear
            else:
                prediction = 'uncertain-search'
                twist.linear.x = 0.0
                twist.angular.z = 0.3
        else:
            self.uncertain_count = 0
            if prediction == 'center':
                twist.linear.x = 0.2
                twist.angular.z = 0.0
            elif prediction == 'left':
                twist.linear.x = 0.1
                twist.angular.z = 0.5
            elif prediction == 'right':
                twist.linear.x = 0.1
                twist.angular.z = -0.5

        self.publisher.publish(twist)
        self.get_logger().info(
            f'Prediction: {prediction} (conf {confidence:.2f}) → cmd_vel')


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()