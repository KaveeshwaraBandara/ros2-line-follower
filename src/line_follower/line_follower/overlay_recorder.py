import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np

from line_follower.model import LineFollowerCNN


class OverlayRecorder(Node):

    def __init__(self):
        super().__init__('overlay_recorder')

        self.bridge = CvBridge()

        self.model = LineFollowerCNN(num_classes=3)
        self.model.load_state_dict(torch.load(
            '/workspaces/ros2-line-follower/line_follower_model.pth',
            map_location='cpu'))
        self.model.eval()

        self.class_names = ['center', 'left', 'right']

        self.frames = []
        self.frame_size = None

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        self.get_logger().info('Overlay recorder started. Ctrl+C to save video.')

    
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
            prediction = self.class_names[predicted_idx.item()]

        annotated = self.draw_overlay(cv_image, prediction, confidence)

        if self.frame_size is None:
            h, w = annotated.shape[:2]
            self.frame_size = (w, h)
        self.frames.append(annotated)


    def draw_overlay(self, image, prediction, confidence):
        img = image.copy()
        h, w = img.shape[:2]

        colors = {
            'center': (0, 255, 0),
            'left':   (255, 150, 0),
            'right':  (0, 150, 255),
        }
        color = colors.get(prediction, (128, 128, 128))

        cv2.rectangle(img, (0, 0), (w, 60), (0, 0, 0), -1)

        label = f'{prediction.upper()}  ({confidence*100:.0f}%)'
        cv2.putText(img, label, (15, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cx, cy = w // 2, h - 40
        if prediction == 'left':
            cv2.arrowedLine(img, (cx, cy), (cx - 80, cy), color, 6, tipLength=0.3)
        elif prediction == 'right':
            cv2.arrowedLine(img, (cx, cy), (cx + 80, cy), color, 6, tipLength=0.3)
        elif prediction == 'center':
            cv2.arrowedLine(img, (cx, cy), (cx, cy - 80), color, 6, tipLength=0.3)

        return img


    
    def save_video(self):
        if not self.frames:
            self.get_logger().warn('No frames recorded, nothing to save.')
            return

        output_path = '/workspaces/ros2-line-follower/overlay_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10
        writer = cv2.VideoWriter(output_path, fourcc, fps, self.frame_size)

        for frame in self.frames:
            writer.write(frame)
        writer.release()

        self.get_logger().info(
            f'Saved {len(self.frames)} frames to {output_path}')


def main(args=None):
    rclpy.init(args=args)
    node = OverlayRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_video()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()