import os
import math
import signal
import shutil
import subprocess
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import cv2

from line_follower.model import LineFollowerCNN

_WS = os.environ.get('ROS2_LF_WORKSPACE', '/workspaces/ros2-line-follower')


class OverlayRecorder(Node):

    def __init__(self):
        super().__init__('overlay_recorder')

        self.bridge = CvBridge()

        self.model = LineFollowerCNN()
        self.model.load_state_dict(torch.load(
            os.path.join(_WS, 'line_follower_model.pth'),
            map_location='cpu',
            weights_only=True,
        ))
        self.model.eval()
        self.angle_scale = self.model.ANGLE_SCALE

        self.frames = []
        self.frame_size = None

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        self.get_logger().info('Overlay recorder started. Ctrl+C to save video.')


    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Match the training pipeline: RGB, 64x48, /255, CHW.
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (64, 48))
        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            angle_deg = self.model(tensor).item() * self.angle_scale

        annotated = self.draw_overlay(cv_image, angle_deg)

        if self.frame_size is None:
            h, w = annotated.shape[:2]
            self.frame_size = (w, h)
        self.frames.append(annotated)


    def draw_overlay(self, image, angle_deg):
        img = image.copy()
        h, w = img.shape[:2]

        # green when near-straight, warmer as the steering angle grows
        mag = min(abs(angle_deg) / 35.0, 1.0)
        color = (0, int(255 * (1 - mag)), int(255 * mag))  # BGR: green -> red

        cv2.rectangle(img, (0, 0), (w, 60), (0, 0, 0), -1)

        side = 'STRAIGHT' if abs(angle_deg) < 3 else ('RIGHT' if angle_deg > 0 else 'LEFT')
        label = f'{angle_deg:+.1f} deg  {side}'
        cv2.putText(img, label, (15, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # arrow from the bottom-center, rotated by the steering angle
        # (+ = right, - = left; 0 = straight up)
        cx, cy = w // 2, h - 40
        length = 80
        a = math.radians(angle_deg)
        tip = (int(cx + length * math.sin(a)), int(cy - length * math.cos(a)))
        cv2.arrowedLine(img, (cx, cy), tip, color, 6, tipLength=0.3)

        return img


    
    FPS = 10

    def save_video(self):
        if not self.frames:
            self.get_logger().warn('No frames recorded, nothing to save.')
            return

        output_path = os.path.join(_WS, 'overlay_video.mp4')

        # OpenCV's bundled FFmpeg here has NO H.264 encoder, so cv2.VideoWriter
        # can only emit mpeg4 (MPEG-4 Part 2) -- playable in VLC but not in
        # browsers/Foxglove. Pipe raw frames to the standalone ffmpeg and encode
        # H.264 + yuv420p (universally playable). Fall back to cv2 if ffmpeg is
        # unavailable.
        if self._save_with_ffmpeg(output_path):
            return
        self._save_with_opencv(output_path)

    def _save_with_ffmpeg(self, output_path):
        if shutil.which('ffmpeg') is None:
            self.get_logger().warn('ffmpeg not found; falling back to cv2 (mpeg4).')
            return False

        w, h = self.frame_size
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}', '-r', str(self.FPS), '-i', 'pipe:0',
            # force even dimensions (libx264 + yuv420p requires them)
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', output_path,
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for frame in self.frames:
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h))
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            rc = proc.wait()
        except Exception as e:  # noqa: BLE001 - want any failure to fall back
            self.get_logger().warn(f'ffmpeg encode failed ({e}); falling back to cv2.')
            return False

        if rc != 0:
            self.get_logger().warn(f'ffmpeg exited {rc}; falling back to cv2.')
            return False

        self.get_logger().info(
            f'Saved {len(self.frames)} frames (H.264) to {output_path}')
        return True

    def _save_with_opencv(self, output_path):
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.FPS, self.frame_size)
        for frame in self.frames:
            writer.write(frame)
        writer.release()
        self.get_logger().info(
            f'Saved {len(self.frames)} frames (mpeg4) to {output_path}')


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main(args=None):
    # The GUI stops this node with SIGTERM (proc.terminate()), while a manual
    # run uses Ctrl+C (SIGINT). Treat SIGTERM like Ctrl+C so the video is always
    # saved on shutdown -- otherwise a stale overlay_video.mp4 from a previous
    # run would be left in place.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    rclpy.init(args=args)
    node = OverlayRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_video()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()