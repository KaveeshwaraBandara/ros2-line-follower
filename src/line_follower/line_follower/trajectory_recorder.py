import os
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_WS = os.environ.get('ROS2_LF_WORKSPACE', '/workspaces/ros2-line-follower')


class TrajectoryRecorder(Node):

    def __init__(self):
        super().__init__('trajectory_recorder')

        self.x_positions = []
        self.y_positions = []

        self.subscription = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        self.get_logger().info('Trajectory recorder started. Ctrl+C to save plot.')

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.x_positions.append(x)
        self.y_positions.append(y)

    
    def save_plot(self):
        if not self.x_positions:
            self.get_logger().warn('No data recorded, nothing to plot.')
            return

        plt.figure(figsize=(10, 6))

        # Reference line track (matches line_track.world)
        plt.plot([0, 3], [0, 0], 'k--', linewidth=2, label='Line segment 1', alpha=0.5)
        plt.plot([3, 5.6], [0, 1.45], 'k--', linewidth=2, label='Line segment 2', alpha=0.5)

        # Robot's actual path
        plt.plot(self.x_positions, self.y_positions,
                 'b-', linewidth=2, label='Robot path')
        plt.plot(self.x_positions[0], self.y_positions[0],
                 'go', markersize=12, label='Start')
        plt.plot(self.x_positions[-1], self.y_positions[-1],
                 'rs', markersize=12, label='End')

        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.title('Line Follower Robot Trajectory')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

        output_path = os.path.join(_WS, 'trajectory.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        self.get_logger().info(f'Trajectory saved to {output_path}')


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_plot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()