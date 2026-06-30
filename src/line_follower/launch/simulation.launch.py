import os
import subprocess
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def _ensure_xvfb():
    """Start Xvfb on :99 if no display is available and Xvfb is installed."""
    if os.environ.get('DISPLAY'):
        return
    try:
        subprocess.Popen(
            ['Xvfb', ':99', '-screen', '0', '1280x720x24'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.environ['DISPLAY'] = ':99'
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    except FileNotFoundError:
        pass  # Xvfb not installed; caller will get a rendering error from gzserver


def launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory('line_follower')

    world_name = LaunchConfiguration('world_name').perform(context)
    world_file = os.path.join(pkg_share, 'worlds', f'{world_name}.world')
    gui = LaunchConfiguration('gui').perform(context)

    urdf_file = os.path.join(pkg_share, 'urdf', 'robot.urdf.xacro')
    robot_description = xacro.process_file(urdf_file).toxml()
    urdf_output_path = '/tmp/line_follower_robot.urdf'
    with open(urdf_output_path, 'w') as f:
        f.write(robot_description)

    set_display = SetEnvironmentVariable('DISPLAY', os.environ.get('DISPLAY', ':99'))
    set_softgl = SetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE', '1')

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True
        }]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch', 'gazebo.launch.py',
            )
        ]),
        launch_arguments={
            'world': world_file,
            'verbose': 'false',
            'gui': gui
        }.items()
    )

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', urdf_output_path,
            '-entity', 'line_follower'
        ],
        output='screen'
    )

    camera_node = Node(
        package='line_follower',
        executable='camera_node',
        output='screen',
        parameters=[{
            'use_sim_time': True
        }]
    )

    return [set_display, set_softgl, robot_state_publisher_node, gazebo, spawn_robot, camera_node]


def generate_launch_description():
    _ensure_xvfb()

    return LaunchDescription([
        DeclareLaunchArgument(
            'world_name',
            default_value='line_track',
            description='World file name (no .world extension) from the worlds/ directory. '
                        'Options: line_track, rectangle, triangle, corridor_maze, curved, zigzag, messy'
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='false',
            description='Open Gazebo GUI window on the host desktop. '
                        'Requires: run "xhost +local:" on the host first.'
        ),
        OpaqueFunction(function=launch_setup),
    ])
