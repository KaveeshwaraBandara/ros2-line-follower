from setuptools import find_packages, setup

package_name = 'line_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf',
        ['urdf/robot.urdf.xacro']),
        ('share/' + package_name + '/launch',
        ['launch/simulation.launch.py']),
        ('share/' + package_name + '/worlds',
        ['worlds/line_track.world']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='79425575+KaveeshwaraBandara@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_node = line_follower.camera_node:main',
            'inference_node = line_follower.inference_node:main',
            'trajectory_recorder = line_follower.trajectory_recorder:main',
            'overlay_recorder = line_follower.overlay_recorder:main',
        ],
    },
)
