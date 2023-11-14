from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolov8_ros2',
            executable='vision_publisher',
            output='screen'),
    ])