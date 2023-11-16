import torch
torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from launch import LaunchDescription

from launch_ros.actions import Node



def generate_launch_description():
    this_package_name='yolov8_ros2'
    
    # Run the yolov8 node, with the set device
    yolov8_node = Node(
        package=this_package_name,
        executable='yolov8_node',
        #name='node2', # Default is name of executable
        output='screen',
        parameters=[
            {'device': f'{torch_device}'},
        ],
    )
    

    # Launch them all!
    return LaunchDescription([
        yolov8_node,
    ])
