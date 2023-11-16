
import os
import torch
torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from ament_index_python import get_package_share_directory


from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node



def generate_launch_description():
    this_package_name='yolov8_ros2'
    realsense_package_name = 'realsense2_camera'
    
    # Declare launch file parameters
    # DeclareLaunchArgument(
    #     'device',
    #     default_value='cuda:0',#torch_device,
    #     description='The device which the Yolo model should run on, default is "cuda:0", if available, otherwise it is set to "cpu"',
    # )
    

    # Launch Realsense camera launch file with aligned depth images publisher
    rs_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory(realsense_package_name), 'launch', 'rs_launch.py'
        )]), launch_arguments={'align_depth.enable': 'true'}.items()
    )
    
    
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
        rs_camera,
        yolov8_node,
    ])
