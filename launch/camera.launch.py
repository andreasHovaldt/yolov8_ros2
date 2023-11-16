
import os

from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource



def generate_launch_description():
    realsense_package_name = 'realsense2_camera'
    
    # Launch Realsense camera launch file with aligned depth images publisher
    rs_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory(realsense_package_name), 'launch', 'rs_launch.py'
        )]), launch_arguments={'align_depth.enable': 'true'}.items()
    )
    
    # Launch them all!
    return LaunchDescription([
        rs_camera,
    ])
