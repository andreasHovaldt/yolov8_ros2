from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'yolov8_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Andreas Hovaldt',
    maintainer_email='andreas.hovaldt@gmail.com',
    description='ROS2 Package for object segmentation using the YoloV8 segmentation model and the Intel Realsense D435',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov8_node = yolov8_ros2.yolov8_node:main',
        ],
    },
)
