from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'yolov8_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/' + package_name, [package_name+'/camera_class.py']),
        (os.path.join('share', package_name), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dreezy',
    maintainer_email='andreas.hovaldt@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_class = yolov8_ros2.camera_class:main',
            'vision_publisher = yolov8_ros2.vision_publisher:main',
            'camera_ros_integration = yolov8_ros2.camera_ros_integration:main',
        ],
    },
)
