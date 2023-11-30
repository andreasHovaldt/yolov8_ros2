# yolov8_ros2
Used in the 5th semester [Robotics LLM-Planner-for-Bimanual-object-mnipulation project](https://github.com/andreasHovaldt/LLM-Planner-for-Bimanual-object-manipulation). 
Package provides functionalities for using the Intel Realsense D435 camera combined with the yoloV8 object segmentation model implemented using ROS2.

## Required libraries
### Python3
```shell
pip install open3d
pip install ultralytics
```

## Quick Start
Install [colcon](https://docs.ros.org/en/humble/Tutorials/Colcon-Tutorial.html#install-colcon) and [rosdep](https://docs.ros.org/en/crystal/Installation/Linux-Install-Binary.html#installing-and-initializing-rosdep), then build this repository:

```shell
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/andreasHovaldt/yolov8_ros2.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source ~/ros2_ws/install/setup.bash
```

## Run code
```shell
ros2 launch yolov8_ros2 camera_yolo.launch.py
```
This launches the camera node, and the yolov8 node. 
The image prediction results of the yolov8 segmentation are published to the topic: ```/yolo/prediction/image```.
These results are also published in a json formatted string to the topic: ```/yolo/prediction/item_dict```.


