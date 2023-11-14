# Basic ros functionalities
import rclpy
from rclpy.node import Node

# Executor and callback imports
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# Interfaces
from sensor_msgs.msg import Image
from rclpy.qos import ReliabilityPolicy, QoSProfile
from cv_bridge import CvBridge

# Additional packages
import cv2


class VisionPublisher(Node):

    def __init__(self):
        super().__init__('vision_publisher')
        
        # Init callback groups
        self.group_1 = MutuallyExclusiveCallbackGroup() # Image_raw subscriber
        self.group_2 = MutuallyExclusiveCallbackGroup() # Timer
        
        # Create timer
        self.timer_delay = 0.5
        self.timer = self.create_timer(self.timer_delay, self.timer_callback, callback_group=self.group_2)
    
        
    def timer_callback(self):
        self.get_logger().info("Timer callback")
        



def main(args=None):
    rclpy.init(args=args)

    # Instansiate node class
    vision_node = VisionPublisher()

    # Create executor
    executor = MultiThreadedExecutor()
    executor.add_node(vision_node)

    
    try:
        # Run executor
        executor.spin()
    finally:
        # Shutdown executor
        executor.shutdown()
        vision_node.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()
