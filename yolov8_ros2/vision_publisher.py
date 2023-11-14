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
from std_msgs.msg import String

# Additional packages
import cv2
#import sys
# Add the path to the directory containing camera_class.py
#sys.path.append("/path/to/directory/containing/camera_class.py")
from camera_class import RealsenseVision


class VisionPublisher(Node):

    def __init__(self):
        super().__init__('vision_publisher')
        
        # Init callback groups
        self.group_1 = MutuallyExclusiveCallbackGroup() # camera timer
        self.group_2 = MutuallyExclusiveCallbackGroup() # publish timer
        
        # Create publisher
        self.item_dict_publisher = self.create_publisher(String, 'item_dict', 10)
        self.item_dict_msg = String()
        
        # Create timer
        self.timer_delay = 0.1
        self.camera_timer = self.create_timer(self.timer_delay, self.camera_callback, callback_group=self.group_1)
        self.publish_timer = self.create_timer(self.timer_delay, self.publish_callback, callback_group=self.group_2)
    
        
    
    def camera_callback(self):
        self.vision_object = RealsenseVision(object_detection_model="yolov8s-seg.pt", 
                                    realsense_product_line="D400",
                                    depth_range=3,
                                    debug_mode=False)
        self.vision_object.streaming_loop()
        
        
    def publish_callback(self):
        
        try:
            self.get_logger().info(f"Detected keys: {self.vision_object.item_dict.keys()}")
            self.item_dict_msg.data = (f"{self.vision_object.item_dict_str}")
            self.item_dict_publisher.publish(self.item_dict_msg)

        except:
            self.item_dict_msg.data = "Object detection failure..."
            self.get_logger().info(f"{self.item_dict_msg.data}")
            self.item_dict_publisher.publish(self.item_dict_msg)
            
    
        
    def shutdown_callback(self):
        self.get_logger().info("Shutting down...")
        



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
        
    except KeyboardInterrupt:
        pass
    
    finally:
        # Shutdown executor
        vision_node.shutdown_callback()
        executor.shutdown()


if __name__ == '__main__':
    main()
