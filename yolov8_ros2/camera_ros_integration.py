# Basic ros functionalities
import rclpy
from rclpy.node import Node

# Executor and callback imports
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# Interfaces
from sensor_msgs.msg import Image, CameraInfo
from rclpy.qos import ReliabilityPolicy, QoSProfile
from cv_bridge import CvBridge
from std_msgs.msg import String

# Additional packages
import cv2, numpy as np, open3d as o3d
from camera_class import RealsenseVision

class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')
        
        # Init callback groups
        self.group_1 = MutuallyExclusiveCallbackGroup() # camera subscriber
        self.group_2 = MutuallyExclusiveCallbackGroup() # show current image timer
        
        # OpenCV2 bridge for interpreting ROS Image msg
        self.cv_bridge = CvBridge()
        self.cv_depth_image = None
        self.cv_color_image = None
        
        # Variables for storing the numpy arrays for the images
        self.np_depth_image = None
        self.np_color_image = None
        
        
        # Camera intrinsics
        self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        
        # Create publisher
        self.item_dict_publisher = self.create_publisher(String, 'item_dict', 10)
        self.item_dict_msg = String()
        
        # Create subscribers
        self.aligned_depth_image_subscriber = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self.aligned_depth_image_callback,
            QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE),
            callback_group=self.group_1
        )
        
        self.color_image_subscriber = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_image_callback,
            QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE),
            callback_group=self.group_1
        )
        
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE),
            callback_group=self.group_1
        )
        
        # Create timers
        self.timer_delay = 0.1
        self.camera_timer = self.create_timer(self.timer_delay, self.camera_callback, callback_group=self.group_1)
        self.publish_timer = self.create_timer(self.timer_delay, self.publish_callback, callback_group=self.group_2)
        
        
        
    def aligned_depth_image_callback(self, msg):
        try:
            # Convert ROS Image msg to cv2 mat then to np array
            self.cv_depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.np_depth_image = np.array(self.cv_depth_image, dtype=np.uint16)
            self.get_logger().debug(f'Depth encoding: {msg.encoding}') #-> 16UC1 = uint16
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')


    def color_image_callback(self, msg):
        try:
            # Convert ROS Image msg to cv2 mat then to np array
            self.cv_color_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.np_color_image = np.array(self.cv_color_image, dtype=np.uint8)
            self.get_logger().debug(f'Color encoding: {msg.encoding}') #-> rgb8 ( = bgr8?)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
            
            
    def camera_info_callback(self, msg):
        try:
            #self.get_logger().info('Camera_info_callback')
            # Set intrinsics in o3d object
            self.camera_intrinsics.set_intrinsics(msg.width,    #msg.width
                                                  msg.height,   #msg.height
                                                  msg.k[0],     #msg.K[0] -> fx
                                                  msg.k[4],     #msg.K[4] -> fy
                                                  msg.k[2],     #msg.K[2] -> cx
                                                  msg.k[5] )    #msg.K[5] -> cy
            self.get_logger().debug('Camera intrinsics have been set!')
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    
    
        
    def show_current_image(self):
        # Display CV2 image
        try:
            if self.cv2_image is not None:
                cv2.imwrite(f'/home/dreezy/dump/image{self.n}.png', self.cv2_image)
                self.get_logger().info(f'Saved image{self.n}')
                self.n += 1
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
        
        
        
    def shutdown_callback(self):
        self.get_logger().info("Shutting down...")
        
        

def main(args=None):
    rclpy.init(args=args)

    # Instansiate node class
    camera_node = CameraSubscriber()

    # Create executor
    executor = MultiThreadedExecutor()
    executor.add_node(camera_node)

    
    try:
        # Run executor
        executor.spin()
        
    except KeyboardInterrupt:
        pass
    
    finally:
        # Shutdown executor
        camera_node.shutdown_callback()
        executor.shutdown()


if __name__ == '__main__':
    main()