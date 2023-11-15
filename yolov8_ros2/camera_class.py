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

import pyrealsense2 as rs
import numpy as np
import cv2, time, json
from ultralytics import YOLO
from pyntcloud import PyntCloud
import open3d as o3d
import matplotlib.pyplot as plt



    

#####################################################
#-------------------- FUNCTIONS --------------------#
#####################################################

def filter_depth_object_img(img, starting_mask, deviation): #TODO: I need explanation from alfredo -Dreez
    """
    parameters 
    img: np depth image 
    deviation: the deviation allowed
    returns 
    filteref image 

    Works by removing pixels too far from the median value
    """
    mdv = np.median(img[starting_mask]) #median depth value
    u_lim = mdv + mdv*deviation #upper limit
    
    uidx = (img >= u_lim)
    
    #we stack the two masks and then takes the max in the new axis
   
    #print(f"pixels removed {np.sum(starting_mask)-np.sum(uidx)}")
    out_img = img
    zero_img = np.zeros_like(img)
    out_img[uidx] = zero_img[uidx]
    return out_img

def display_object_pointclouds(object_point_clouds):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for pointcloud in object_point_clouds:
        nppc = np.asarray(pointcloud.points)
        ax.scatter(nppc[:,0],nppc[:,2],-nppc[:,1])
    plt.show()


#######################################################################
#-------------------- VISION CLASS MIGRATION CODE --------------------#
#######################################################################

class RealsenseVision(Node): # Currently works for the D400 product line, other Intel Realsense product lines might not work
    '''
    RealSenseVision class for capturing frames from a RealSense camera, performing object detection,
    and displaying the results with additional features such as background removal and point cloud visualization.

    Note:
    - Currently works for the D400 product line; other Intel RealSense product lines might not be compatible.
    - The object detection model used should be a segmentation model (e.g., <model_name>-seg.pt)

    Parameters:
    - object_detection_model (str): Object detection model to be used (e.g., YoloV8x-seg.pt).
    - realsense_product_line (str, optional): WIP, NOT USED. Product line of the RealSense camera. Default is 'D400'.
    - depth_range (int, optional): Depth range for background removal. Default is 3.
    - debug_mode (bool, optional): Enable debug mode for additional visualizations. Default is False.

    Attributes:
    - depth_range (int): Depth range for background removal.
    - debug_mode (bool): Flag to enable or disable debug mode.
    - model: Object detection model initialized with the provided model path.
    - pipeline: RealSense pipeline for capturing frames.
    - pipeline_config: Configuration for the RealSense pipeline.
    - decimation_filter: RealSense decimation filter for data processing.
    - rs_intrinsics: Open3D camera intrinsic parameters for depth-to-color alignment.
    - align: RealSense align object for aligning depth images to color images.
    - clipping_distance (float): Clipping distance calculated based on depth range.

    Methods:
    - streaming_loop: Runs a continuous loop to capture, process, and display RealSense frames.
    
    Private methods (Should not be called manually):
    - __init__: Initializes the RealSenseVision object.
    - __initialize_camera_stream: Initializes the RealSense camera stream and configuration.
    - __set_camera_intrinsics: Sets the camera intrinsics for depth-to-color alignment.
    - __get_clipping_distance: Calculates the clipping distance based on the depth range.

    Usage:
    ```
    # Example usage
    realsense_vision = RealsenseVision( # Create vision object
        object_detection_model='yolov8s-seg.pt',
        realsense_product_line='D400',
        depth_range=3,
        debug_mode=True
    )
    realsense_vision.streaming_loop() # Start vision stream
    ```
    '''
    #----------------------#
    # CLASS INITIALIZATION #
    #----------------------#
    def __init__(self, object_detection_model: str, realsense_product_line='D400', depth_range=3, debug_mode=False):
        ############# ROS INT #############
        super().__init__('camera_subscriber')
        self.group_1 = MutuallyExclusiveCallbackGroup() # camera subscriber
        self.group_2 = MutuallyExclusiveCallbackGroup() # show current image timer
        
        # CV bidge
        self.cv_bridge = CvBridge()
        self.cv_depth_image = None
        self.cv_color_image = None
        
        # For storing images
        self.np_depth_image = None
        self.np_color_image = None
        
        # Camera intrinsics init
        self.camera_intrinsics = None
        
        # Create timer
        self.camera_timer = self.create_timer(0.04, self.camera_callback, callback_group=self.group_2) # 25 hz
        
        # Create publisher
        self.item_dict_publisher = self.create_publisher(String, 'item_dict', 10, callback_group=self.group_2)
        self.item_dict_msg = String()
        self.ite = 0
        
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
            QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE),
            callback_group=self.group_1
        )
        
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE),
            callback_group=self.group_1
        )
        
        
        ############# EXTRA #############
        
        
        self.depth_range = depth_range
        self.debug_mode = debug_mode
        
        # Initializing object detection model
        self.model = YOLO(object_detection_model)
        
        # Initialize camera
        #self.__initialize_camera_stream()
        self.depth_scale = 0.001
        self.clipping_distance = depth_range/self.depth_scale
        
        
    #################
    ###### ROS ######
    #################
    def aligned_depth_image_callback(self, msg):
        #self.get_logger().info('aligned_depth_image_callback')
        try:
            # Convert ROS Image msg to cv2 mat then to np array
            self.cv_depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.np_depth_image = np.array(self.cv_depth_image, dtype=np.uint16)
            #self.get_logger().debug(f'Depth encoding: {msg.encoding}') #-> 16UC1 = uint16

            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')


    def color_image_callback(self, msg):
        #self.get_logger().info('color_image_callback')
        try:
            # Convert ROS Image msg to cv2 mat then to np array
            self.cv_color_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.np_color_image = np.array(self.cv_color_image, dtype=np.uint8)
            #self.get_logger().debug(f'Color encoding: {msg.encoding}') #-> rgb8 ( = bgr8?)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
            
            
    def camera_info_callback(self, msg):
        try:
            if self.camera_intrinsics is None:
                # Set intrinsics in o3d object
                self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
                self.camera_intrinsics.set_intrinsics(msg.width,    #msg.width
                                                  msg.height,   #msg.height
                                                  msg.k[0],     #msg.K[0] -> fx
                                                  msg.k[4],     #msg.K[4] -> fy
                                                  msg.k[2],     #msg.K[2] -> cx
                                                  msg.k[5] )    #msg.K[5] -> cy
                self.get_logger().info('Camera intrinsics have been set!')
            
        except Exception as e:
            self.get_logger().error(f'camera_info_callback Error: {e}')


    def camera_callback(self):
        # self.vision_object = RealsenseVision(object_detection_model="yolov8s-seg.pt", 
        #                             realsense_product_line="D400",
        #                             depth_range=3,
        #                             debug_mode=False)
        # self.vision_object.streaming_loop()
        #self.item_dict_msg.data = (f'Test msg {self.ite}')
        #self.item_dict_publisher.publish(self.item_dict_msg)
        #self.ite += 1
        self.streaming_loop()
    
    
    
    #--------------------------#
    # CLASS CALLABLE FUNCTIONS #
    #--------------------------#

    def streaming_loop(self):
        '''
        Run a continuous loop to capture frames from a RealSense camera, process them, and display the results.

        The method performs the following steps:
        1. Starts camera streaming based on the provided pipeline configuration.
        2. Sets camera intrinsics and obtains a clipping distance for background removal.
        3. Enters a continuous loop to capture frames, align depth to color, and perform various image processing tasks.
        4. Removes the background from the frames and uses an object recognition model to predict objects.
        5. For each detected object, extracts its depth information, filters the depth data, and generates a point cloud.
        6. Displays the masks and processed frames, including the depth-to-color alignment and detected object point clouds.
        7. Allows the user to exit the loop by pressing 'q' or 'esc'.

        Note:
        - The method uses OpenCV and NumPy for image processing and visualization.
        - It relies on an object recognition model to detect and predict objects in the scene.
        - The `debug_mode` attribute controls whether additional debug information is displayed.

        Parameters:
        None
        
        Returns:
        None
        '''
        
        # Check camera intrinsics
        if self.camera_intrinsics is None or self.np_depth_image is None or self.np_color_image is None:
            self.get_logger().info("Program not ready...")
            return
        
        #try:
            
        #frames = self.pipeline.wait_for_frames() # Get frames from stream
        
        #aligned_frames = self.align.process(frames) # Align depth to color image
        #aligned_depth_frame = aligned_frames.get_depth_frame() # 640x480 depth image
        #color_frame = aligned_frames.get_color_frame()
        
        # Validate that both frames are valid
        #if not aligned_depth_frame or not color_frame:
        #    continue
        
        # Convert frames to numpy arrays
        depth_image = self.np_depth_image
        color_image = self.np_color_image
        
        # Remove background - Sets pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image)) # depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d != depth_image_3d), grey_color, color_image)
        self.get_logger().info("bg_removed")
        cv2.imwrite(f'/home/dreezy/dump/bg_removed_{self.ite}.png', bg_removed)
        self.ite += 1
        self.get_logger().info("bg_removed saved")
        
        
        # Predict the objects in the background removed image using the object recognition model
        results = self.model(source=bg_removed, show=False, stream=False, conf=0.6, verbose=self.debug_mode)
        self.get_logger().info("results pred done...")
        
        # Go through each model detection
        for detection in results:
            object_boxes = detection.boxes.xyxy.cpu().numpy()
            n_objects = object_boxes.shape[0]
            
            try:
                masks = detection.masks.data
            except AttributeError:
                continue
            
            npmasks = masks.cpu().numpy().astype(np.uint8)
            
            # Declare variables used later
            objects_global_point_clouds = []
            objects_median_center = []
            
            # To display all masks at once we SMACK em together bang bang
            #mask = np.max(npmasks, axis=0)
            
            # Display masks
            #if self.debug_mode: cv2.imshow("masks", mask*254)
            
            
            for i in range(n_objects):
                # Get mask for the i'th object 
                single_selection_mask = npmasks[i]
                
                # Get binary selection matrix, idx 
                # (Matrix with same size as mask image, but has either true or false instead of zeros and ones)
                idx = (single_selection_mask == 1)
                
                # Constructing the output array
                single_object_depth = np.zeros_like(depth_image)
                
                # The idx mask is used to create a depth image with only the object
                single_object_depth[idx] = depth_image[idx]
                
                # Filter pixels which are too far away from the object pixel median?
                single_object_depth = filter_depth_object_img(single_object_depth, idx, 0.15)
                
                self.get_logger().info(f'{single_object_depth}')
                ### Get the pointcloud for the i'th object
                depth_raw = o3d.geometry.Image(single_object_depth)
                object_pointcloud = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, self.camera_intrinsics)
                
                
                ## Reduce precision of pointcloud to improve performance
                voxel_grid = object_pointcloud.voxel_down_sample(0.01)
                voxel_grid, _ = voxel_grid.remove_radius_outlier(nb_points=30, radius=0.05)
                
                # Save i'th object pointcloud to list
                objects_global_point_clouds.append(voxel_grid)
                
                
                ## Extract pointcloud points to numpy array
                np_pointcloud = np.asarray(object_pointcloud.points)
                
                # Get median xyz value
                median_center = np.median(np_pointcloud, axis=0)
                
                # Save i'th object pointcloud median center to list
                objects_median_center.append(median_center)
                if self.debug_mode: print(f"median center{i} {median_center}")
                
                
                # Item dict creation
                item_dict = {}
                detection_class = detection.boxes.cls.cpu().numpy()
                detection_conf = detection.boxes.conf.cpu().numpy()
                
                for item, n, median, conf in zip(detection_class, range(n_objects), objects_median_center, detection_conf):
                    item_dict[f'item_{n}'] = {'class': detection.names[item],
                                             'confidence': conf.tolist(),
                                             'median_center_': median.tolist()}
                self.item_dict = item_dict
                self.item_dict_str = json.dumps(self.item_dict)
                        
                        
                
                
                if self.debug_mode: 
                    # Display object pointclouds using matplotlib.plt
                    try:
                        display_object_pointclouds(objects_global_point_clouds)
                    except UnboundLocalError:
                        print("display_object_pointclouds error")
                    
                    # Display depth to color alignment
                    try:
                        cv2.putText(depth_image, "{} m".format(median_center), (10,30), 0, 1, (0,0,255),3)
                        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                        images = np.hstack((bg_removed, depth_colormap))
                        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL) # Allows the window to be resized
                        cv2.imshow('Align Example', images)
                    except NameError:
                        print("display depth to color alignment error")

                
                # # Press 'q' or 'esc' to close the image window
                # key = cv2.waitKey(1)
                # if key & 0xFF == ord('q') or key == 27:
                #     cv2.destroyAllWindows()
                    
        # except:
        #     self.get_logger().info("Went through except...")
                      
        # finally:
        #     print("Ending pipeline stream")
        #     self.pipeline.stop()
            



#####################################################
#-------------------- MAIN CODE --------------------#
#####################################################

def main(args=None):
    rclpy.init(args=args)

    # Instansiate node class
    vision_node = RealsenseVision(object_detection_model="yolov8s-seg.pt", 
                                  realsense_product_line="D400",
                                  depth_range=3,
                                  debug_mode=False)

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




if __name__ == "__main__":
    
    # vision_object = RealsenseVision(object_detection_model="yolov8s-seg.pt", 
    #                                 realsense_product_line="D400",
    #                                 depth_range=3,
    #                                 debug_mode=False)
    # vision_object.streaming_loop()
    
    main()