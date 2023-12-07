# Basic ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import ReliabilityPolicy, QoSProfile


# Executor and callback imports
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# ROS2 interfaces
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String

# Image msg parser
from cv_bridge import CvBridge

# Vision model
from ultralytics import YOLO

# Others
import numpy as np
import open3d as o3d
import time, json, torch


class Yolov8Node(Node):
    
    def __init__(self):
        super().__init__("yolov8_node")
        rclpy.logging.set_logger_level('yolov8_node', rclpy.logging.LoggingSeverity.INFO)
        
        ## Declare parameters for node
        self.declare_parameter("model", "yolov8s-seg.pt")
        model = self.get_parameter("model").get_parameter_value().string_value
        
        self.declare_parameter("device", "cuda:0")
        self.device = self.get_parameter("device").get_parameter_value().string_value
        
        self.declare_parameter("depth_threshold", 1.2)
        self.depth_threshold = self.get_parameter("depth_threshold").get_parameter_value().double_value
        
        self.declare_parameter("threshold", 0.6)
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        
        self.declare_parameter("enable_yolo", True)
        self.enable_yolo = self.get_parameter("enable_yolo").get_parameter_value().bool_value
        
        
        self.tf_world_to_camera = np.array([[-0.000, -1.000,  0.000, -0.017], [0.559,  0.000,  0.829, -0.272], [-0.829,  0.000,  0.559,  0.725], [0.000,  0.000,  0.000,  1.000]])
        self.tf_camera_to_optical = np.array([[-0.003,  0.001,  1.000,  0.000], [-1.000, -0.002, -0.003,  0.015], [0.002, -1.000,  0.001, -0.000], [0.000,  0.000,  0.000,  1.000]])
        self.tf_world_to_optical = np.matmul(self.tf_world_to_camera, self.tf_camera_to_optical)

        
        ## other inits
        self.group_1 = MutuallyExclusiveCallbackGroup() # camera subscribers
        self.group_2 = MutuallyExclusiveCallbackGroup() # vision timer
        
        self.cv_bridge = CvBridge()
        self.yolo = YOLO(model)
        self.yolo.fuse() # Conv2d and BatchNorm2d Layer Fusion:
                         # Conv2d layers are often followed by BatchNorm2d layers in deep neural networks.
                         # Fusing these layers means combining the operations of the convolutional layer and the batch normalization layer into a single operation.
                         # This can reduce the computational cost and improve inference speed.
        self.color_image_msg = None
        self.depth_image_msg = None
        self.camera_intrinsics = None
        self.pred_image_msg = Image()
        
        # Set clipping distance for background removal
        depth_scale = 0.001
        self.depth_threshold = self.depth_threshold/depth_scale
        
        
        # Publishers
        self._item_dict_pub = self.create_publisher(String, "/yolo/prediction/item_dict", 10)
        self._pred_pub = self.create_publisher(Image, "/yolo/prediction/image", 10)
        
        # Subscribers
        self._color_image_sub = self.create_subscription(Image, "/camera/color/image_raw", self.color_image_callback, qos_profile_sensor_data, callback_group=self.group_1)
        self._depth_image_sub = self.create_subscription(Image, "/camera/aligned_depth_to_color/image_raw", self.depth_image_callback, qos_profile_sensor_data, callback_group=self.group_1)
        self._camera_info_subscriber = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_callback, QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE), callback_group=self.group_1)

        # Timers
        self._vision_timer = self.create_timer(0.04, self.object_segmentation, callback_group=self.group_2) # 25 hz

    
    def color_image_callback(self, msg):
        self.color_image_msg = msg
        
    def depth_image_callback(self, msg):
        self.depth_image_msg = msg
    
    def camera_info_callback(self, msg):
        try:
            if self.camera_intrinsics is None:
                # Set intrinsics in o3d object
                self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
                self.camera_intrinsics.set_intrinsics(msg.width,    #msg.width
                                                  msg.height,       #msg.height
                                                  msg.k[0],         #msg.K[0] -> fx
                                                  msg.k[4],         #msg.K[4] -> fy
                                                  msg.k[2],         #msg.K[2] -> cx
                                                  msg.k[5] )        #msg.K[5] -> cy
                self.get_logger().info('Camera intrinsics have been set!')
            
        except Exception as e:
            self.get_logger().error(f'camera_info_callback Error: {e}')


    def bg_removal(self, color_img_msg: Image, depth_img_msg: Image):
        if self.color_image_msg is not None and self.depth_image_msg is not None:
        
            # Convert color image msg
            cv_color_image = self.cv_bridge.imgmsg_to_cv2(color_img_msg, desired_encoding='bgr8')
            np_color_image = np.array(cv_color_image, dtype=np.uint8)

            # Convert depth image msg
            cv_depth_image = self.cv_bridge.imgmsg_to_cv2(depth_img_msg, desired_encoding='passthrough')
            np_depth_image = np.array(cv_depth_image, dtype=np.uint16)

            # bg removal
            grey_color = 153
            depth_image_3d = np.dstack((np_depth_image, np_depth_image, np_depth_image)) # depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > self.depth_threshold) | (depth_image_3d != depth_image_3d), grey_color, np_color_image)
            
            return bg_removed, np_color_image, np_depth_image
        self.get_logger().error("Background removal error, color or depth msg was None")
    
    
    def filter_depth_object_img(self, img, starting_mask, deviation): #TODO: I need explanation from alfredo -Dreez
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
        out_img = img
        zero_img = np.zeros_like(img)
        out_img[uidx] = zero_img[uidx]
        return out_img
    
    
    def object_segmentation(self):
        if self.enable_yolo and self.color_image_msg is not None and self.depth_image_msg is not None:
            self.get_logger().debug("Succesfully acquired color and depth image msgs")
            
            # Remove background
            bg_removed, np_color_image, np_depth_image = self.bg_removal(self.color_image_msg, self.depth_image_msg)
            self.get_logger().debug("Succesfully removed background")
            
            # Predict on image "bg_removed"
            results = self.yolo.predict(
                source=bg_removed,
                show=False,
                verbose=False,
                stream=False,
                conf=self.threshold,
                device=self.device
            )
            self.get_logger().debug("Succesfully predicted")
            
            
            # Go through detections in prediction results
            for detection in results:
                
                # Extract image with yolo predictions
                pred_img = detection.plot()
                self.pred_image_msg = self.cv_bridge.cv2_to_imgmsg(pred_img, encoding='passthrough')
                self._pred_pub.publish(self.pred_image_msg)
                
                # Get number of objects in the scene
                object_boxes = detection.boxes.xyxy.cpu().numpy()
                n_objects = object_boxes.shape[0]

                try:
                    masks = detection.masks.data
                except AttributeError:
                    continue
                
                npmasks = masks.cpu().numpy().astype(np.uint8)
                self.get_logger().debug("Succesfully extracted boxes and masks")
                
                # Declare variables used later
                objects_global_point_clouds = []
                objects_median_center = []
                objects_median_center_transform = []



                for i in range(n_objects):
                    # Get mask for the i'th object 
                    single_selection_mask = npmasks[i]

                    # Get binary selection matrix, idx 
                    # (Matrix with same size as mask image, but has either true or false instead of zeros and ones)
                    idx = (single_selection_mask == 1)

                    # Constructing the output array
                    single_object_depth = np.zeros_like(np_depth_image)

                    # The idx mask is used to create a depth image with only the object
                    single_object_depth[idx] = np_depth_image[idx]

                    # Filter pixels which are too far away from the object pixel median?
                    single_object_depth = self.filter_depth_object_img(single_object_depth, idx, 0.15)

                    ### Get the pointcloud for the i'th object
                    depth_raw = o3d.geometry.Image(single_object_depth.astype(np.uint16))
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
                    median_center = np.append(median_center, 1)
                    median_center_transformed = np.matmul(self.tf_world_to_optical, median_center)

                    # Save i'th object pointcloud median center to list
                    objects_median_center.append(median_center)
                    objects_median_center_transform.append(median_center_transformed)


                # Item dict creation
                item_dict = {}
                detection_class = detection.boxes.cls.cpu().numpy()
                detection_conf = detection.boxes.conf.cpu().numpy()
                
                for item, n, median_tf in zip(detection_class, range(n_objects), objects_median_center_transform):
                    item_dict[f'item_{n}'] = {'class': detection.names[item],
                                             #'confidence': conf.tolist(),
                                             #'median_center': median.tolist(),
                                             'position': median_tf.tolist()}
                
                self.item_dict = item_dict
                self.item_dict_str = json.dumps(self.item_dict)
                #self.get_logger().info(f"Yolo detected items: {detection.names[detection_class]}")
                self.get_logger().info(f"Yolo detected items: {[detection.names[item] for item in detection_class]}")
                
                item_dict_msg = String()
                item_dict_msg.data = self.item_dict_str
                self._item_dict_pub.publish(item_dict_msg)
                
                self.get_logger().debug("Item dictionary succesfully created and published")
            
            
            
    def shutdown_callback(self):
        self.get_logger().warn("Shutting down...")
        
        

def main(args=None):
    rclpy.init(args=args)

    # Instansiate node class
    vision_node = Yolov8Node()

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
    main()