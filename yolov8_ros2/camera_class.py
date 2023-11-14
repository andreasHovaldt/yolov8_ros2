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

class RealsenseVision(): # Currently works for the D400 product line, other Intel Realsense product lines might not work
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
        
        self.depth_range = depth_range
        self.debug_mode = debug_mode
        
        # Initializing object detection model
        self.model = YOLO(object_detection_model)
        
        # Initialize camera
        self.__initialize_camera_stream()
        
        
        
    def __initialize_camera_stream(self):
        # Initializing camera
        self.pipeline = rs.pipeline()
        self.pipeline_config = rs.config()
        self.decimation_filter = rs.decimation_filter()
        
        # Define the included streams for the pipeline 
        self.pipeline_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # Include depth stream
        self.pipeline_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # Include color stream
        
        
    def __set_camera_intrinsics(self): # TODO: ROS INTEGRATION: sensor_msgs/msg/CameraInfo -> msg.K -> This would yield the intrinsic camera matrix
        # Get intrinsics from pipeline stream
        intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
        # Set intrinsics in o3d object
        self.rs_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        self.rs_intrinsics.set_intrinsics(intrinsics.width,      #msg.width
                                     intrinsics.height,     #msg.height
                                     intrinsics.fx,         #msg.K[0] -> fx
                                     intrinsics.fy,         #msg.K[4] -> fy
                                     intrinsics.width/2,    #msg.K[2] -> cx
                                     intrinsics.height/2 )  #msg.K[5] -> cy
        
        #TODO: ROS INTEGRATION: https://dev.intelrealsense.com/docs/ros2-align-depth
        self.align = rs.align(rs.stream.color) # align object used for aligning depth images to color images
        

    def __get_clipping_distance(self, depth_range_in_meters):
        depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        return depth_range_in_meters / depth_scale

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
        # Start camera streaming
        self.profile = self.pipeline.start(self.pipeline_config)
        
        # Set camera intrinsics
        self.__set_camera_intrinsics()
        
        # Get clipping distance, used for background removal
        self.clipping_distance = self.__get_clipping_distance(self.depth_range)
        
        try:
            while True:
                frames = self.pipeline.wait_for_frames() # Get frames from stream
                
                aligned_frames = self.align.process(frames) # Align depth to color image
                aligned_depth_frame = aligned_frames.get_depth_frame() # 640x480 depth image
                color_frame = aligned_frames.get_color_frame()
                
                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue
                
                # Convert frames to numpy arrays
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Remove background - Sets pixels further than clipping_distance to grey
                grey_color = 153
                depth_image_3d = np.dstack((depth_image, depth_image, depth_image)) # depth image is 1 channel, color is 3 channels
                bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d != depth_image_3d), grey_color, color_image)
                
                # Predict the objects in the background removed image using the object recognition model
                results = self.model(source=bg_removed, show=True, stream=False, conf=0.6)
                
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
                    mask = np.max(npmasks, axis=0)
                    
                    # Display masks
                    if self.debug_mode: cv2.imshow("masks", mask*254)
                    
                    
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
                        
                        
                        if self.debug_mode: cv2.imshow("sod before filt", cv2.normalize(np.array(single_object_depth,np.uint8),None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F))

                        # Filter pixels which are too far away from the object pixel median?
                        single_object_depth = filter_depth_object_img(single_object_depth, idx, 0.15)
                        
                        if self.debug_mode: cv2.imshow("sod before filt", cv2.normalize(np.array(single_object_depth,np.uint8),None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F))
                    

                        ### Get the pointcloud for the i'th object
                        depth_raw = o3d.geometry.Image(single_object_depth)
                        object_pointcloud = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, self.rs_intrinsics)
                        
                        
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
                        
                        
                        item_dict = {}
                        detection_class = detection.boxes.cls.cpu().numpy()
                        detection_conf = detection.boxes.conf.cpu().numpy()
                        # for item, median in zip(detection_class, objects_median_center):
                        #     item_dict[detection.names[item]] = median
                        # print(item_dict.keys())
                        
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
                        continue
                    
                    # Display depth to color alignment
                    try:
                        cv2.putText(depth_image, "{} m".format(median_center), (10,30), 0, 1, (0,0,255),3)
                        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                        images = np.hstack((bg_removed, depth_colormap))
                        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL) # Allows the window to be resized
                        cv2.imshow('Align Example', images)
                    except NameError:
                        continue

                
                # Press 'q' or 'esc' to close the image window
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break    
                    
                      
        finally:
            print("Ending pipeline stream")
            self.pipeline.stop()
            



#####################################################
#-------------------- MAIN CODE --------------------#
#####################################################


if __name__ == "__main__":
    
    vision_object = RealsenseVision(object_detection_model="yolov8s-seg.pt", 
                                    realsense_product_line="D400",
                                    depth_range=3,
                                    debug_mode=False)
    vision_object.streaming_loop()