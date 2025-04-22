import pyrealsense2 as rs
import numpy as np
import cv2
import os


class CalibrateCameras:
    def __init__(self):
        self.context = rs.context()
        self.devices = self.context.query_devices()
        self.device_count = len(self.devices)

        if self.device_count == 0:
            print("No RealSense cameras detected.")
            return
        
        print(f"Number of RealSense cameras detected: {self.device_count}")

    def save_frames_for_calibration(self, save_path="cameras/calibration_images", num_frames=10):
        os.makedirs(save_path, exist_ok=True)
        
        for frame_num in range(num_frames):
            for i, device in enumerate(self.devices):
                serial = device.get_info(rs.camera_info.serial_number)
                stream_path = os.path.join(save_path, str(frame_num))      
                os.makedirs(stream_path, exist_ok=True)        
                print(f"Saving frame {frame_num} from Camera", i + 1)
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                profile = pipeline.start(config)

                # get the intrinsic matrix
                intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                intrinsic_matrix = [
                    [intrinsics.fx, 0, intrinsics.ppx],
                    [0, intrinsics.fy, intrinsics.ppy],
                    [0, 0, 1]
                ]

                # get distortion coefficients
                distortion_coeff = intrinsics.coeffs
                np.save(os.path.join(stream_path, f"{serial}_intrinsic_matrix.npy"), intrinsic_matrix)
                np.save(os.path.join(stream_path, f"{serial}_distortion_coeff.npy"), distortion_coeff)

                frameset = pipeline.wait_for_frames()
                # align color frame to depth frame
                align = rs.align(rs.stream.color)
                aligned_frames = align.process(frameset)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                cv2.imwrite(os.path.join(stream_path, f"{serial}_color.png"), color_image)
                np.save(os.path.join(stream_path, f"{serial}_depth.npy"), depth_image)

                pipeline.stop()
                config.disable_all_streams()
            _ = input("press enter to continue")

    def detect_charuco_corners(self, image, board, aruco_dict, aruco_params, image_name=None):
        """Detect Charuco corners in an image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        print(type(gray), type(aruco_dict), type(aruco_params))
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        # exit()
        if ids is not None:
            if image_name is not None:
                output_image = image.copy()
                cv2.aruco.drawDetectedMarkers(output_image, corners, ids)
                # cv2.imwrite(image_name, output_image)
                # print(f"Detected markers saved as {image_name}")

            # Detect Charuco corners
            charuco_num_detection, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)

            if charuco_num_detection > 3:  # Need at least 4 points
                return charuco_corners, charuco_ids
            # else:
            #     print(f"Not enough Charuco corners detected. Need at least 4. Detected: {charuco_num_detection}")
        else:
            print("No markers detected. Image:", image_name)

        return None, None

    def estimate_camera_pose(self, charuco_corners, charuco_ids, board, intrinsics_matrix, distortion_coeffs):
        """Estimate camera pose using a Charuco Board."""
        rvec = np.zeros((1, 3), dtype=np.float64)
        tvec = np.zeros((1, 3), dtype=np.float64)
        success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, intrinsics_matrix, distortion_coeffs, rvec, tvec
        )
        if success:
            return rvec, tvec
        else:
            print("Pose estimation failed.")
            return None, None

    def compute_extrinsics(self, rvec1, tvec1, rvec2, tvec2):
        """Compute the transformation matrix from Camera 2 to Camera 1."""
        R1, _ = cv2.Rodrigues(rvec1)
        R2, _ = cv2.Rodrigues(rvec2)

        R = R2 @ R1.T  # Relative rotation
        T = tvec2 - R @ tvec1  # Relative translation

        # Create 4x4 transformation matrix
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = T.flatten()
        return extrinsic_matrix

    def calibrate_cameras(self, camera_serials=None, saved_path="cameras/calibration_images_1"):
        """
        Calibrate multiple cameras and compute transformation matrices between them.
        
        Args:
            camera_serials: List of camera serial numbers
            saved_path: Path where calibration images are stored
        
        Returns:
            Dictionary containing transformation matrices between cameras
        """


        if camera_serials is None:
            return None

        # Create ChArUco board for calibration
        aruco_dict_id = cv2.aruco.DICT_6X6_250
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        squares_verticle = 7
        squares_horizontal = 5
        # square_length = 0.078  # meters
        # marker_length = 0.069  # meters

        square_length = 0.035  # meters
        marker_length = 0.026  # meters
        
        board = cv2.aruco.CharucoBoard(
            (squares_horizontal, squares_verticle),
            square_length,
            marker_length,
            aruco_dict
        )
        
        # Set up ArUco detection parameters
        aruco_params = cv2.aruco.DetectorParameters()
        
        # Get all frames for calibration
        frames_data = {}
        for frame_num in range(1):  # Assuming 10 frames as in save_frames_for_calibration
            if frame_num == 15 or frame_num == 21:
                continue

            frame_path = os.path.join(saved_path, str(frame_num))
            if not os.path.exists(frame_path):
                continue
                
            for serial in camera_serials:
                color_path = os.path.join(frame_path, f"{serial}_color.png")
                intrinsic_path = os.path.join(frame_path, f"{serial}_intrinsic_matrix.npy")
                distortion_path = os.path.join(frame_path, f"{serial}_distortion_coeff.npy")
                
                if os.path.exists(color_path) and os.path.exists(intrinsic_path):
                    if serial not in frames_data:
                        frames_data[serial] = {
                            'color_images': [],
                            'intrinsics': None,
                            'distortion': None
                        }
                    
                    color_image = cv2.imread(color_path)
                    frames_data[serial]['color_images'].append(color_image)
                    
                    # Load intrinsics if not already loaded
                    if frames_data[serial]['intrinsics'] is None and os.path.exists(intrinsic_path):
                        intrinsic_matrix = np.load(intrinsic_path)
                        frames_data[serial]['intrinsics'] = intrinsic_matrix
                    
                    # Load distortion coefficients if not already loaded
                    if frames_data[serial]['distortion'] is None and os.path.exists(distortion_path):
                        distortion_coeff = np.load(distortion_path)
                        frames_data[serial]['distortion'] = distortion_coeff
        
        # Calculate transformation matrices between cameras
        transformation_matrices = {}
        camera1_serial = camera_serials[0]
        camera2_serial = camera_serials[1]
        
        # Check if we have data for both cameras
        if camera1_serial not in frames_data or camera2_serial not in frames_data:
            print(f"Missing data for one or both cameras: {camera1_serial}, {camera2_serial}")
            return None
        
        # For each frame pair, calculate the transformation matrix
        best_transformation_matrix = None
        best_num_corners = 0
        
        for i in range(min(len(frames_data[camera1_serial]['color_images']), 
                            len(frames_data[camera2_serial]['color_images']))):
            
            image1 = frames_data[camera1_serial]['color_images'][i]
            image2 = frames_data[camera2_serial]['color_images'][i]
            
            intrinsic_matrix1 = frames_data[camera1_serial]['intrinsics']
            intrinsic_matrix2 = frames_data[camera2_serial]['intrinsics']
            
            # Use loaded distortion coefficients or zeros if not available
            distortion_coeffs1 = frames_data[camera1_serial]['distortion'] if frames_data[camera1_serial]['distortion'] is not None else np.zeros((5, 1))
            distortion_coeffs2 = frames_data[camera2_serial]['distortion'] if frames_data[camera2_serial]['distortion'] is not None else np.zeros((5, 1))
            
            # Detect ChArUco corners in both images
            charuco_corners1, charuco_ids1 = self.detect_charuco_corners(
                image1, board, aruco_dict, aruco_params, f"calib_{camera1_serial}_{i}.jpg")
            
            charuco_corners2, charuco_ids2 = self.detect_charuco_corners(
                image2, board, aruco_dict, aruco_params, f"calib_{camera2_serial}_{i}.jpg")
            
            if charuco_corners1 is not None and charuco_corners2 is not None:
                # Count number of corners detected
                num_corners = min(charuco_corners1.shape[0], charuco_corners2.shape[0])
                
                # Estimate camera poses
                rvec1, tvec1 = self.estimate_camera_pose(
                    charuco_corners1, charuco_ids1, board, intrinsic_matrix1, distortion_coeffs1)
                
                rvec2, tvec2 = self.estimate_camera_pose(
                    charuco_corners2, charuco_ids2, board, intrinsic_matrix2, distortion_coeffs2)
                
                if rvec1 is not None and rvec2 is not None:
                    # Compute extrinsic matrix (camera 2 to camera 1)
                    extrinsic_matrix = self.compute_extrinsics(rvec1, tvec1, rvec2, tvec2)
                    
                    # Keep the transformation with the most corners detected
                    if num_corners > best_num_corners:
                        best_transformation_matrix = extrinsic_matrix
                        best_num_corners = num_corners
        
        if best_transformation_matrix is not None:
            # Save the transformation matrix
            output_dir = "cameras/camera_calibration_utils"
            os.makedirs(output_dir, exist_ok=True)
            
            # This is transformation from camera 2 to camera 1
            transformation_matrix_2_to_1 = best_transformation_matrix
            np.save(os.path.join(output_dir, f"transformation_matrix_{camera2_serial}_to_{camera1_serial}.npy"), 
                    transformation_matrix_2_to_1)
            
            # Inverse
            # transformation_matrix_2_to_1 = np.linalg.inv(transformation_matrix_2_to_1)
            
            print(f"Transformation Matrix from Camera {camera2_serial} to Camera {camera1_serial}:")
            print(transformation_matrix_2_to_1)
            
            return transformation_matrix_2_to_1
        else:
            print("Failed to compute transformation matrix. Check the calibration images.")
            return None    

    def get_camera_frame(self):
        frames_data = {}
        
        for i, device in enumerate(self.devices):
            serial = device.get_info(rs.camera_info.serial_number)
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = pipeline.start(config)

            frameset = pipeline.wait_for_frames()
            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frameset)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            frames_data[serial] = {
                'color_image': color_image,
                'depth_image': depth_image
            }

            pipeline.stop()
            config.disable_all_streams()


        return frames_data
     
    
    
if __name__ == "__main__":
    calibrator = CalibrateCameras()
    # if input("Collect frames for calibration? (y/n): ") == "y":
    #     calibrator.save_frames_for_calibration(save_path="cameras/test_images", num_frames=1)

    
    calibrator.calibrate_cameras(['130322273305', '126122270722'])
    # calibrator.calibrate_cameras(['f1371463', 'f1380660'])

    