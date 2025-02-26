import cv2
import time
import numpy as np

from typing import Tuple

def generate_and_save_charuco_board(
        filename : str = None, 
        aruco_dict_id : int = cv2.aruco.DICT_6X6_250
    ) -> Tuple[cv2.aruco.CharucoBoard , cv2.aruco.Dictionary]:
    """Generate a Charuco board image and save it to a file."""
    squares_verticle = 7
    squares_horizontal = 5
    square_length = 0.04 # meters
    marker_length = 0.03 # meters
    margin_size = 20 # pixels

    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    board = cv2.aruco.CharucoBoard(
        (squares_horizontal, squares_verticle), 
        square_length, 
        marker_length, 
        aruco_dict,
        )
    size_ratio = squares_verticle/squares_horizontal
    board_image = cv2.aruco.CharucoBoard.generateImage(
        board,
        (640, int(640*size_ratio)), 
        marginSize=margin_size,
        )
    ## Save the image
    if filename is not None:
        cv2.imwrite(filename, board_image)
        print(f"Charuco board image saved as {filename}")

    return board, aruco_dict


def detect_charuco_corners(image : np.ndarray, 
                           board: cv2.aruco.CharucoBoard,
                           aruco_dict: cv2.aruco.Dictionary,
                           aruco_params: cv2.aruco.DetectorParameters,
                           image_name: str = None,
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Detect Charuco corners in an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        if image_name is not None:
            output_image = image.copy()
            cv2.aruco.drawDetectedMarkers(output_image, corners, ids)
            image_name = str(time.time()) + image_name
            cv2.imwrite(image_name, output_image)
            print(f"Detected markers saved as {image_name}")

        # Step 2: Detect Charuco corners
        charuco_num_detection, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)

        if charuco_num_detection > 3:  # Need at least 4 points
            return charuco_corners, charuco_ids
        else:
            print("Not enough Charuco corners detected. Need at least 4. Detected:", charuco_num_detection)
    else:
        print("No markers detected.")

    return None, None


def estimate_camera_pose(charuco_corners : np.ndarray,
                         charuco_ids : np.ndarray,
                         board: cv2.aruco.CharucoBoard,
                         intrinsics_matrix : np.ndarray,
                         distortion_coeffs : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def compute_extrinsics(rvec1, tvec1, rvec2, tvec2):
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


def get_transformation_matrix_between_camera1_and_camera2(image1 : np.ndarray, 
                                                          image2 : np.ndarray,
                                                          intrinsic_matrix1 : np.ndarray,
                                                          intrinsic_matrix2 : np.ndarray,
                                                          distortion_coeffs1 : np.ndarray,
                                                          distortion_coeffs2 : np.ndarray,
                                                          board : cv2.aruco.CharucoBoard,
                                                          aruco_dict : cv2.aruco.Dictionary,
                                                          aruco_params : cv2.aruco.DetectorParameters,) -> np.ndarray:
    ## Detect the Charuco corners in the images
    charuco_corners1, charuco_ids1 = detect_charuco_corners(image1, board, aruco_dict, aruco_params)
    charuco_corners2, charuco_ids2 = detect_charuco_corners(image2, board, aruco_dict, aruco_params)

    if charuco_corners1 is None or charuco_corners2 is None:
        print("Charuco corners detection failed.")
        return None
    else:
        rvec1, tvec1 = estimate_camera_pose(charuco_corners1, charuco_ids1, board, intrinsic_matrix1, distortion_coeffs1)
        rvec2, tvec2 = estimate_camera_pose(charuco_corners2, charuco_ids2, board, intrinsic_matrix2, distortion_coeffs2)
        extrinsic_matrix = compute_extrinsics(rvec1, tvec1, rvec2, tvec2)
        return np.linalg.inv(extrinsic_matrix)


def transform_point_cloud(point_cloud, extrinsic_matrix):
    """Applies the extrinsic transformation to a point cloud."""
    points = np.array(point_cloud)
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (extrinsic_matrix @ points_homogeneous.T).T

    return transformed_points[:, :3]


def get_transformation_from_base_to_wrist_camera(eef_pose: list) -> np.ndarray:
    ## Transformation between the base and the wrist camera
    T_base2eef = np.eye(4)
    T_base2eef[0:3, 0:3] = cv2.Rodrigues(np.array(eef_pose[3:6]))[0]
    T_base2eef[0:3, 3] = eef_pose[0:3]

    ## Transformation between the wrist camera and the end effector
    T_eef2cam = np.eye(4)
    ## Change the z here to make the PCD Correct.
    T_eef2cam[0:3, 3] = np.array([-.01, -.08, 0.01]) ## 1cm, 8cm, 1cm offset
    ## Make rotaion matrix with 15 degree rotation aroud x axis
    T_eef2cam[0:3, 0:3] = cv2.Rodrigues(np.array([-np.pi/12, 0, 0]))[0]

    ## Transformation between the base and the wrist camera
    T_w2cam = T_base2eef @ T_eef2cam

    return T_w2cam, T_base2eef


def get_transformation_matrix(data_path : str = "resources/camera_calibration/",
                                     camera1_serial : str = "130322273305",
                                     camera2_serial : str = "127122270512",
                                     camera3_serial : str = "126122270722",
                                     lightning_eef_pose : list = [-0.5019622959075773, 0.10288288422589228, 0.845704241201291, -1.674059653851395, 1.740895410756312, 1.1504085564683173]) -> Tuple[np.ndarray, np.ndarray]:
    # Generate and save Charuco board image, Pass None to not save.
    board, aruco_dict = generate_and_save_charuco_board(None)

    ## Load Images
    rgb1 = np.load(f'{data_path}inference_color_image_{camera1_serial}.npy')
    rgb2 = np.load(f'{data_path}inference_color_image_{camera2_serial}.npy')

    rgb3 = np.load(f'{data_path}inference_color_image_{camera3_serial}.npy')

    depth1 = np.load(f'{data_path}inference_depth_image_{camera1_serial}.npy')
    depth2 = np.load(f'{data_path}inference_depth_image_{camera2_serial}.npy')
    depth3 = np.load(f'{data_path}inference_depth_image_{camera3_serial}.npy')

    camera1_intrinsics = np.load(f'{data_path}camera_intrinsic_{camera1_serial}.npy')
    camera2_intrinsics = np.load(f'{data_path}camera_intrinsic_{camera2_serial}.npy')
    camera3_intrinsics = np.load(f'{data_path}camera_intrinsic_{camera3_serial}.npy')

    cx1, cy1, fx1, fy1 = camera1_intrinsics[0], camera1_intrinsics[1], camera1_intrinsics[2], camera1_intrinsics[3]
    cx2, cy2, fx2, fy2 = camera2_intrinsics[0], camera2_intrinsics[1], camera2_intrinsics[2], camera2_intrinsics[3]
    cx3, cy3, fx3, fy3 = camera3_intrinsics[0], camera3_intrinsics[1], camera3_intrinsics[2], camera3_intrinsics[3]

    camera1_intrinsics = np.array([[fx1, 0, cx1],
                                    [0, fy1, cy1],
                                    [0, 0, 1]])

    camera2_intrinsics = np.array([[fx2, 0, cx2],
                                    [0, fy2, cy2],
                                    [0, 0, 1]])

    camera3_intrinsics = np.array([[fx3, 0, cx3],
                                    [0, fy3, cy3],
                                    [0, 0, 1]])
    
    ## Load the camera distortion coefficients, if available
    try:   
        camera1_distortion = np.load(f'{data_path}camera_distortion_{camera1_serial}.npy')
        camera2_distortion = np.load(f'{data_path}camera_distortion_{camera2_serial}.npy')
        camera3_distortion = np.load(f'{data_path}camera_distortion_{camera3_serial}.npy')
    except:
        camera1_distortion = np.zeros((5, 1))
        camera2_distortion = np.zeros((5, 1))
        camera3_distortion = np.zeros((5, 1))

    ## Generate the parameter.
    aruco_params = cv2.aruco.DetectorParameters()

    transformation_matrix_1_2 = get_transformation_matrix_between_camera1_and_camera2(
        rgb1, rgb2, camera1_intrinsics, camera2_intrinsics, camera1_distortion, camera2_distortion, board, aruco_dict, aruco_params
    )

    transformation_matrix_1_3 = get_transformation_matrix_between_camera1_and_camera2(
        rgb1, rgb3, camera1_intrinsics, camera3_intrinsics, camera1_distortion, camera3_distortion, board, aruco_dict, aruco_params
    )

    # ## Transformation from base to Lightning camera
    # T_base2wristcam, T_base2lightning_eef = get_transformation_from_base_to_wrist_camera(lightning_eef_pose)

    # ## Transformation from base to back camera
    # T_base2backcam = T_base2wristcam @ np.linalg.inv(transformation_matrix_1_2)

    # print("Saving transformation matrices to disk...")
    # np.save(f"{data_path}/transformation_matrix_1_2.npy", transformation_matrix_1_2)
    # np.save(f"{data_path}/transformation_matrix_1_3.npy", transformation_matrix_1_3)
    # np.save(f"{data_path}/T_base2wristcam.npy", T_base2wristcam)
    # np.save(f"{data_path}/T_base2lightning_eef.npy", T_base2lightning_eef)
    # np.save(f"{data_path}/T_base2backcam.npy", T_base2backcam)

    return transformation_matrix_1_2, transformation_matrix_1_3

def generate_transformation_matrices(
    num_iterations : int,
    data_path : str = "resources/camera_calibration/",
    camera1_serial : str = "130322273305",
    camera2_serial : str = "127122270512",
    camera3_serial : str = "126122270722",
    lightning_eef_pose : list = [-0.5019622959075773, 0.10288288422589228, 0.845704241201291,
                                    -1.674059653851395, 1.740895410756312, 1.1504085564683173]
    ) -> Tuple[np.ndarray, np.ndarray]:

    T_1to2 = np.zeros((num_iterations, 6))
    T_1to3 = np.zeros((num_iterations, 6))

    for i in range(num_iterations):
        T_1to2_curr, T_1to3_curr = get_transformation_matrix(
            data_path=f"{data_path}{i+1}/",
            camera1_serial=camera1_serial,
            camera2_serial=camera2_serial,
            camera3_serial=camera3_serial,
            lightning_eef_pose = lightning_eef_pose,
        )
        T_1to2[i][0:3] = T_1to2_curr[0:3, 3]
        T_1to2[i][3:6] = cv2.Rodrigues(T_1to2_curr[0:3, 0:3])[0].flatten()
        T_1to3[i][0:3] = T_1to3_curr[0:3, 3]
        T_1to3[i][3:6] = cv2.Rodrigues(T_1to3_curr[0:3, 0:3])[0].flatten()

    ## Take Mean of the transformation matrices
    T_1to2 = np.mean(T_1to2, axis=0) 
    T_1to3 = np.mean(T_1to3, axis=0)

    ## Convert to 4x4 transformation matrix
    transformation_matrix_1_2 = np.eye(4)
    transformation_matrix_1_2[0:3, 0:3] = cv2.Rodrigues(T_1to2[3:6])[0]
    transformation_matrix_1_2[0:3, 3] = T_1to2[0:3]

    transformation_matrix_1_3 = np.eye(4)
    transformation_matrix_1_3[0:3, 0:3] = cv2.Rodrigues(T_1to3[3:6])[0]
    transformation_matrix_1_3[0:3, 3] = T_1to3[0:3]

    ## Transformation from base to Lightning camera
    T_base2wristcam, T_base2lightning_eef = get_transformation_from_base_to_wrist_camera(lightning_eef_pose)

    ## Transformation from base to back camera
    T_base2backcam = T_base2wristcam @ np.linalg.inv(transformation_matrix_1_2)

    np.save(f"{data_path}transformation_matrix_1_2.npy", transformation_matrix_1_2)
    np.save(f"{data_path}transformation_matrix_1_3.npy", transformation_matrix_1_3)
    np.save(f"{data_path}T_base2wristcam.npy", T_base2wristcam)
    np.save(f"{data_path}T_base2lightning_eef.npy", T_base2lightning_eef)
    np.save(f"{data_path}T_base2backcam.npy", T_base2backcam)
    print("Saved files to location: ", data_path)

    return transformation_matrix_1_2, transformation_matrix_1_3

if __name__ == "__main__":

    num_iterations = 10
    robot_pose = np.load('resources/robot/robot_home_pose.npy', allow_pickle=True).item()
    lightning_eef_pose = robot_pose['lightning']['eef_pose']

    T_1to2, T_1to3 = generate_transformation_matrices(
        num_iterations,
        data_path="resources/camera_calibration/",
        lightning_eef_pose=lightning_eef_pose,
        camera1_serial="130322273305",
        camera2_serial="127122270512",
        camera3_serial="126122270722",
    )





    """Check Later for thunder to Base transformation."""
    # ## Create coordinate frame
    # import open3d as o3d
    # origin_coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])

    # lightning_cam_matrix = get_transformation_from_base_to_wrist_camera(lightning_capture_eff_pose)
    # lightning_cam_coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    # lightning_cam_coordinate.transform(lightning_cam_matrix)

    # ## Transformation between lightning and thunder base
    # T_base2thunder = np.eye(4)
    # X_DISPLACEMENT = 0.016
    # Y_DISPLACEMENT = 0.710
    # Z_DISPLACEMENT = 0.005
    # T_base2thunder[0, 3] = X_DISPLACEMENT
    # T_base2thunder[1, 3] = Y_DISPLACEMENT
    # T_base2thunder[2, 3] = Z_DISPLACEMENT

    # T_base2thunder[0:3, 0:3] = cv2.Rodrigues(np.array([0, 0, np.pi]))[0]
        

    # thunder_cam_matrix = get_transformation_from_base_to_wrist_camera(thunder_capture_eff_pose)
    # thunder_cam_matrix[0:3, 0:3] = thunder_cam_matrix[0:3, 0:3]
    # thunder_cam_coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    # thunder_cam_coordinate.transform(T_base2thunder @ thunder_cam_matrix)

    # back_cam_coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    # back_cam_coordinate.transform(T_robot2backcam)

    # o3d.visualization.draw_geometries([origin_coordinate,
    #                                    lightning_cam_coordinate,
    #                                    thunder_cam_coordinate,
    #                                    back_cam_coordinate,
    #                                    #pcd1, 
    #                                    #pcd2
    #                                    ], window_name="Coordinate Frames")


    # wrist_cam_coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # wrist_cam_coordinate.transform(T_w2cam)
    # pcd2.transform(T_w2cam) 

    # T_robot2cam = 
    # robot_2_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # robot_2_cam.transform(T_robot2cam)

    # o3d.visualization.draw_geometries([origin_coordinate, 
    #                                    wrist_cam_coordinate,
    #                                    pcd2,
    #                                    robot_2_cam
    #                                    #pcd1, 
    #                                    #pcd2
    #                                    ])
    
    # np.save('transformation_matrix_wrist_camera.npy', T_robot2cam)



# if __name__ == "__main__":

#     board, aruco_dict = generate_and_save_charuco_board("camera_calibration_utils/charuco_board.png")

#     camera_num_2_serial = {
#         1 : "130322273305",
#         2 : "128422270081",
#         3 : "127122270512"
#     }

#     # ## Load Images
#     rgb1 = np.load('mohit/robot_camera_calibration/color_image_130322273305.npy')
#     rgb2 = np.load('mohit/robot_camera_calibration/color_image_126122270307.npy')

#     ## Flip rgb to bgr
#     rgb1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR)
#     rgb2 = cv2.cvtColor(rgb2, cv2.COLOR_RGB2BGR)

#     depth1 = np.load('mohit/robot_camera_calibration/depth_image_130322273305.npy')
#     depth2 = np.load('mohit/robot_camera_calibration/depth_image_126122270307.npy')

#     camera1_intrinsics = np.load('mohit/robot_camera_calibration/intrinsic_130322273305.npy')
#     camera2_intrinsics = np.load('mohit/robot_camera_calibration/intrinsic_126122270307.npy')

#     # print(camera1_intrinsics)

#     cx1, cy1, fx1, fy1 = camera1_intrinsics[0], camera1_intrinsics[1], camera1_intrinsics[2], camera1_intrinsics[3]
#     cx2, cy2, fx2, fy2 = camera2_intrinsics[0], camera2_intrinsics[1], camera2_intrinsics[2], camera2_intrinsics[3]

#     camera1_intrinsics = np.array([[fx1, 0, cx1],
#                                     [0, fy1, cy1],
#                                     [0, 0, 1]])

#     camera2_intrinsics = np.array([[fx2, 0, cx2],
#                                     [0, fy2, cy2],
#                                     [0, 0, 1]])
    
#     aruco_params = cv2.aruco.DetectorParameters()
#     distortion_coeffs = np.zeros((5, 1))  # Assuming no distortion

#     # transformation_matrix_1_2 = get_transformation_matrix_between_camera1_and_camera2(
#     #     rgb1, rgb2, camera1_intrinsics, camera2_intrinsics, distortion_coeffs, distortion_coeffs, board, aruco_dict, aruco_params
#     # )

#     # np.save('transformation_martix_1_wrist_camera.npy', transformation_matrix_1_2)

#     transformation_matrix_1_2 = np.load('transformation_martix_1_wrist_camera.npy')
#     ## Create coordinate frame
#     import open3d as o3d
#     origin_coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

#     wrist_cam_coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
#     wrist_cam_coordinate.transform(transformation_matrix_1_2)

#     ## Make the point clouds
#     rgbd_image1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
#         o3d.geometry.Image(rgb1), o3d.geometry.Image(depth1), depth_scale=1/0.0001, depth_trunc=0.5, convert_rgb_to_intensity=False
#     )
#     rgbd_image2 = o3d.geometry.RGBDImage.create_from_color_and_depth(
#         o3d.geometry.Image(rgb2), o3d.geometry.Image(depth2), depth_scale=1/0.0001, depth_trunc=0.5, convert_rgb_to_intensity=False
#     )
#     o3d_intrinsic1 = o3d.camera.PinholeCameraIntrinsic()
#     o3d_intrinsic1.set_intrinsics(640, 480, fx1, fy1, cx1, cy1)
#     o3d_intrinsic2 = o3d.camera.PinholeCameraIntrinsic()
#     o3d_intrinsic2.set_intrinsics(640, 480, fx2, fy2, cx2, cy2)

#     pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(
#         rgbd_image1, o3d_intrinsic1
#     )
#     pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(
#         rgbd_image2, o3d_intrinsic2
#     )   
#     pcd1.translate([0, 0, 0.04])
#     pcd2.translate([0, 0, 0.04])
#     # pcd2.transform(transformation_matrix_1_2)


#     ## Get Robot to Camera Transformation
#     from robot import RobotController
#     robot = RobotController('lightning', False, False)
#     eef_pose = robot.get_eff_pose()
#     T_w2cam = get_transformation_from_base_to_wrist_camera(eef_pose)

#     wrist_cam_coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
#     wrist_cam_coordinate.transform(T_w2cam)
#     pcd2.transform(T_w2cam) 

#     T_robot2cam = T_w2cam @ np.linalg.inv(transformation_matrix_1_2)
#     # T_robot2cam = 
#     robot_2_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
#     robot_2_cam.transform(T_robot2cam)

#     o3d.visualization.draw_geometries([origin_coordinate, 
#                                        wrist_cam_coordinate,
#                                        pcd2,
#                                        robot_2_cam
#                                        #pcd1, 
#                                        #pcd2
#                                        ])
    
#     np.save('transformation_matrix_wrist_camera.npy', T_robot2cam)


    # transformation_matrix_1_3 = get_transformation_matrix_between_camera1_and_camera2(
    #     rgb1, rgb3, camera1_intrinsics, camera3_intrinsics, distortion_coeffs, distortion_coeffs, board, aruco_dict, aruco_params
    # )

    # print("Transformation Matrix from Camera 1 to Camera 2:")
    # print(transformation_matrix_1_2)

    # print("Transformation Matrix from Camera 1 to Camera 3:")
    # print(transformation_matrix_1_3)

    # ## Save the transformation matrix for later use.
    # np.save("camera_calibration_utils/transformation_matrix_1_2.npy", transformation_matrix_1_2)
    # np.save("camera_calibration_utils/transformation_matrix_1_3.npy", transformation_matrix_1_3)