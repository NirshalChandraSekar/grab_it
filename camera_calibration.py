import cv2
import sys
import time
import numpy as np

from typing import Tuple

def generate_and_save_charuco_board(
        filename : str = "charuco_board.png", 
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
        aruco_dict
        )
    size_ratio = squares_verticle/squares_horizontal
    board_image = cv2.aruco.CharucoBoard.generateImage(
        board,
        (640, int(640*size_ratio)), 
        marginSize=margin_size
        )
    ## Save the image
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
    charuco_corners1, charuco_ids1 = detect_charuco_corners(image1, board, aruco_dict, aruco_params, "detections1.jpg")
    charuco_corners2, charuco_ids2 = detect_charuco_corners(image2, board, aruco_dict, aruco_params, "detections2.jpg")

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
    T_base2eef[0:3, 0:3] = cv2.Rodrigues(eef_pose[3:6])[0]
    T_base2eef[0:3, 3] = eef_pose[0:3]

    ## Transformation between the wrist camera and the end effector
    T_eef2cam = np.eye(4)
    T_eef2cam[0:3, 3] = np.array([-.01, -.08, 0.01])
    ## Make rotaion matrix with 15 degree rotation aroud x axis
    T_eef2cam[0:3, 0:3] = cv2.Rodrigues(np.array([-np.pi/12, 0, 0]))[0]

    ## Transformation between the base and the wrist camera
    T_w2cam = T_base2eef @ T_eef2cam

    return T_w2cam


if __name__ == "__main__":

    # Generate and save Charuco board image
    board, aruco_dict = generate_and_save_charuco_board("camera_calibration_utils/charuco_board.png")

    camera_num_2_serial = {
        1 : "130322273305",
        2 : "128422270081",
        3 : "127122270512"
    }

    ## Load Images
    camera_data = np.load("multi_camera_calibration.npy", allow_pickle=True).item()
    rgb1 = camera_data['130322273305']['color']
    rgb2 = camera_data['128422270081']['color']
    rgb3 = camera_data['127122270512']['color']

    camera1_intrinsics = camera_data['130322273305']['intrinsics']
    camera2_intrinsics = camera_data['128422270081']['intrinsics']
    camera3_intrinsics = camera_data['127122270512']['intrinsics']

    cx1, cy1, fx1, fy1 = camera1_intrinsics['ppx'], camera1_intrinsics['ppy'], camera1_intrinsics['fx'], camera1_intrinsics['fy']
    cx2, cy2, fx2, fy2 = camera2_intrinsics['ppx'], camera2_intrinsics['ppy'], camera2_intrinsics['fx'], camera2_intrinsics['fy']   
    cx3, cy3, fx3, fy3 = camera3_intrinsics['ppx'], camera3_intrinsics['ppy'], camera3_intrinsics['fx'], camera3_intrinsics['fy']

    camera1_intrinsics = np.array([[fx1, 0, cx1],
                                    [0, fy1, cy1],
                                    [0, 0, 1]])

    camera2_intrinsics = np.array([[fx2, 0, cx2],
                                    [0, fy2, cy2],
                                    [0, 0, 1]])

    camera3_intrinsics = np.array([[fx3, 0, cx3],
                                    [0, fy3, cy3],
                                    [0, 0, 1]])
    
    aruco_params = cv2.aruco.DetectorParameters()
    distortion_coeffs = np.zeros((5, 1))  # Assuming no distortion

    transformation_matrix_1_2 = get_transformation_matrix_between_camera1_and_camera2(
        rgb1, rgb2, camera1_intrinsics, camera2_intrinsics, distortion_coeffs, distortion_coeffs, board, aruco_dict, aruco_params
    )

    transformation_matrix_1_3 = get_transformation_matrix_between_camera1_and_camera2(
        rgb1, rgb3, camera1_intrinsics, camera3_intrinsics, distortion_coeffs, distortion_coeffs, board, aruco_dict, aruco_params
    )

    print("Transformation Matrix from Camera 1 to Camera 2:")
    print(transformation_matrix_1_2)

    print("Transformation Matrix from Camera 1 to Camera 3:")
    print(transformation_matrix_1_3)

    ## Save the transformation matrix for later use.
    np.save("camera_calibration_utils/transformation_matrix_1_2.npy", transformation_matrix_1_2)
    np.save("camera_calibration_utils/transformation_matrix_1_3.npy", transformation_matrix_1_3)