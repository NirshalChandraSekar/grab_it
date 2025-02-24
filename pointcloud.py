import cv2
import numpy as np
import open3d as o3d

from typing import Tuple

def generate_pointcloud_from_images(rgb_image : np.ndarray, 
                                    depth_image : np.ndarray, 
                                    intrinsic_matrix: np.ndarray, 
                                    depth_scale : float = 1, 
                                    depth_truncation : float = None) -> o3d.geometry.PointCloud:
    """Generate a pointcloud from RGB and depth images."""
    ## Flip the color of image from BGR to RGB
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    rgb_o3d = o3d.geometry.Image(rgb_image)
    depth_o3d = o3d.geometry.Image(depth_image)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_scale, depth_truncation, convert_rgb_to_intensity=False
    )

    ## Create PointCloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_matrix)

    return pcd

def filter_pointcloud_radius_outlier_removal(
    pointcloud : o3d.geometry.PointCloud,
    radius : float = 0.005,
    min_neighbors : int = 20, 
) -> o3d.geometry.PointCloud:
    """Filter pointcloud using radius outlier removal.
    Args:
        pointcloud (o3d.geometry.PointCloud): Pointcloud to filter.
        radius (float, optional): Radius of the neighborhood. Defaults to 0.005.
        min_neighbors (int, optional): Minimum number of neighbors that the sphere must contain. Defaults to 10.
    """
    return pointcloud.remove_radius_outlier(nb_points=min_neighbors,
                                             radius=radius)

def voxel_downsample_pointcloud(pointcloud : o3d.geometry.PointCloud,
                                 voxel_size : float = 0.005) -> o3d.geometry.PointCloud:
    """Downsample pointcloud using random sampling.
    Args:
        pointcloud (o3d.geometry.PointCloud): Pointcloud to downsample.
        voxel_size (float, optional): Voxel size. Defaults to 0.005.
    """
    return pointcloud.voxel_down_sample(voxel_size=voxel_size)

def generate_fused_pointcloud_from_images(
    rs_images_path : str,
    transformation_matrix_1_2_path : str,
    transformation_matrix_1_3_path : str,
    depth_scale : float = 1/0.00025, ## For L515: 1/0.00025 For D455: 1/0.0001
    depth_truncation : float = 1.0,
    downsample_voxel_size : float = None,
    remove_outliers : bool = True,
    radius_outlier_removal : float = 0.005,
    num_neighbors_outlier_removal : int = 20,
    visualize : bool = False,
)    -> o3d.geometry.PointCloud:
    """Generate a merged pointcloud from RGB and depth images.
    Args:
        rs_images_path (str): Path to RGB and depth images.
        transformation_matrix_1_2_path (str): Path to transformation matrix from camera 1 to camera 2.
        transformation_matrix_1_3_path (str): Path to transformation matrix from camera 1 to camera 3.
        depth_scale (float, optional): Depth scale. Defaults to 1/0.00025.
        depth_truncation (float, optional): Depth truncation. Defaults to 1.0.
        downsample_voxel_size (float, optional): Voxel size for downsampling. PC not downsampled if None. Defaults to None.
        visualize (bool, optional): Whether to visualize the pointcloud. Defaults to False.
    Returns:
        o3d.geometry.PointCloud: Merged pointcloud.
    """   
    ## Load Images.
    # rs_image_data = np.load(rs_images_path, allow_pickle=True).item()
    '''
    Camera Serial to number 
    130322273305 -> 1
    128422270081 -> 2
    127122270512 -> 3
    '''
    camera_num_2_serial = {
        1 : "130322273305",
        2 : "128422270081",
        3 : "127122270512"
    }

    rgb1 = np.load(f"resources/dino/inference_color_image_{camera_num_2_serial[1]}.npy")
    rgb2 = np.load(f"resources/dino/inference_color_image_{camera_num_2_serial[2]}.npy")
    rgb3 = np.load(f"resources/dino/inference_color_image_{camera_num_2_serial[3]}.npy")

    depth1 = np.load(f"resources/dino/inference_depth_image_{camera_num_2_serial[1]}.npy")
    depth2 = np.load(f"resources/dino/inference_depth_image_{camera_num_2_serial[2]}.npy")
    depth3 = np.load(f"resources/dino/inference_depth_image_{camera_num_2_serial[3]}.npy")

    ## Load camera intrinsic parameters
    camera1_intrinsics = np.load(f"resources/dino/camera_intrinsic_{camera_num_2_serial[1]}.npy")
    camera2_intrinsics = np.load(f"resources/dino/camera_intrinsic_{camera_num_2_serial[2]}.npy")
    camera3_intrinsics = np.load(f"resources/dino/camera_intrinsic_{camera_num_2_serial[3]}.npy")

    camera1_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=rgb1.shape[1],height=rgb1.shape[0],fx=camera1_intrinsics[2], fy=camera1_intrinsics[3],cx=camera1_intrinsics[0],cy=camera1_intrinsics[1])
    camera2_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=rgb2.shape[1],height=rgb2.shape[0],fx=camera2_intrinsics[2], fy=camera2_intrinsics[3],cx=camera2_intrinsics[0],cy=camera2_intrinsics[1])
    camera3_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=rgb3.shape[1],height=rgb3.shape[0],fx=camera3_intrinsics[2], fy=camera3_intrinsics[3],cx=camera3_intrinsics[0],cy=camera3_intrinsics[1])

    ## Generate pointclouds from Images.
    pointcloud1 = generate_pointcloud_from_images(rgb1, depth1, camera1_intrinsics, depth_scale, depth_truncation)
    pointcloud2 = generate_pointcloud_from_images(rgb2, depth2, camera2_intrinsics, depth_scale, depth_truncation)
    pointcloud3 = generate_pointcloud_from_images(rgb3, depth3, camera3_intrinsics, depth_scale, depth_truncation)

    z_displacement = 0.04
    pointcloud1.transform([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, z_displacement],
                            [0, 0, 0, 1]])
    pointcloud2.transform([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, z_displacement],
                            [0, 0, 0, 1]])
    pointcloud3.transform([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, z_displacement],
                            [0, 0, 0, 1]])

    ## Load the transformation matrix
    transformation_matrix_1_2 = np.load(transformation_matrix_1_2_path)
    transformation_matrix_1_3 = np.load(transformation_matrix_1_3_path)

    ## Filter Pointclouds to remove outliers (_ are indices)
    if remove_outliers:
        pointcloud1, _ = filter_pointcloud_radius_outlier_removal(pointcloud1, radius=radius_outlier_removal, min_neighbors=num_neighbors_outlier_removal)
        pointcloud2, _ = filter_pointcloud_radius_outlier_removal(pointcloud2, radius=radius_outlier_removal, min_neighbors=num_neighbors_outlier_removal)
        pointcloud3, _ = filter_pointcloud_radius_outlier_removal(pointcloud3, radius=radius_outlier_removal, min_neighbors=num_neighbors_outlier_removal)

    ## Transform pointcloud2 and pointcloud3 to camera1 coordinate frame
    pointcloud2.transform(transformation_matrix_1_2)
    pointcloud3.transform(transformation_matrix_1_3)

    ## Merge the pointclouds into a single pointcloud
    fused_pointcloud = pointcloud1 + pointcloud2 + pointcloud3

    ## Downsample the merged pointcloud
    if downsample_voxel_size is not None:
        fused_pointcloud = fused_pointcloud.voxel_down_sample(voxel_size=downsample_voxel_size)

    if visualize:
        ## Visualize pointclouds
        camera1_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        camera2_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        camera3_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        camera2_coordinate_frame.transform(transformation_matrix_1_2)
        camera3_coordinate_frame.transform(transformation_matrix_1_3)
        o3d.visualization.draw_geometries([fused_pointcloud, 
                                        camera1_coordinate_frame, 
                                        camera2_coordinate_frame, 
                                        camera3_coordinate_frame], 
                                        window_name="Point Cloud")
    
    return fused_pointcloud
    
def generate_antipodal_point_pairs(pointcloud : o3d.geometry.PointCloud,
                                   max_gripper_width : float = 0.08,
                                   closest_distance : float = 0.01,
                                   angle_threshold : float = np.pi/18, ## 10 degrees
                                   neighborhood_size : int = 30,) -> np.ndarray:
    """Generate antipodal point pairs from a pointcloud."""

    ## Generate normals for the pointcloud
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=neighborhood_size))
    pointcloud.normalize_normals()

    points = np.asarray(pointcloud.points)
    normals = np.asarray(pointcloud.normals)

    antipodal_pairs = []

    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            dist_between_points = np.linalg.norm(points[i] - points[j])
            if dist_between_points < max_gripper_width and dist_between_points > closest_distance:  ## Skip points that are too close or too far
                normal1 = normals[i]
                normal2 = normals[j]

                dot_product_of_normals = np.dot(normal1, normal2)
                if (dot_product_of_normals > 0 and dot_product_of_normals > np.cos(angle_threshold)) or (dot_product_of_normals < 0 and dot_product_of_normals < -np.cos(angle_threshold)):
                    antipodal_pairs.append((i, j)) 
    return np.array(antipodal_pairs)

def find_antipodal_point_wrt_query_point(query_point : np.ndarray,
                                  normal_set_point : np.ndarray,
                                  pointcloud : o3d.geometry.PointCloud,
                                  max_gripper_width : float = 0.08,
                                  closest_distance_threshold : float = 0.005, ## 5 mm
                                  angle_threshold : float = np.pi/6, ## 20 degrees
                                  neighborhood_size : int = 30,) -> Tuple[np.ndarray, np.ndarray]:
    """Find antipodal point to a query point in the pointcloud."""
    ## Generate normals for the pointcloud
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=neighborhood_size))
    pointcloud.normalize_normals()

    points = np.asarray(pointcloud.points)
    normals = np.asarray(pointcloud.normals)

    ## Set the normal to be outward facing from the normal set point
    for i in range(points.shape[0]):
        direction_from_normal_set_point = points[i] - normal_set_point
        ## Normal of point should be in the same direction as the direction from normal set point
        if np.dot(normals[i], direction_from_normal_set_point) < 0:
            normals[i] = -normals[i]
    ## Normalize the normals
    # normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

    ## Find the closest point to the query point
    distances = np.linalg.norm(points - query_point, axis=1)
    closest_point_idx = np.argmin(distances)
    closest_point = points[closest_point_idx]
    closest_point_normal = normals[closest_point_idx]

    antipodal_points_idx = []
    ## Find the antipodal point
    for i in range(points.shape[0]):
        dist_between_points = np.linalg.norm(points[i] - closest_point)
        if dist_between_points < max_gripper_width and dist_between_points > closest_distance_threshold:  ## Skip points that are too close or too far
            normal = normals[i]
            dot_product_of_normals = np.dot(closest_point_normal, normal)
            if dot_product_of_normals < -np.cos(angle_threshold):
            # if (dot_product_of_normals > 0 and dot_product_of_normals > np.cos(angle_threshold)) or (dot_product_of_normals < 0 and dot_product_of_normals < -np.cos(angle_threshold)):
                antipodal_points_idx.append(i)
    return closest_point_idx, np.array(antipodal_points_idx)

def get_gripper_pose_plane_based(rotation_matrix : np.ndarray,
                    pointcloud : o3d.geometry.PointCloud,
                    max_gripper_width : float = 0.08,
                    plane_threshold : float = 0.005,
                    closest_distance : float = 0.01,
                    angle_threshold : float = np.pi/18, ## 10 degrees
                    neighborhood_size : int = 30,) -> Tuple[int, np.ndarray]:
    """Get the gripper pose from the pointcloud."""

    points = np.asarray(pointcloud.points)
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=neighborhood_size))
    pointcloud.normalize_normals()

    ## Get idx of the closest point to the gripper point
    distances = np.linalg.norm(points - rotation_matrix[:3, 3].T, axis=1).reshape(-1, 1)
    grasp_point_idx = np.argmin(distances)
    grasp_point = points[grasp_point_idx]
    grasp_point_normal = np.array(pointcloud.normals)[grasp_point_idx]

    ## Filter the points within the gripper width
    distance_from_gripper_point = np.linalg.norm(points - grasp_point, axis=1).reshape(-1, 1)
    idx_within_gripper_width = np.where(((distance_from_gripper_point < max_gripper_width) & (distance_from_gripper_point > closest_distance)) | (distance_from_gripper_point == 0))[0]
    print("Shape of idx_within_gripper_width: ", idx_within_gripper_width.shape)
    # idx_within_gripper_width = np.concatenate((idx_within_gripper_width, [grasp_point_idx]), axis=0)
    print("Shape of idx_within_gripper_width: ", idx_within_gripper_width.shape)
    pcd_filtered_by_distance = pointcloud.select_by_index(idx_within_gripper_width)

    ## Filter Points on the gripper plane
    d = -np.dot(rotation_matrix[:3, 3], rotation_matrix[:3, 0]) ## Dot product of point andnormal(x-axis)
    distance_from_gripper_plane = np.abs(np.dot(points[idx_within_gripper_width], rotation_matrix[:3, 0]) + d) / np.linalg.norm(rotation_matrix[:3, 0])
    idx_in_plane = np.where(distance_from_gripper_plane < plane_threshold)[0]

    pcd_filtered_by_distance_and_plane = pcd_filtered_by_distance.select_by_index(idx_in_plane)

    ## Get the antipodal metric
    pcd_normals = np.asarray(pcd_filtered_by_distance_and_plane.normals)

    antipodal_metric = np.abs(np.dot(pcd_normals, grasp_point_normal)) ## There are Cos Angles
    antipodal_pair_idx = np.argsort(antipodal_metric)[-2]
    antipodal_point = np.asarray(pcd_filtered_by_distance_and_plane.points)[antipodal_pair_idx]

    ## Make a sphere around the antipodal point
    antipodal_point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    antipodal_point_sphere.translate(antipodal_point)
    antipodal_point_sphere.paint_uniform_color([0, 0, 1]) ## Blue color

    ## Make a sphere aroud the grasp point
    grasp_point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    grasp_point_sphere.translate(grasp_point)
    grasp_point_sphere.paint_uniform_color([0, 0, 1]) ## Blue color

    camera1_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd_filtered_by_distance_and_plane,
                                        camera1_coordinate_frame,
                                        grasp_point_sphere,
                                        antipodal_point_sphere
                                        ],
                                        point_show_normal=True, 
                                        window_name="Filtered Point Cloud")


def get_gripper_pose(rotation_matrix : np.ndarray,
                    pointcloud : o3d.geometry.PointCloud,
                    max_gripper_width : float = 0.08,
                    plane_threshold : float = 0.005,
                    closest_distance : float = 0.01,
                    angle_threshold : float = np.pi/18, ## 10 degrees
                    neighborhood_size : int = 30,) -> Tuple[int, np.ndarray]:
    """Get the gripper pose from the pointcloud."""

    points = np.asarray(pointcloud.points)
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=neighborhood_size))
    pointcloud.normalize_normals()

    ## Get idx of the closest point to the gripper point
    distances = np.linalg.norm(points - rotation_matrix[:3, 3].T, axis=1).reshape(-1, 1)
    grasp_point_idx = np.argmin(distances)
    grasp_point = points[grasp_point_idx]
    grasp_point_normal = np.array(pointcloud.normals)[grasp_point_idx]

    # ## Filter the points within the gripper width
    distance_from_gripper_point = np.linalg.norm(points - grasp_point, axis=1).reshape(-1, 1)
    idx_within_gripper_width = np.where((distance_from_gripper_point < max_gripper_width) & (distance_from_gripper_point > closest_distance))[0] # | (distance_from_gripper_point == 0))[0]
    pcd_filtered_by_distance = pointcloud.select_by_index(idx_within_gripper_width)

    points_pcd_filtered = np.asarray(pcd_filtered_by_distance.points)
    ## Distance of points to line from gripper point in normal direction.
    distance_from_gripper_normal = np.linalg.norm(np.cross((points_pcd_filtered - grasp_point), grasp_point_normal), axis=1) / np.linalg.norm(grasp_point_normal)
    antipodal_point = points_pcd_filtered[np.argmin(distance_from_gripper_normal)]

    ## Mid Point
    offset_distance = 0.1
    mid_point = (grasp_point + antipodal_point) / 2
    gripper_translation = mid_point - offset_distance*rotation_matrix[:3, 0].T

    x_axis = (grasp_point - mid_point) / np.linalg.norm(grasp_point - mid_point)
    z_axis = rotation_matrix[:3, 0].T
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    gripper_pose = np.eye(4)
    gripper_pose[:3, 0] = x_axis
    gripper_pose[:3, 1] = y_axis
    gripper_pose[:3, 2] = z_axis
    gripper_pose[:3, 3] = gripper_translation

    ## Gripper Coordinate Frame
    gripper_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    gripper_coordinate_frame.transform(gripper_pose)

    ## Make a sphere around the antipodal point
    antipodal_point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    antipodal_point_sphere.translate(antipodal_point)
    antipodal_point_sphere.paint_uniform_color([0, 0, 1]) ## Blue color

    ## Make a sphere aroud the grasp point
    grasp_point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    grasp_point_sphere.translate(grasp_point)
    grasp_point_sphere.paint_uniform_color([0, 0, 1]) ## Blue color

    camera1_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    ## Visualize the pointcloud
    o3d.visualization.draw_geometries([pcd_filtered_by_distance,
                                        camera1_coordinate_frame,
                                        grasp_point_sphere,
                                        gripper_coordinate_frame,
                                        antipodal_point_sphere
                                        ],
                                        point_show_normal=False, 
                                        window_name="Filtered Point Cloud")



    # ## Filter Points on the gripper plane
    # d = -np.dot(rotation_matrix[:3, 3], rotation_matrix[:3, 0]) ## Dot product of point andnormal(x-axis)
    # distance_from_gripper_plane = np.abs(np.dot(points[idx_within_gripper_width], rotation_matrix[:3, 0]) + d) / np.linalg.norm(rotation_matrix[:3, 0])
    # idx_in_plane = np.where(distance_from_gripper_plane < plane_threshold)[0]

    # pcd_filtered_by_distance_and_plane = pcd_filtered_by_distance.select_by_index(idx_in_plane)

    # ## Get the antipodal metric
    # pcd_normals = np.asarray(pcd_filtered_by_distance_and_plane.normals)

    # antipodal_metric = np.abs(np.dot(pcd_normals, grasp_point_normal)) ## There are Cos Angles
    # antipodal_pair_idx = np.argsort(antipodal_metric)[-2]
    # antipodal_point = np.asarray(pcd_filtered_by_distance_and_plane.points)[antipodal_pair_idx]

    # ## Make a sphere around the antipodal point
    # antipodal_point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    # antipodal_point_sphere.translate(antipodal_point)
    # antipodal_point_sphere.paint_uniform_color([0, 0, 1]) ## Blue color

    # ## Make a sphere aroud the grasp point
    # grasp_point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    # grasp_point_sphere.translate(grasp_point)
    # grasp_point_sphere.paint_uniform_color([0, 0, 1]) ## Blue color

    # camera1_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd_filtered_by_distance_and_plane,
    #                                     camera1_coordinate_frame,
    #                                     grasp_point_sphere,
    #                                     antipodal_point_sphere
    #                                     ],
    #                                     point_show_normal=True, 
    #                                     window_name="Filtered Point Cloud")

if __name__ == "__main__":
    camera_num_2_serial = {
        1 : "130322273305",
        2 : "128422270081",
        3 : "127122270512"
    }
    
    # fused_pointcloud = generate_fused_pointcloud_from_images(
    #                         rs_images_path="multi_camera_pouch.npy",
    #                         transformation_matrix_1_2_path="camera_calibration_utils/transformation_matrix_1_2.npy",
    #                         transformation_matrix_1_3_path="camera_calibration_utils/transformation_matrix_1_3.npy",
    #                         downsample_voxel_size=0.003, ## This will speed up the processing significantly. Keep 0.01
    #                         visualize=True,
    #                         remove_outliers=True,
    #                         radius_outlier_removal=0.005,
    #                         num_neighbors_outlier_removal=20,
    #                         depth_scale=1/0.0001,
    #                         depth_truncation=0.4
    #                     )
    # ## Save the PCD
    # o3d.io.write_point_cloud("fused_pointcloud_dino.pcd", fused_pointcloud)

    ## Load the PCD
    fused_pointcloud = o3d.io.read_point_cloud("fused_pointcloud_dino.pcd")
    ### Camera 3 view.
    ## Load transformation matrix
    transformation_matrix_1_3 = np.load("camera_calibration_utils/transformation_matrix_1_3.npy")
    transformation_matrix_1_2 = np.load("camera_calibration_utils/transformation_matrix_1_2.npy")

    ## Input is point where the human Gripped the Item and the Rotation metrix.

    gripper_point = np.array([-0.0445967326784003, 0.00028400676882089546, 0.297 + 0.04])
    gripper_rotation_matrix = np.array([[0.88892108,  0.19825181,  0.41293526],
                                        [0.42529549, -0.02239377, -0.90477747],
                                        [-0.17012659,  0.97989527, -0.10422189]])

    gripper_matrix = np.eye(4)
    gripper_matrix[:3, 3] = gripper_point
    gripper_matrix[:3, :3] = gripper_rotation_matrix

    ## Trasform the gripper point to camera 1 coordinate frame
    gripper_matrix = transformation_matrix_1_3 @ gripper_matrix

    gripper_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    gripper_coordinate_frame.transform(gripper_matrix)

    # closest_point_idx, antipodal_point_idxs = find_antipodal_point_wrt_query_point(
    #     gripper_point,
    #     #normal_set_point,
    #     fused_pointcloud,
    #     max_gripper_width=0.08,
    #     closest_distance_threshold=0.005,
    #     angle_threshold=np.pi/18,
    #     neighborhood_size=30)

    ## Get the gripper pose
    gripper_pose = get_gripper_pose(
        rotation_matrix=gripper_matrix,
        pointcloud=fused_pointcloud,
        max_gripper_width=0.08,
        plane_threshold=0.005,
        closest_distance=0.02,
        angle_threshold=np.pi/18, ## 10 degrees
        neighborhood_size=30,)
        

    exit()
    
    print(f"Closest point index: {closest_point_idx}")
    print(f"Length of antipodal points: {len(antipodal_point_idxs)}")
    if len(antipodal_point_idxs) == 0:
        antipodal_point_idxs = np.array([closest_point_idx])
        
    antipodal_points = np.asarray(fused_pointcloud.points)[antipodal_point_idxs]

    ## Generate pointcloud of antipodal points
    antipodal_pointcloud = o3d.geometry.PointCloud()
    antipodal_pointcloud.points = o3d.utility.Vector3dVector(antipodal_points)
    antipodal_pointcloud.paint_uniform_color([0, 0, 1])  ## Red color

    ## Show the closest point as sphere
    closest_point = np.asarray(fused_pointcloud.points)[closest_point_idx]
    closest_point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    closest_point_sphere.translate(closest_point)
    closest_point_sphere.paint_uniform_color([0, 1, 0])  ## Green color

    gripper_point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    gripper_point_sphere.translate(gripper_point)
    gripper_point_sphere.paint_uniform_color([1, 1, 0])  ## Yellow color

    normal_set_point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    normal_set_point_sphere.translate(normal_set_point)
    normal_set_point_sphere.paint_uniform_color([0, 0, 1])  ## Blue color

    ## Visualize the pointcloud
    ## Normals
    fused_pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    camera2_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    camera2_coordinate_frame.transform(transformation_matrix_1_2)

    camera3_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    camera3_coordinate_frame.transform(transformation_matrix_1_3)
    
    o3d.visualization.draw_geometries([fused_pointcloud, 
                                       normal_set_point_sphere,
                                       antipodal_pointcloud,
                                       closest_point_sphere,
                                       origin_frame,
                                       gripper_point_sphere,
                                       camera2_coordinate_frame,
                                       camera3_coordinate_frame], 
                                       window_name="Point Cloud with Antipodal Points",
                                       point_show_normal=False,)