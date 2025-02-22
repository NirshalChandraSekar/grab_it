import cv2
import numpy as np
import open3d as o3d

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
    depth_scale : float = 1/0.00025, ## For L515
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
    rs_image_data = np.load(rs_images_path, allow_pickle=True).item()
    '''
    Camera Serial to number 
    130322273305 -> 1
    128422270081 -> 2
    127122270512 -> 3
    '''

    rgb1 = rs_image_data['130322273305']['color']
    rgb2 = rs_image_data['128422270081']['color']
    rgb3 = rs_image_data['127122270512']['color']

    depth1 = rs_image_data['130322273305']['depth']
    depth2 = rs_image_data['128422270081']['depth']
    depth3 = rs_image_data['127122270512']['depth']

    ## Load camera intrinsic parameters
    camera1_intrinsics = rs_image_data['130322273305']['intrinsics']
    camera2_intrinsics = rs_image_data['128422270081']['intrinsics']
    camera3_intrinsics = rs_image_data['127122270512']['intrinsics']

    camera1_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=rgb1.shape[1],height=rgb1.shape[0],fx=camera1_intrinsics['fx'],fy=camera1_intrinsics['fy'],cx=camera1_intrinsics['ppx'],cy=camera1_intrinsics['ppy'])
    camera2_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=rgb2.shape[1],height=rgb2.shape[0],fx=camera2_intrinsics['fx'],fy=camera2_intrinsics['fy'],cx=camera2_intrinsics['ppx'],cy=camera2_intrinsics['ppy'])
    camera3_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=rgb3.shape[1],height=rgb3.shape[0],fx=camera3_intrinsics['fx'],fy=camera3_intrinsics['fy'],cx=camera3_intrinsics['ppx'],cy=camera3_intrinsics['ppy'])

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

if __name__ == "__main__":
    fused_pointcloud = generate_fused_pointcloud_from_images(
                            rs_images_path="multi_camera_pouch.npy",
                            transformation_matrix_1_2_path="transformation_matrix_1_2.npy",
                            transformation_matrix_1_3_path="transformation_matrix_1_3.npy",
                            downsample_voxel_size=0.01, ## This will speed up the processing significantly. Keep 0.01
                            visualize=True,
                            remove_outliers=True,
                            radius_outlier_removal=0.005,
                            num_neighbors_outlier_removal=20,
                            depth_scale=1/0.0001,
                            depth_truncation=0.3
                        )

    # antipodal_pairs = generate_antipodal_point_pairs(fused_pointcloud)

