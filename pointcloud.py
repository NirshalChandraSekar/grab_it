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
    image1_path : str,
    image2_path : str,
    image3_path : str,
    transformation_matrix_1_2_path : str,
    transformation_matrix_1_3_path : str,
    depth_scale : float = 1/0.00025, ## For L515
    depth_truncation : float = 1.0,
    downsample_voxel_size : float = None,
    visualize : bool = False,
)    -> o3d.geometry.PointCloud:
    """Generate a merged pointcloud from RGB and depth images.
    Args:
        image1_path (str): Path to image 1.
        image2_path (str): Path to image 2.
        image3_path (str): Path to image 3.
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
    image1_path = "camera_calibration_images/2/"
    image2_path = "camera_calibration_images/3/"
    image3_path = "camera_calibration_images/3/"

    rgb1 = np.load(f"{image1_path}color_image.npy")
    rgb2 = np.load(f"{image2_path}color_image.npy")
    rgb3 = np.load(f"{image3_path}color_image.npy")

    depth1 = np.load(f"{image1_path}depth_image.npy")
    depth2 = np.load(f"{image2_path}depth_image.npy")
    depth3 = np.load(f"{image3_path}depth_image.npy")

    ## Load camera intrinsic parameters
    camera1_intrinsics = np.load(f"{image1_path}intrinsic.npy")
    camera2_intrinsics = np.load(f"{image2_path}intrinsic.npy")
    camera3_intrinsics = np.load(f"{image3_path}intrinsic.npy")

    camera1_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=rgb1.shape[1],height=rgb1.shape[0],fx=camera1_intrinsics[2],fy=camera1_intrinsics[3],cx=camera1_intrinsics[0],cy=camera1_intrinsics[1])
    camera2_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=rgb2.shape[1],height=rgb2.shape[0],fx=camera2_intrinsics[2],fy=camera2_intrinsics[3],cx=camera2_intrinsics[0],cy=camera2_intrinsics[1])
    camera3_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=rgb3.shape[1],height=rgb3.shape[0],fx=camera3_intrinsics[2],fy=camera3_intrinsics[3],cx=camera3_intrinsics[0],cy=camera3_intrinsics[1])

    ## Generate pointclouds from Images.
    pointcloud1 = generate_pointcloud_from_images(rgb1, depth1, camera1_intrinsics, depth_scale, depth_truncation)
    pointcloud2 = generate_pointcloud_from_images(rgb2, depth2, camera2_intrinsics, depth_scale, depth_truncation)
    pointcloud3 = generate_pointcloud_from_images(rgb3, depth3, camera3_intrinsics, depth_scale, depth_truncation)

    ## Load the transformation matrix
    transformation_matrix_1_2 = np.load(transformation_matrix_1_2_path)
    transformation_matrix_1_3 = np.load(transformation_matrix_1_3_path)

    ## Filter Pointclouds to remove outliers
    pointcloud1, indices1 = filter_pointcloud_radius_outlier_removal(pointcloud1, radius=0.005, min_neighbors=20)
    pointcloud2, indices2 = filter_pointcloud_radius_outlier_removal(pointcloud2, radius=0.005, min_neighbors=20)
    pointcloud3, indices3 = filter_pointcloud_radius_outlier_removal(pointcloud3, radius=0.005, min_neighbors=20)

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
                            image1_path="camera_calibration_images/1/",
                            image2_path="camera_calibration_images/2/",
                            image3_path="camera_calibration_images/3/",
                            transformation_matrix_1_2_path="transformation_matrix_1_2.npy",
                            transformation_matrix_1_3_path="transformation_matrix_1_3.npy",
                            downsample_voxel_size=0.01, ## This will speed up the processing significantly.
                            visualize=False,
                        )

    antipodal_pairs = generate_antipodal_point_pairs(fused_pointcloud)

