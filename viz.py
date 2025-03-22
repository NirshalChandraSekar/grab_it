import open3d as o3d
import numpy as np
import cv2
from sklearn.decomposition import PCA 
from sklearn.linear_model import RANSACRegressor
from scipy.spatial.transform import Rotation as R

def pca_2d(pixels, center, image):
    # pca_image = image.copy()
    # pca_image = cv2.cvtColor(pca_image, cv2.COLOR_RGB2BGR)  
    # for key in pixels:
    #     pca = PCA(n_components=2)
    #     pca.fit(pixels[key])
    #     direction = pca.components_[0]
    #     length = 30
    #     p1 = (int(center[key][0] - length * direction[0]), int(center[key][1] - length * direction[1]))
    #     p2 = (int(center[key][0] + length * direction[0]), int(center[key][1] + length * direction[1]))
    #     cv2.line(pca_image, p1, p2, (0, 255, 0), 2)
    #     cv2.circle(pca_image, center[key], 3, (0, 0, 255), -1)
    

    # cv2.imshow("PCA Line Overlay", pca_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    pca_image = image.copy()
    pca_image = cv2.cvtColor(pca_image, cv2.COLOR_RGB2BGR)  

    for key in pixels:
        points = np.array(pixels[key])

        # Fit PCA to find the main direction
        pca = PCA(n_components=2)

        # Use RANSAC to remove outliers
        x_coords = points[:, 0].reshape(-1, 1)
        y_coords = points[:, 1]

        ransac = RANSACRegressor()
        ransac.fit(x_coords, y_coords)
        inlier_mask = ransac.inlier_mask_  # Identified inliers

        # Recompute PCA on inliers only
        inlier_points = points[inlier_mask]
        pca.fit(inlier_points)
        direction = pca.components_[0]

        # Extend line in both directions
        length = 30
        p1 = (int(center[key][0] - length * direction[0]), int(center[key][1] - length * direction[1]))
        p2 = (int(center[key][0] + length * direction[0]), int(center[key][1] + length * direction[1]))

        # Draw the robust PCA line
        cv2.line(pca_image, p1, p2, (0, 255, 0), 2)
        cv2.circle(pca_image, center[key], 3, (0, 0, 255), -1)

    cv2.imshow("Robust PCA Line Overlay", pca_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pca_3d(points, intrinsics_, depth_image, color_image, contact_point, inference_directional_point):
    # Extract camera intrinsic parameters
    cx, cy, fx, fy = intrinsics_

    # Create Open3D camera intrinsic object
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        color_image.shape[1],  # Image width
        color_image.shape[0],  # Image height
        fx, fy, cx, cy
    )

    # Create RGBD image from color and depth
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image),
        o3d.geometry.Image(depth_image),
        depth_scale=1.0,
        depth_trunc=0.5,  # Adjust depth truncation as needed
        convert_rgb_to_intensity=False
    )

    # Create point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

    # Convert 2D points to 3D and filter invalid points
    v_coords = points[:, 0]  # v-coordinates (rows)
    u_coords = points[:, 1]  # u-coordinates (columns)

    valid_x, valid_y, valid_z = [], [], []

    # Convert each 2D point to 3D
    for v, u in zip(v_coords, u_coords):
        z = depth_image[int(v), int(u)]  # Depth value at (v, u)

        # Check if depth value is valid
        if np.isfinite(z) and 0 < z < 0.5:  # Adjust depth truncation as needed
            x = (u - cx) * z / fx  # X coordinate
            y = (v - cy) * z / fy  # Y coordinate
            valid_x.append(x)
            valid_y.append(y)
            valid_z.append(z)

    # Store valid 3D points
    points_3d = np.column_stack([valid_x, valid_y, valid_z])

    # Convert contact point to 3D
    z_contact = depth_image[int(contact_point[0]), int(contact_point[1])]
    print('debug z contact', z_contact)
    if np.isfinite(z_contact) and 0 < z_contact < 0.5:
        contact_point_3d = np.array([
            (contact_point[1] - cx) * z_contact / fx,
            (contact_point[0] - cy) * z_contact / fy,
            z_contact
        ])
    else:
        contact_point_3d = None  # Mark as invalid

    # Convert directional point to 3D
    z_dir = depth_image[int(inference_directional_point[0]), int(inference_directional_point[1])]
    print('debug z dir', z_dir)
    if np.isfinite(z_dir) and 0 < z_dir < 0.5:
        directional_point_3d = np.array([
            (inference_directional_point[1] - cx) * z_dir / fx,
            (inference_directional_point[0] - cy) * z_dir / fy,
            z_dir
        ])
    else:
        directional_point_3d = None  # Mark as invalid

    # Perform PCA if there are valid points
    if len(points_3d) == 0:
        print("No valid 3D points. Skipping PCA.")
        return None

    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(points_3d)

    # Store PCA results
    grasp_axis = {
        'center': contact_point_3d,
        'axes': pca.components_
    }

    # Adjust the first principal axis to align with the inferred direction
    if directional_point_3d is not None and contact_point_3d is not None:
        dir_vector = directional_point_3d - contact_point_3d
        if np.dot(dir_vector, grasp_axis['axes'][0]) < 0:
            grasp_axis['axes'][0] = -grasp_axis['axes'][0]

    # Ensure the axes are orthogonal
    grasp_axis['axes'][1] = np.cross(grasp_axis['axes'][2], grasp_axis['axes'][0])
    grasp_axis['axes'][1] /= np.linalg.norm(grasp_axis['axes'][1])
    grasp_axis['axes'][2] = np.cross(grasp_axis['axes'][0], grasp_axis['axes'][1])


    # Visualize the results
    if grasp_axis['center'] is not None:
        center = np.array(grasp_axis['center'])
        axes = grasp_axis['axes']

        # Scale the axes for visualization
        scale = 0.5
        axis_points = np.array([
            center, center + scale * axes[0],  # Principal axis 1
            center, center + scale * axes[1],  # Principal axis 2
            center, center + scale * axes[2]   # Principal axis 3
        ])

        # Create a line set for visualization
        lines = [[0, 1], [2, 3], [4, 5]]
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Red, Green, Blue
        axis_lines = o3d.geometry.LineSet()
        axis_lines.points = o3d.utility.Vector3dVector(axis_points)
        axis_lines.lines = o3d.utility.Vector2iVector(lines)
        axis_lines.colors = o3d.utility.Vector3dVector(colors)

        # Create a camera coordinate frame for visualization
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

        # Visualize the point cloud and PCA axes
        o3d.visualization.draw_geometries([pcd, camera, axis_lines])

    grasp_axis['axes'] = grasp_axis['axes'].T  # Transpose the axes

    # Convert to transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = grasp_axis['axes']
    transformation_matrix[:3, 3] = grasp_axis['center']
    return transformation_matrix




# def pca_3d(points, intrinsics_, depth_image, color_image, contact_point, inference_directional_point):
#     # Extract camera intrinsic parameters
#     cx, cy, fx, fy = intrinsics_

#     # Create Open3D camera intrinsic object
#     intrinsics = o3d.camera.PinholeCameraIntrinsic(
#         color_image.shape[1],  # Image width
#         color_image.shape[0],  # Image height
#         fx, fy, cx, cy
#     )

#     # Create RGBD image from color and depth
#     rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#         o3d.geometry.Image(color_image),
#         o3d.geometry.Image(depth_image),
#         depth_scale=1.0,
#         depth_trunc=0.5,  # Adjust depth truncation as needed
#         convert_rgb_to_intensity=False
#     )

#     # Create point cloud from RGBD image
#     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

#     # Initialize dictionaries to store 3D points and axes
#     points_3d = {}
#     contact_point_3d = {}
#     directional_point_3d = {}
#     grasp_axis = {}

#     # Convert 2D points to 3D and filter invalid points
#     for key in points:
#         v_coords = points[key][:, 0]  # v-coordinates (rows)
#         u_coords = points[key][:, 1]  # u-coordinates (columns)

#         # Initialize lists to store valid 3D points
#         valid_x = []
#         valid_y = []
#         valid_z = []

#         # Convert each 2D point to 3D
#         for v, u in zip(v_coords, u_coords):
#             z = depth_image[int(v), int(u)]  # Depth value at (v, u)

#             # Check if depth value is valid
#             if np.isfinite(z) and z > 0 and z < 3:  # Adjust depth truncation as needed
#                 x = (u - cx) * z / fx  # X coordinate
#                 y = (v - cy) * z / fy  # Y coordinate
#                 valid_x.append(x)
#                 valid_y.append(y)
#                 valid_z.append(z)

#         # Store valid 3D points
#         points_3d[key] = np.column_stack([valid_x, valid_y, valid_z])

#         # Convert contact point to 3D
#         z_contact = depth_image[int(contact_point[key][0]), int(contact_point[key][1])]
#         print('debug z contact', z_contact)
#         if np.isfinite(z_contact) and z_contact > 0 and z_contact < 0.5:
#             contact_point_3d[key] = [
#                 (contact_point[key][1] - cx) * z_contact / fx,
#                 (contact_point[key][0] - cy) * z_contact / fy,
#                 z_contact
#             ]
#         else:
#             contact_point_3d[key] = None  # Mark as invalid

#         # Convert directional point to 3D
#         z_dir = depth_image[int(inference_directional_point[key][0]), int(inference_directional_point[key][1])]
#         print('debug z dir', z_dir)
#         if np.isfinite(z_dir) and z_dir > 0 and z_dir < 0.5:
#             directional_point_3d[key] = np.array([
#                 (inference_directional_point[key][1] - cx) * z_dir / fx,
#                 (inference_directional_point[key][0] - cy) * z_dir / fy,
#                 z_dir
#             ])
#         else:
#             directional_point_3d[key] = None  # Mark as invalid

#     # Compute PCA for each set of 3D points
#     for key in points_3d:
#         if len(points_3d[key]) == 0:
#             print(f"No valid 3D points for key {key}. Skipping PCA.")
#             continue

#         # Perform PCA
#         pca = PCA(n_components=3)
#         pca.fit(points_3d[key])

#         # Store PCA results
#         grasp_axis[key] = {
#             'center': contact_point_3d[key],
#             'axes': pca.components_
#         }

#         # Adjust the first principal axis to align with the inferred direction
#         if directional_point_3d[key] is not None and contact_point_3d[key] is not None:
#             dir_vector = directional_point_3d[key] - contact_point_3d[key]
#             if np.dot(dir_vector, grasp_axis[key]['axes'][0]) < 0:
#                 grasp_axis[key]['axes'][0] = -grasp_axis[key]['axes'][0]

#         # Ensure the axes are orthogonal
#         grasp_axis[key]['axes'][1] = np.cross(grasp_axis[key]['axes'][2], grasp_axis[key]['axes'][0])
#         grasp_axis[key]['axes'][1] /= np.linalg.norm(grasp_axis[key]['axes'][1])
#         grasp_axis[key]['axes'][2] = np.cross(grasp_axis[key]['axes'][0], grasp_axis[key]['axes'][1])

#     # Visualize the results
#     axis_lines = {}
#     for key in grasp_axis:
#         if grasp_axis[key]['center'] is None:
#             continue

#         center = np.array(grasp_axis[key]['center'])
#         axes = grasp_axis[key]['axes']

#         # Scale the axes for visualization
#         scale = 0.5
#         axis_points = np.array([
#             center, center + scale * axes[0],  # Principal axis 1
#             center, center + scale * axes[1],  # Principal axis 2
#             center, center + scale * axes[2]   # Principal axis 3
#         ])

#         # Create a line set for visualization
#         lines = [[0, 1], [2, 3], [4, 5]]
#         colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Red, Green, Blue
#         axis_lines[key] = o3d.geometry.LineSet()
#         axis_lines[key].points = o3d.utility.Vector3dVector(axis_points)
#         axis_lines[key].lines = o3d.utility.Vector2iVector(lines)
#         axis_lines[key].colors = o3d.utility.Vector3dVector(colors)

#     # Create a camera coordinate frame for visualization
#     camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

#     # Combine all objects for visualization
#     objects_to_visualize = [pcd, camera]
#     for key in axis_lines:
#         objects_to_visualize.append(axis_lines[key])

#     # Visualize the point cloud and PCA axes
#     o3d.visualization.draw_geometries(objects_to_visualize)

#     for key in grasp_axis:
#         grasp_axis[key]['axes'] = grasp_axis[key]['axes'].T

#     for key in grasp_axis:
#         transformation_matrix = np.eye(4)
#         transformation_matrix[:3, :3] = grasp_axis[key]['axes']
#         transformation_matrix[:3, 3] = grasp_axis[key]['center']
#         grasp_axis[key] = transformation_matrix

#     return grasp_axis


def get_gt(color_image, depth_image, intrinsics_):
    """
    Generates a point cloud from a color and depth image using given intrinsics,
    overlays a movable coordinate frame, and returns the final transformation
    matrix of the grasp axis relative to the camera frame.

    Parameters:
        color_image (numpy.ndarray): Color image (H, W, 3)
        depth_image (numpy.ndarray): Depth image (H, W)
        intrinsics (o3d.camera.PinholeCameraIntrinsic): Camera intrinsics

    Returns:
        numpy.ndarray: 4x4 transformation matrix of the grasp axis relative to the camera frame
    """
    cx, cy, fx, fy = intrinsics_

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        color_image.shape[1],  # Image width
        color_image.shape[0],  # Image height
        fx, fy, cx, cy
    )

    # Step 1: Convert images to Open3D format
    color_o3d = o3d.geometry.Image(color_image)
    depth_o3d = o3d.geometry.Image(depth_image)

    # Step 2: Create RGBD image and generate point cloud
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=0.5, convert_rgb_to_intensity=False
    )
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

    # Step 3: Create a static camera frame (default)
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # Step 4: Create the movable grasp axis coordinate frame
    grasp_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # Initialize transformation matrix
    transformation = np.eye(4)
    transformation[:3, 3] = [0, 0, 0]  # Start at origin

    # Translation & Rotation step sizes
    translation_step = 0.02
    rotation_step = np.radians(5)

    def apply_transformation():
        """Reapply the updated transformation matrix to the grasp axis."""
        nonlocal grasp_axis
        grasp_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        grasp_axis.transform(transformation)

    def translate(dx, dy, dz):
        """Move the grasp axis in its local frame."""
        nonlocal transformation
        local_translation = transformation[:3, :3] @ np.array([dx, dy, dz])
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = local_translation
        transformation = translation_matrix @ transformation
        apply_transformation()

    def rotate(rx, ry, rz):
        """Rotate the grasp axis around its local frame axes."""
        nonlocal transformation
        local_x = transformation[:3, 0]  # Local X-axis
        local_y = transformation[:3, 1]  # Local Y-axis
        local_z = transformation[:3, 2]  # Local Z-axis

        def rotation_matrix(axis, theta):
            """Returns a 3x3 rotation matrix using axis-angle representation."""
            return o3d.geometry.get_rotation_matrix_from_axis_angle(theta * axis)

        if rx != 0:
            R_x = rotation_matrix(local_x, rx)
            transformation[:3, :3] = R_x @ transformation[:3, :3]
        if ry != 0:
            R_y = rotation_matrix(local_y, ry)
            transformation[:3, :3] = R_y @ transformation[:3, :3]
        if rz != 0:
            R_z = rotation_matrix(local_z, rz)
            transformation[:3, :3] = R_z @ transformation[:3, :3]

        apply_transformation()

    def key_callback(vis, key, action):
        """Handles key press events and preserves the camera view."""
        if action == 1:  # Key down event
            # Save the current view control parameters
            ctr = vis.get_view_control()
            camera_params = ctr.convert_to_pinhole_camera_parameters()

            if key == 265:  # Up arrow (Move forward in local Z)
                translate(0, 0, -translation_step)
            elif key == 264:  # Down arrow (Move backward in local Z)
                translate(0, 0, translation_step)
            elif key == 263:  # Left arrow (Move left in local X)
                translate(-translation_step, 0, 0)
            elif key == 262:  # Right arrow (Move right in local X)
                translate(translation_step, 0, 0)
            elif key == ord('Q'):  # Move up in local Y
                translate(0, translation_step, 0)
            elif key == ord('E'):  # Move down in local Y
                translate(0, -translation_step, 0)
            elif key == ord('R'):  # Roll (rotate around X)
                rotate(rotation_step, 0, 0)
            elif key == ord('P'):  # Pitch (rotate around Y)
                rotate(0, rotation_step, 0)
            elif key == ord('Y'):  # Yaw (rotate around Z)
                rotate(0, 0, rotation_step)

            # Clear and re-add geometries
            vis.clear_geometries()
            vis.add_geometry(point_cloud)
            vis.add_geometry(camera_frame)
            vis.add_geometry(grasp_axis)
            vis.update_renderer()

            # Restore the previous camera view
            ctr.convert_from_pinhole_camera_parameters(camera_params)

    # Open3D Visualization
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(point_cloud)
    vis.add_geometry(camera_frame)
    vis.add_geometry(grasp_axis)

    # Register key callbacks
    vis.register_key_callback(265, lambda vis: key_callback(vis, 265, 1))  # Up arrow
    vis.register_key_callback(264, lambda vis: key_callback(vis, 264, 1))  # Down arrow
    vis.register_key_callback(263, lambda vis: key_callback(vis, 263, 1))  # Left arrow
    vis.register_key_callback(262, lambda vis: key_callback(vis, 262, 1))  # Right arrow
    vis.register_key_callback(ord('Q'), lambda vis: key_callback(vis, ord('Q'), 1))  # Move up in Y
    vis.register_key_callback(ord('E'), lambda vis: key_callback(vis, ord('E'), 1))  # Move down in Y
    vis.register_key_callback(ord('R'), lambda vis: key_callback(vis, ord('R'), 1))  # Roll
    vis.register_key_callback(ord('P'), lambda vis: key_callback(vis, ord('P'), 1))  # Pitch
    vis.register_key_callback(ord('Y'), lambda vis: key_callback(vis, ord('Y'), 1))  # Yaw

    vis.run()  # Start visualization loop

    # Print and return final transformation
    print("\nFinal Grasp Axis Transformation (Relative to Camera Frame):")
    print(transformation)

    vis.destroy_window()  # Close visualization
    return transformation


def visualize_gripper(color_image, depth_image, intrinsics_, grasp_pose):
    
    cx, cy, fx, fy = intrinsics_
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        color_image.shape[1],  # Image width
        color_image.shape[0],  # Image height
        fx, fy, cx, cy
    )

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image),
        o3d.geometry.Image(depth_image),
        depth_scale=1.0,
        depth_trunc=0.5,  # Adjust depth truncation as needed
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

    gripper_mesh = o3d.io.read_triangle_mesh("/home/nirshal/Downloads/ImageToStl.com_2f85_opened_20190924-sep-06-2024-02-25-46-4707-pm/ImageToStl.com_2f85_opened_20190924-sep-06-2024-02-25-46-4707-pm.stl")
    
    gripper_mesh.scale(0.0001, gripper_mesh.get_center())
    rotation = grasp_pose[0]['axes']
    translation = grasp_pose[0]['center']
    gripper_mesh.rotate(rotation, center = tuple(translation))
    gripper_mesh.translate(translation)

    o3d.visualization.draw_geometries([gripper_mesh, pcd])

def visualize_rotated_axes(pcd, imp_pcd, contact_point_3d, axes, angles=[30, 60, 90], scale=0.5):
    geometries = [pcd, imp_pcd]  # Include the original point cloud and important points

    # Convert contact point to a NumPy array
    center = np.array(contact_point_3d)

    # Principal Axes
    principal_x = axes[0]  # First principal component (Rotation Axis)
    principal_y = axes[1]  # Second principal component
    principal_z = axes[2]  # Third principal component

    # Create initial coordinate frame
    axis_points = np.array([
        center, center + scale * principal_x,  # Principal Axis 1 (X)
        center, center + scale * principal_y,  # Principal Axis 2 (Y)
        center, center + scale * principal_z   # Principal Axis 3 (Z)
    ])
    lines = [[0, 1], [2, 3], [4, 5]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB: Red, Green, Blue

    axis_lines = o3d.geometry.LineSet()
    axis_lines.points = o3d.utility.Vector3dVector(axis_points)
    axis_lines.lines = o3d.utility.Vector2iVector(lines)
    axis_lines.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(axis_lines)  # Add original axes to visualization

    # Generate rotated axes
    for angle in angles:
        rot_matrix = R.from_rotvec(np.radians(angle) * principal_x).as_matrix()

        # Rotate the secondary and tertiary axes around the primary axis
        rotated_y = rot_matrix @ principal_y
        rotated_z = rot_matrix @ principal_z

        # Create a new axis visualization
        rotated_axis_points = np.array([
            center, center + scale * principal_x,  # X-Axis remains unchanged
            center, center + scale * rotated_y,    # Rotated Y-Axis
            center, center + scale * rotated_z     # Rotated Z-Axis
        ])
        rotated_axis_lines = o3d.geometry.LineSet()
        rotated_axis_lines.points = o3d.utility.Vector3dVector(rotated_axis_points)
        rotated_axis_lines.lines = o3d.utility.Vector2iVector(lines)
        rotated_axis_lines.colors = o3d.utility.Vector3dVector(colors)

        geometries.append(rotated_axis_lines)

    # Camera axis
    camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    geometries.append(camera)

    # Visualize in Open3D
    o3d.visualization.draw_geometries(geometries)

