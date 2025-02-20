import open3d as o3d
import numpy as np
import cv2
from sklearn.decomposition import PCA 
from sklearn.linear_model import RANSACRegressor
from scipy.spatial.transform import Rotation as R

def pca_2d(pixels, center, image):
    pca_image = image.copy()
    pca_image = cv2.cvtColor(pca_image, cv2.COLOR_RGB2BGR)  
    for key in pixels:
        pca = PCA(n_components=2)
        pca.fit(pixels[key])
        direction = pca.components_[0]
        length = 30
        p1 = (int(center[key][0] - length * direction[0]), int(center[key][1] - length * direction[1]))
        p2 = (int(center[key][0] + length * direction[0]), int(center[key][1] + length * direction[1]))
        cv2.line(pca_image, p1, p2, (0, 255, 0), 2)
        cv2.circle(pca_image, center[key], 3, (0, 0, 255), -1)
    

    cv2.imshow("PCA Line Overlay", pca_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # pca_image = image.copy()
    # pca_image = cv2.cvtColor(pca_image, cv2.COLOR_RGB2BGR)  

    # for key in pixels:
    #     points = np.array(pixels[key])

    #     # Fit PCA to find the main direction
    #     pca = PCA(n_components=2)

    #     # Use RANSAC to remove outliers
    #     x_coords = points[:, 0].reshape(-1, 1)
    #     y_coords = points[:, 1]

    #     ransac = RANSACRegressor()
    #     ransac.fit(x_coords, y_coords)
    #     inlier_mask = ransac.inlier_mask_  # Identified inliers

    #     # Recompute PCA on inliers only
    #     inlier_points = points[inlier_mask]
    #     pca.fit(inlier_points)
    #     direction = pca.components_[0]

    #     # Extend line in both directions
    #     length = 30
    #     p1 = (int(center[key][0] - length * direction[0]), int(center[key][1] - length * direction[1]))
    #     p2 = (int(center[key][0] + length * direction[0]), int(center[key][1] + length * direction[1]))

    #     # Draw the robust PCA line
    #     cv2.line(pca_image, p1, p2, (0, 255, 0), 2)
    #     cv2.circle(pca_image, center[key], 3, (0, 0, 255), -1)

    # cv2.imshow("Robust PCA Line Overlay", pca_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# 
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
        depth_trunc=3.0,  # Adjust depth truncation as needed
        convert_rgb_to_intensity=False
    )

    # Create point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

    # Initialize dictionaries to store 3D points and axes
    points_3d = {}
    contact_point_3d = {}
    directional_point_3d = {}
    grasp_axis = {}

    # Convert 2D points to 3D and filter invalid points
    for key in points:
        v_coords = points[key][:, 0]  # v-coordinates (rows)
        u_coords = points[key][:, 1]  # u-coordinates (columns)

        # Initialize lists to store valid 3D points
        valid_x = []
        valid_y = []
        valid_z = []

        # Convert each 2D point to 3D
        for v, u in zip(v_coords, u_coords):
            z = depth_image[int(v), int(u)]  # Depth value at (v, u)

            # Check if depth value is valid
            if np.isfinite(z) and z > 0 and z < 3:  # Adjust depth truncation as needed
                x = (u - cx) * z / fx  # X coordinate
                y = (v - cy) * z / fy  # Y coordinate
                valid_x.append(x)
                valid_y.append(y)
                valid_z.append(z)

        # Store valid 3D points
        points_3d[key] = np.column_stack([valid_x, valid_y, valid_z])

        # Convert contact point to 3D
        z_contact = depth_image[int(contact_point[key][0]), int(contact_point[key][1])]
        if np.isfinite(z_contact) and z_contact > 0 and z_contact < 3:
            contact_point_3d[key] = [
                (contact_point[key][1] - cx) * z_contact / fx,
                (contact_point[key][0] - cy) * z_contact / fy,
                z_contact
            ]
        else:
            contact_point_3d[key] = None  # Mark as invalid

        # Convert directional point to 3D
        z_dir = depth_image[int(inference_directional_point[key][0]), int(inference_directional_point[key][1])]
        if np.isfinite(z_dir) and z_dir > 0 and z_dir < 3:
            directional_point_3d[key] = np.array([
                (inference_directional_point[key][1] - cx) * z_dir / fx,
                (inference_directional_point[key][0] - cy) * z_dir / fy,
                z_dir
            ])
        else:
            directional_point_3d[key] = None  # Mark as invalid

    # Compute PCA for each set of 3D points
    for key in points_3d:
        if len(points_3d[key]) == 0:
            print(f"No valid 3D points for key {key}. Skipping PCA.")
            continue

        # Perform PCA
        pca = PCA(n_components=3)
        pca.fit(points_3d[key])

        # Store PCA results
        grasp_axis[key] = {
            'center': contact_point_3d[key],
            'axes': pca.components_
        }

        # Adjust the first principal axis to align with the inferred direction
        if directional_point_3d[key] is not None:
            dir_vector = directional_point_3d[key] - contact_point_3d[key]
            if np.dot(dir_vector, grasp_axis[key]['axes'][0]) < 0:
                grasp_axis[key]['axes'][0] = -grasp_axis[key]['axes'][0]

        # Ensure the axes are orthogonal
        grasp_axis[key]['axes'][1] = np.cross(grasp_axis[key]['axes'][2], grasp_axis[key]['axes'][0])
        grasp_axis[key]['axes'][1] /= np.linalg.norm(grasp_axis[key]['axes'][1])
        grasp_axis[key]['axes'][2] = np.cross(grasp_axis[key]['axes'][0], grasp_axis[key]['axes'][1])

    # Visualize the results
    axis_lines = {}
    for key in grasp_axis:
        if grasp_axis[key]['center'] is None:
            continue

        center = np.array(grasp_axis[key]['center'])
        axes = grasp_axis[key]['axes']

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
        axis_lines[key] = o3d.geometry.LineSet()
        axis_lines[key].points = o3d.utility.Vector3dVector(axis_points)
        axis_lines[key].lines = o3d.utility.Vector2iVector(lines)
        axis_lines[key].colors = o3d.utility.Vector3dVector(colors)

    # Create a camera coordinate frame for visualization
    camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    # Combine all objects for visualization
    objects_to_visualize = [pcd, camera]
    for key in axis_lines:
        objects_to_visualize.append(axis_lines[key])

    # Visualize the point cloud and PCA axes
    o3d.visualization.draw_geometries(objects_to_visualize)

    return grasp_axis

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

