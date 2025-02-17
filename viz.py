import open3d as o3d
import numpy as np
import cv2
from sklearn.decomposition import PCA 
from scipy.spatial.transform import Rotation as R

def pca_2d(pixels, center, image):
    pca = PCA(n_components=2)
    pca.fit(pixels)
    
    direction = pca.components_[0]
    length = 30
    p1 = (int(center[0] - length * direction[0]), int(center[1] - length * direction[1]))
    p2 = (int(center[0] + length * direction[0]), int(center[1] + length * direction[1]))

    pca_image = image.copy()
    pca_image = cv2.cvtColor(pca_image, cv2.COLOR_RGB2BGR)
    cv2.line(pca_image, p1, p2, (0, 255, 0), 2)
    cv2.circle(pca_image, center, 3, (0, 0, 255), -1)

    cv2.imshow("PCA Line Overlay", pca_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pca_3d(points, intrinsics_, depth_image, color_image, contact_point, inference_directional_point):
    
    cx, cy, fx, fy = intrinsics_

    intrinsics = o3d.camera.PinholeCameraIntrinsic(color_image.shape[1], color_image.shape[0], fx, fy, cx, cy)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                            o3d.geometry.Image(color_image), 
                            o3d.geometry.Image(depth_image), 
                            depth_scale=1.0, 
                            depth_trunc=3, 
                            convert_rgb_to_intensity=False
                            )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics) # whole point cloud

    # Extract the 3D points corresponding to the important pixels
    v_coords = points[:, 0]
    u_coords = points[:, 1]

    z_values = depth_image[v_coords, u_coords]
    x_values = (u_coords - cx) * z_values / fx
    y_values = (v_coords - cy) * z_values / fy

    points_3d = np.column_stack([x_values, y_values, z_values])
    imp_pcd = o3d.geometry.PointCloud()
    imp_pcd.points = o3d.utility.Vector3dVector(points_3d)
    imp_pcd.paint_uniform_color([1, 0, 0])

    # Extract the 3D point corresponding to the contact point
    contact_point_3d = [(contact_point[1] - cx) * depth_image[int(contact_point[0]), int(contact_point[1])] / fx,
                        (contact_point[0] - cy) * depth_image[int(contact_point[0]), int(contact_point[1])] / fy,
                        depth_image[int(contact_point[0]), int(contact_point[1])]]
    
    directional_point_3d = np.array([
        (inference_directional_point[1] - cx) * depth_image[int(inference_directional_point[0]), int(inference_directional_point[1])] / fx,
        (inference_directional_point[0] - cy) * depth_image[int(inference_directional_point[0]), int(inference_directional_point[1])] / fy,
        depth_image[int(inference_directional_point[0]), int(inference_directional_point[1])]
    ])

    # Compute PCA
    pca = PCA(n_components=3)
    pca.fit(points_3d)

    center = contact_point_3d
    axes = pca.components_

    dir_vector = directional_point_3d - contact_point_3d

    if np.dot(dir_vector, axes[0]) < 0:
        axes[0] = -axes[0]

    axes[1] = np.cross(axes[2], axes[0])
    axes[1] /= np.linalg.norm(axes[1])  # Normalize

    axes[2] = np.cross(axes[0], axes[1])  # Recompute the third axis


    scale = 0.5
    axis_points = np.array([
        center, center + scale * axes[0],  # Principal axis 1
        center, center + scale * axes[1],  # Principal axis 2
        center, center + scale * axes[2]   # Principal axis 3
    ])

    # Create a line set
    lines = [[0, 1], [2, 3], [4, 5]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB: Red, Green, Blue
    axis_lines = o3d.geometry.LineSet()
    axis_lines.points = o3d.utility.Vector3dVector(axis_points)
    axis_lines.lines = o3d.utility.Vector2iVector(lines)
    axis_lines.colors = o3d.utility.Vector3dVector(colors)


    # camera axis
    camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([pcd, imp_pcd, axis_lines, camera])

    # o3d.visualization.draw_geometries([pcd, imp_pcd, axis_lines])
    return pcd, imp_pcd, contact_point_3d, axes


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

