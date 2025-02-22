import open3d as o3d
import numpy as np

# Create a static coordinate frame at the origin (reference frame)
static_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

# Create a moving coordinate frame
moving_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

# Initial transformation for the moving frame
transformation = np.eye(4)
transformation[:3, 3] = [1, 1, 0]  # Start at (1,1,0)

# Step sizes
translation_step = 0.1
rotation_step = np.radians(10)  # Convert degrees to radians

def apply_transformation():
    """Reapply the updated transformation matrix to the moving frame."""
    global moving_frame, transformation
    moving_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    moving_frame.transform(transformation)

def translate(dx, dy, dz):
    """Translates the moving frame along its local axes."""
    global transformation
    local_translation = transformation[:3, :3] @ np.array([dx, dy, dz])  # Convert to local coordinates
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = local_translation  # Move in local frame
    transformation = translation_matrix @ transformation  # Apply transformation
    apply_transformation()

def rotate(rx, ry, rz):
    """Rotates the moving frame around its own axes."""
    global transformation
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
    """Handles key press events."""
    if action == 1:  # Key down event
        if key == 265:  # Up arrow
            translate(0, 0, -translation_step)
        elif key == 264:  # Down arrow
            translate(0, 0, translation_step)
        elif key == 263:  # Left arrow
            translate(-translation_step, 0, 0)
        elif key == 262:  # Right arrow
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

        vis.clear_geometries()
        vis.add_geometry(static_frame)
        vis.add_geometry(moving_frame)
        vis.update_renderer()

def main():
    global transformation

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(static_frame)
    vis.add_geometry(moving_frame)

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

    # Print final transformation matrix before closing
    print("\nFinal Transformation Matrix (Relative to Default Frame):")
    print(transformation)

    vis.destroy_window()  # Close window

if __name__ == "__main__":
    main()
