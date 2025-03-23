import cv2
import numpy as np
import open3d as o3d
from camera_calibration import get_transformation_from_base_to_wrist_camera

LIGHTNING_GRABIT_HOME_EEF_POSE = np.load('resources/robot/robot_home_pose.npy', allow_pickle=True).item()['lightning']['eef_pose']

def make_coordinate_frame(transformation_matrix : np.ndarray, 
                          size=0.1):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0]).transform(transformation_matrix)

def main():
    object = 'pouch'
    T_base2wrist_cam, T_base2eef = get_transformation_from_base_to_wrist_camera(LIGHTNING_GRABIT_HOME_EEF_POSE)
    cam_coordinate_frame = make_coordinate_frame(T_base2wrist_cam)
    eef_coordinate_frame = make_coordinate_frame(T_base2eef)

    ## Load the PC
    fused_pcd = o3d.io.read_point_cloud(f"resources/{object}/fused_pointcloud.ply")
    ## transformation between base and back camera
    T_base2backcam = np.load(f"resources/camera_calibration/T_base2backcam.npy")
    fused_pcd = fused_pcd.transform(T_base2backcam)

    ## Back Cam
    backcam_coordinate_frame = make_coordinate_frame(T_base2backcam)

    ## Load the predicted grasp
    lightning_gripper_pose = np.load(f"resources/{object}/gripper_pose_wrt_base.npy")

    ## Load Gripper Mesh
    gripper_mesh = o3d.io.read_triangle_mesh(f"resources/gripper-mesh/ImageToStl.com_2f85_opened_20190924.ply")
    gripper_mesh.compute_vertex_normals()

    ## Color with a gradient
    colors = np.zeros((len(gripper_mesh.vertices), 3))
    for i in range(len(gripper_mesh.vertices)):
        color_value = 0.7 * (i / len(gripper_mesh.vertices))
        colors[i] = [color_value, color_value, color_value]
    gripper_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    gripper_mesh.scale(0.001, center=gripper_mesh.get_center())
    gripper_mesh.translate(gripper_mesh.get_center() * -1)

    gripper_mesh_rotation = o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi/2, 0, 0])
    gripper_mest_ratation_matrix = np.eye(4)
    gripper_mest_ratation_matrix[:3, :3] = gripper_mesh_rotation
    gripper_mesh.transform(gripper_mest_ratation_matrix)

    gripper_mesh.transform(lightning_gripper_pose)

    ## Make gripper coordinate frame
    gripper_coordinate_frame = make_coordinate_frame(lightning_gripper_pose)

    ## Filter the point cloud for points that are less than 0.3m z
    # fused_pcd = fused_pcd.select_by_index(np.where(np.array(fused_pcd.points)[:, 2] > 0.5)[0])
    # fused_pcd = fused_pcd.select_by_index(np.where(np.array(fused_pcd.points)[:, 2] < 1)[0])

    base_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    base_coordinate_frame.transform(T_base2wrist_cam)

    o3d.visualization.draw_geometries([
                                  base_coordinate_frame, 
                                  cam_coordinate_frame,
                                  gripper_mesh,
                                  backcam_coordinate_frame,
                                  gripper_coordinate_frame,
                                  fused_pcd,
                                  cam_coordinate_frame, 
                                  eef_coordinate_frame
                                  ],
                                  window_name="Coordinate Frames",
                                  )

if __name__ == "__main__":
    main()