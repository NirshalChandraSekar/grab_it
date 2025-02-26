import open3d as o3d


if __name__ == "__main__":
    
    # import gripper stl
    gripper_mesh = o3d.io.read_triangle_mesh("/home/nirshal/Downloads/ImageToStl.com_2f85_opened_20190924-sep-06-2024-02-25-46-4707-pm/ImageToStl.com_2f85_opened_20190924-sep-06-2024-02-25-46-4707-pm.stl")
    global_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0, origin=[0, 0, 0])
    gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0)  # Adjust size if needed
    gripper_frame.translate(gripper_mesh.get_center())

    o3d.visualization.draw_geometries([gripper_mesh, global_frame, gripper_frame])