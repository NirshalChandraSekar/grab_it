import open3d as o3d
# draw a sphere
mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
mesh_sphere.compute_vertex_normals()
mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
# vizualize the sphere
vis = o3d.visualization.Visualizer()
vis.create_window()
# vis.add_geometry(mesh_sphere)
vis.draw_plotly(mesh_sphere)
#
# # draw a cube
vis.run()
vis.destroy_window()