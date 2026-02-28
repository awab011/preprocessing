import open3d as o3d

# Path to your .pcd file
pcd_path = "gamefield.pcd"

# Load point cloud
pcd = o3d.io.read_point_cloud(pcd_path)

# Check if loaded properly
if not pcd.has_points():
    print("Error: Point cloud is empty or file not found.")
    exit()

print("Point cloud loaded successfully!")
print(pcd)

# Visualize
o3d.visualization.draw_geometries([pcd],
                                  window_name="PCD Viewer",
                                  width=800,
                                  height=600)

# import open3d as o3d
# import numpy as np

# # Load mesh
# mesh = o3d.io.read_triangle_mesh("/home/utmrbc/Downloads/Red gamefield with all rack (1).STL")

# # Sample points from mesh surface
# pcd = mesh.sample_points_uniformly(number_of_points=100000)

# # Save as PCD
# o3d.io.write_point_cloud("gamefield.pcd", pcd)

# # Visualize
# o3d.visualization.draw_geometries([pcd])