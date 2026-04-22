import numpy as np
import open3d as o3d

pts  = np.load("C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/Isaac-sims/ptcloud_examples/pointcloud_noisy_0000.npy")
rgba = np.load("C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/Isaac-sims/ptcloud_examples/pointcloud_noisy_0000_rgb.npy")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(rgba[:, :3] / 255.0)

o3d.visualization.draw_geometries([pcd], window_name="Pallet Point Cloud")