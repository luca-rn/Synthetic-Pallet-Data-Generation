import numpy as np
import open3d as o3d

pts  = np.load("C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/SDG_output/pointcloud_0001.npy")
rgba = np.load("C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/SDG_output/pointcloud_rgb_0001.npy")
"""
noise = np.random.normal(0, 0.0005, pts.shape[0])
pts[:, 2] += noise
dropout_mask = np.random.random(pts.shape[0]) < 0.02  # 2% dropout
pts = pts[~dropout_mask]
n_outliers = int(pts.shape[0] * 0.001)  # 0.1% outliers
outlier_idx = np.random.choice(pts.shape[0], n_outliers, replace=False)
pts[outlier_idx] += np.random.normal(0, 0.05, (n_outliers, 3))
"""
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(rgba[:, :3] / 255.0)

o3d.visualization.draw_geometries([pcd], window_name="Pallet Point Cloud")
"""
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)

# Uniform colour - black on white
pcd.paint_uniform_color([0.0, 0.0, 0.0])  # black dots (1.0,1.0,1.0 = white)

o3d.visualization.draw_geometries(
    [pcd],
    window_name="Pallet Point Cloud",
)
"""