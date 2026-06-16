from pathlib import Path
import numpy as np
import open3d as o3d

_PTCLOUD_DIR = Path(__file__).parent.parent / "ptcloud_examples"
pts  = np.load(_PTCLOUD_DIR / "pointcloud_noisy_0001.npy")
rgba = np.load(_PTCLOUD_DIR / "pointcloud_noisy_0001_rgb.npy")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(rgba[:, :3] / 255.0)

o3d.visualization.draw_geometries([pcd], window_name="Pallet Point Cloud")