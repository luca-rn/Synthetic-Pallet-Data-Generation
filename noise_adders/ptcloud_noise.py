"""
add_noise.py — Post-processing noise for synthetic point clouds

Requires open3d to run visualisation - available on python 3.8 - 3.11

Usage:
    uv run add_noise.py --pointcloud pointcloud_0000.npy
                        --rgba pointcloud_rgb_0000.npy
                        --depth distance_to_camera_0000.npy
                        --output pointcloud_noisy_0000.npy
                        --visualise

Noise types and where they are applied:
    1. Gaussian noise   — pts only    (positional uncertainty)
    2. Edge noise       — pts (flying pixels) + rgba (colour bleeding at boundaries)
    3. Dropout          — pts + rgba  (missing point has no colour)
    4. Outliers         — pts only    (colour sensor sees valid colour, depth is wrong)
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.ndimage import sobel, binary_dilation


# ---------------------------------------------------------------------------
# CONFIG — Noise Parameters
# ---------------------------------------------------------------------------

GAUSSIAN_SIGMA: float   = 0.00055   # metres (~0.55mm std, M70 point precision is 0.055mm but is not effective enough)

EDGE_THRESHOLD: float   = 0.05      # normalised gradient to classify as edge
EDGE_DILATION: int      = 1         # pixels to dilate edge mask
EDGE_NOISE_SIGMA: float = 0.0001      # metres — flying pixel displacement (0.1mm std)
EDGE_COLOR_SIGMA: float = 5.0      # uint8 — colour bleeding std at edges

DROPOUT_RATE: float     = 0.005     # fraction of points to remove (0.5%)

OUTLIER_RATE: float     = 0.001     # fraction of points to corrupt (0.1%)
OUTLIER_SIGMA: float    = 0.001      # metres — outlier displacement std (1mm)


# ---------------------------------------------------------------------------

def load_pointcloud(path: Path) -> np.ndarray:
    #Load XYZ point cloud
    pts: np.ndarray = np.load(path).astype(np.float32)
    assert pts.shape[1] == 3, f"Expected (N, 3), got {pts.shape}"
    print(f"[Noise] Loaded {pts.shape[0]:,} points from {path.name}")
    return pts

def load_rgba(path: Path) -> np.ndarray:
    rgba: np.ndarray = np.load(path)
    assert rgba.shape[1] == 4, f"Expected (N, 4), got {rgba.shape}"
    print(f"[Noise] Loaded {rgba.shape[0]:,} RGBA values from {path.name}")
    return rgba

def build_edge_mask(depth: np.ndarray, threshold: float, dilation: int) -> np.ndarray:
    """
    Build a boolean edge mask from a depth map using Sobel gradient magnitude.
    Args:
        depth:     (H, W)
        threshold: Normalised gradient threshold to classify as edge
        dilation:  Pixels to dilate the binary edge mask.
    Returns:
        (H*W,) flattened boolean mask — True at edge pixels.
    """
    dx: np.ndarray = sobel(depth, axis=1)
    dy: np.ndarray = sobel(depth, axis=0)
    magnitude: np.ndarray = np.hypot(dx, dy)
    max_val: float = float(magnitude.max())
    if max_val > 0:
        magnitude /= max_val
    binary: np.ndarray = magnitude > threshold
    if dilation > 0:
        binary = binary_dilation(binary, iterations=dilation)
    n_edge: int = int(binary.sum())
    print(f"[Noise] Edge pixels: {n_edge:,} ({100 * n_edge / binary.size:.2f}% of image)")
    return binary.flatten()


def add_gaussian_noise(
    pts: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Add per-point Gaussian noise to XYZ positions only
    sigma: Std in metres
    """
    noise: np.ndarray = np.random.normal(0, sigma, pts.shape).astype(np.float32)
    print(f"[Noise] Gaussian noise applied (sigma={sigma*1000:.2f}mm)")
    return pts + noise


def add_edge_noise(
    pts: np.ndarray,
    rgba: np.ndarray,
    edge_mask: np.ndarray,
    pos_sigma: float,
    color_sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply structured light edge artefacts:
      - Flying pixels: large positional displacement at depth discontinuities
      - Colour bleeding: colour corruption at boundaries (adjacent surface bleeds in)
    Args:
        edge_mask:   (N,) boolean mask of edge pixels
        pos_sigma:   Positional noise std in metres for edge points
        color_sigma: Colour noise std (uint8) for edge pixels
    """
    pts  = pts.copy()
    rgba = rgba.copy()
    n_edge: int = int(edge_mask.sum())

    if n_edge == 0:
        print("[Noise] No edge points found — skipping edge noise")
        return pts, rgba

    # Flying pixels — strong positional noise at edges
    pts[edge_mask] += np.random.normal(0, pos_sigma, (n_edge, 3)).astype(np.float32)
    print(f"[Noise] Edge position noise: {n_edge:,} points (sigma={pos_sigma*1000:.1f}mm)")

    # Colour bleeding — mild colour corruption at edges (RGB only, preserve alpha)
    color_noise: np.ndarray = np.random.normal(0, color_sigma, (n_edge, 3))
    rgba_float: np.ndarray = rgba[edge_mask, :3].astype(np.float32) + color_noise
    rgba[edge_mask, :3] = np.clip(rgba_float, 0, 255).astype(np.uint8)
    print(f"[Noise] Edge colour bleeding: {n_edge:,} pixels (sigma={color_sigma:.1f})")

    return pts, rgba


def add_dropout(
    pts: np.ndarray,
    rgba: np.ndarray,
    rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly remove points and their corresponding colours.
    Simulates structured light failure on dark/reflective surfaces.
    """
    n_before: int = pts.shape[0]
    keep_mask: np.ndarray = np.random.random(n_before) > rate
    pts  = pts[keep_mask]
    rgba = rgba[keep_mask]
    n_removed: int = n_before - pts.shape[0]
    print(f"[Noise] Dropout removed {n_removed:,} points ({rate*100:.1f}%)")
    return pts, rgba


def add_outliers(
    pts: np.ndarray,
    rate: float,
    sigma: float,
) -> np.ndarray:
    # sigma: Std of outlier offset in metres
    n_outliers: int = int(pts.shape[0] * rate)
    idx: np.ndarray = np.random.choice(pts.shape[0], n_outliers, replace=False)
    pts[idx] += np.random.normal(0, sigma, (n_outliers, 3)).astype(np.float32)
    print(f"[Noise] Added {n_outliers:,} outliers (sigma={sigma*1000:.0f}mm)")
    return pts


def visualise(
    pts_clean: np.ndarray,
    pts_noisy: np.ndarray,
    rgba_clean: np.ndarray,
    rgba_noisy: np.ndarray,
) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("[Noise] open3d not installed — skipping visualisation")
        return

    def make_pcd(pts: np.ndarray, rgba: np.ndarray) -> "o3d.geometry.PointCloud":
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(rgba[:, :3] / 255.0)
        return pcd

    pcd_clean = make_pcd(pts_clean, rgba_clean)
    pcd_noisy = make_pcd(pts_noisy, rgba_noisy)
    pcd_noisy.translate([2.5, 0, 0])  # offset for side-by-side comparison

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Clean (left) vs Noisy (right)", width=1600, height=800)
    vis.add_geometry(pcd_clean)
    vis.add_geometry(pcd_noisy)
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    opt.point_size = 1.0
    vis.run()
    vis.destroy_window()


# ---------------------------------------------------------------------------
# MAIN

def parse_args() -> argparse.Namespace:
    #Parse CLI arguments, falling back to defaults
    parser = argparse.ArgumentParser(description="Add structured light noise to point cloud")
    parser.add_argument("--pointcloud",     type=Path,  required=True)
    parser.add_argument("--rgba",           type=Path,  required=True)
    parser.add_argument("--depth",          type=Path,  required=True)
    parser.add_argument("--output",         type=Path,  required=True)
    parser.add_argument("--gaussian-sigma", type=float, default=GAUSSIAN_SIGMA)
    parser.add_argument("--edge-threshold", type=float, default=EDGE_THRESHOLD)
    parser.add_argument("--edge-dilation",  type=int,   default=EDGE_DILATION)
    parser.add_argument("--edge-sigma",     type=float, default=EDGE_NOISE_SIGMA)
    parser.add_argument("--edge-color-sigma", type=float, default=EDGE_COLOR_SIGMA)
    parser.add_argument("--dropout-rate",   type=float, default=DROPOUT_RATE)
    parser.add_argument("--outlier-rate",   type=float, default=OUTLIER_RATE)
    parser.add_argument("--outlier-sigma",  type=float, default=OUTLIER_SIGMA)
    parser.add_argument("--visualise",      action="store_true")
    parser.add_argument("--seed",           type=int,   default=42)
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    # Load inputs
    pts_clean: np.ndarray  = load_pointcloud(args.pointcloud)
    rgba_clean: np.ndarray = load_rgba(args.rgba)
    depth: np.ndarray      = np.load(args.depth).astype(np.float32)

    assert pts_clean.shape[0] == rgba_clean.shape[0], \
        f"pts and rgba must have same N: {pts_clean.shape[0]} vs {rgba_clean.shape[0]}"

    # Build edge mask from depth map
    edge_mask: np.ndarray = build_edge_mask(depth, args.edge_threshold, args.edge_dilation)

    # Apply noise pipeline
    pts: np.ndarray  = pts_clean.copy()
    rgba: np.ndarray = rgba_clean.copy()

    pts                = add_gaussian_noise(pts, sigma=args.gaussian_sigma)
    pts, rgba          = add_edge_noise(pts, rgba, edge_mask,
                                        pos_sigma=args.edge_sigma,
                                        color_sigma=args.edge_color_sigma)
    pts, rgba          = add_dropout(pts, rgba, rate=args.dropout_rate)
    pts                = add_outliers(pts, rate=args.outlier_rate, sigma=args.outlier_sigma)

    # Save outputs
    rgba_output: Path = args.output.with_stem(args.output.stem + "_rgb")
    np.save(args.output, pts)
    np.save(rgba_output, rgba)
    print(f"[Noise] Saved noisy pts  -> {args.output}")
    print(f"[Noise] Saved noisy rgba -> {rgba_output}")
    print(f"[Noise] Points: {pts_clean.shape[0]:,} -> {pts.shape[0]:,}")

    if args.visualise:
        visualise(pts_clean, pts, rgba_clean, rgba)


if __name__ == "__main__":
    main()