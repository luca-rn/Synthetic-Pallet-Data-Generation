"""
replicator_script_1.py — Pallet Synthetic Data Generation
Run after stage_setup.py

GUI usage - run in Isaac Sim Script Editor.
 
Headless usage:
    ./isaac-sim.headless.bat --/omni/replicator/script="replicator_script_1.py"
        --output-dir "C:/my_output"
        --num-frames 500
        --cam-dist-min 1.3
        --cam-dist-max 2.3
"""
import argparse
import math
import random
import sys
from typing import List, Tuple

import omni.replicator.core as rep
from omni.replicator.core import Writer
from omni.replicator.core.scripts.utils.viewport_manager import HydraTexture


print("Running replicator script...")

# Config

DEFAULTS = {
    "pallet_path":  "/scene/Meshes",
    "output_dir":   "C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/SDG_output",
    "num_frames":   5, # Set low to avoid accidental large runs
    # How close to pallet
    "cam_dist_min": 1.3,
    "cam_dist_max": 2.3,
    # degrees above base plane
    "cam_elev_min": 15.0,
    "cam_elev_max": 75.0,
    # Light randomization limits
    "key_int_min":  300.0,
    "key_int_max":  7000.0,
    "fill_int_min": 300.0,
    "fill_int_max": 800.0,
    "dome_int_min": 300.0,
    "dome_int_max": 800.0,
}

# Camera intrinsics - need to match to real camera later
RESOLUTION: Tuple[int, int]      = (2448, 2048)    # Zivid 2 M70 resolution
FOCAL_LENGTH: float    = 5.94            # mm, derived from FOV
H_APERTURE: float     = 6.4             # mm, 1/2" sensor

PALLET_CENTRE: Tuple[float, float, float]     = (0.0, 0.072, 0.0)
PALLET_ROTATIONS: List[Tuple[int, int, int]]  = [(0,0,0), (0,90,0), (0,180,0), (0,270,0)]

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments, falling back to defaults if not provided."""
    parser = argparse.ArgumentParser(description="Pallet SDG Replicator")
    parser.add_argument("--pallet-path",  type=str,   default=DEFAULTS["pallet_path"])
    parser.add_argument("--output-dir",   type=str,   default=DEFAULTS["output_dir"])
    parser.add_argument("--num-frames",   type=int,   default=DEFAULTS["num_frames"])
    parser.add_argument("--cam-dist-min", type=float, default=DEFAULTS["cam_dist_min"])
    parser.add_argument("--cam-dist-max", type=float, default=DEFAULTS["cam_dist_max"])
    parser.add_argument("--cam-elev-min", type=float, default=DEFAULTS["cam_elev_min"])
    parser.add_argument("--cam-elev-max", type=float, default=DEFAULTS["cam_elev_max"])
    parser.add_argument("--key-int-min",  type=float, default=DEFAULTS["key_int_min"])
    parser.add_argument("--key-int-max",  type=float, default=DEFAULTS["key_int_max"])
    parser.add_argument("--fill-int-min", type=float, default=DEFAULTS["fill_int_min"])
    parser.add_argument("--fill-int-max", type=float, default=DEFAULTS["fill_int_max"])
    parser.add_argument("--dome-int-min", type=float, default=DEFAULTS["dome_int_min"])
    parser.add_argument("--dome-int-max", type=float, default=DEFAULTS["dome_int_max"])
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args

# Spherical Camera Placement
def sample_camera_positions(
    n: int,
    centre: Tuple[float, float, float],
    dist_min: float,
    dist_max: float,
    elev_min_deg: float,
    elev_max_deg: float) -> List[Tuple[float, float, float]]:
    """
    Sample n camera positions on a sphere around centre.
    elevation is clamped to avoid ground-level or pure top-down shots.
 
    Args:
        n:            Number of positions to sample.
        centre:       World-space target point (pallet centre).
        dist_min:     Minimum distance from centre in metres.
        dist_max:     Maximum distance from centre in metres.
        elev_min_deg: Minimum elevation angle in degrees (above horizon).
        elev_max_deg: Maximum elevation angle in degrees.
 
    Returns:
        List of (x, y, z) camera positions in world space.
    """
    positions: List[Tuple[float, float, float]] = []
    for _ in range(n):
        distance: float  = random.uniform(dist_min, dist_max)
        azimuth: float   = random.uniform(0.0, 360.0)          # degrees, full circle
        elevation: float = random.uniform(elev_min_deg, elev_max_deg)  # degrees
 
        az_rad: float    = math.radians(azimuth)
        el_rad: float    = math.radians(elevation)
 
        x: float = centre[0] + distance * math.cos(el_rad) * math.sin(az_rad)
        y: float = centre[1] + distance * math.sin(el_rad)
        z: float = centre[2] + distance * math.cos(el_rad) * math.cos(az_rad)
 
        positions.append((x, y, z))
 
    return positions
 
def create_camera() -> HydraTexture:
    # Create the SDG camera and render product matching given intrinsics
    camera = rep.create.camera(
        focal_length=FOCAL_LENGTH,
        horizontal_aperture=H_APERTURE,
        clipping_range=(0.1, 100.0),
        name="Camera"
    )
    render_product: HydraTexture = rep.create.render_product(camera, RESOLUTION)
    return camera, render_product

def create_lights() -> Tuple:
    #Create lights with baseline intensities
    key_light = rep.create.light(
        light_type="Distant", intensity=600,
        color=(1.0, 0.97, 0.9), rotation=(225, 0, 0), name="KeyLight")
    
    fill_light = rep.create.light(
        light_type="Sphere", intensity=400,
        color=(0.8, 0.85, 1.0), position=(-3.0, 2.0, 1.0), name="FillLight")
    
    dome_light = rep.create.light(light_type="Dome", intensity=300, name="DomeLight")
    return key_light, fill_light, dome_light

def randomize_camera(
        camera,camera_positions: List[Tuple[float, float, float]], pallet_path: str) -> None:
    #Randomize camera position each frame, always looking at the pallet
    with camera:
        rep.modify.pose(position=rep.distribution.choice(camera_positions),look_at=pallet_path)

def randomize_pallet(pallet) -> None:
    #Rotate pallet each frame - maybe unnecessary given changing camera positions
    with pallet:
        rep.modify.pose(rotation=rep.distribution.choice(PALLET_ROTATIONS))

def randomize_lights(
    key_light, fill_light, dome_light,
    key_int_min: float, key_int_max: float,
    fill_int_min: float, fill_int_max: float,
    dome_int_min: float, dome_int_max: float) -> None:
    #Randomize intensity, colour, and position of all lights each frame
    with key_light:
        rep.modify.attribute("inputs:intensity", rep.distribution.uniform(key_int_min, key_int_max))
        rep.modify.attribute("inputs:color",     rep.distribution.uniform((0.85,0.75,0.6), (1.0,1.0,1.0)))
        rep.modify.pose(rotation=rep.distribution.uniform((225,-15,0), (245,15,0)))
 
    with fill_light:
        rep.modify.pose(position=rep.distribution.uniform((-4.0,1.0,-3.0), (4.0,4.0,3.0)))
        rep.modify.attribute("inputs:intensity", rep.distribution.uniform(fill_int_min, fill_int_max))
 
    with dome_light:
        rep.modify.attribute("inputs:intensity", rep.distribution.uniform(dome_int_min, dome_int_max))
 
def attach_writer(render_product: HydraTexture, output_dir: str) -> Writer:
    #Initialise BasicWriter with all required annotators and attach to render product
    writer: Writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=output_dir,
        rgb=True,
        bounding_box_2d_tight=True,
        bounding_box_2d_loose=True,
        bounding_box_3d=True,
        instance_segmentation=True,
        semantic_segmentation=True,
        distance_to_camera=True,
        normals=True,
        camera_params=True,
    )
    writer.attach([render_product])
    return writer

def main() -> None:
    args = parse_args()
 
    camera_positions: List[Tuple[float, float, float]] = sample_camera_positions(
        n=args.num_frames,
        centre=PALLET_CENTRE,
        dist_min=args.cam_dist_min,
        dist_max=args.cam_dist_max,
        elev_min_deg=args.cam_elev_min,
        elev_max_deg=args.cam_elev_max,
    )

    with rep.new_layer():
 
        pallet = rep.get.prim_at_path(args.pallet_path)
        camera, render_product = create_camera()
        key_light, fill_light, dome_light = create_lights()
 
        with rep.trigger.on_frame(max_execs=args.num_frames):
            randomize_camera(camera, camera_positions, args.pallet_path)
            randomize_pallet(pallet)
            randomize_lights(
                key_light, fill_light, dome_light,
                args.key_int_min, args.key_int_max,
                args.fill_int_min, args.fill_int_max,
                args.dome_int_min, args.dome_int_max,
            )

        attach_writer(render_product, args.output_dir)
 
    rep.orchestrator.run()
    print(f"[Replicator] Done. {args.num_frames} frames written to {args.output_dir}")

if __name__ == "__main__":
    main()