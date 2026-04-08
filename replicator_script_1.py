"""
Pallet SDG
Run after stage-setup.py
 
Stage units: metres (meters_per_unit = 1.0), Y-up
Pallet size: 1.2 x 0.8 x 0.144m, centred at X=0, Z=0, sitting on Y=0
 
run in Isaac Sim Script editor.
"""

import math
import random
from typing import List, Tuple
import omni.replicator.core as rep
from omni.replicator.core import Writer
from omni.replicator.core.scripts.utils.viewport_manager import HydraTexture


print("Running replicator script...")

# Config

PALLET_PATH: str                              = "/scene/Meshes"
OUTPUT_DIR: str                               = "C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/SDG_output"
NUM_FRAMES: int                               = 50
# Camera intrinsics - need to match to real camera later
RESOLUTION: Tuple[int, int]      = (2448, 2048)    # actual M70 resolution
FOCAL_LENGTH: float    = 5.94            # mm, derived from FOV
H_APERTURE: float     = 6.4             # mm, 1/2" sensor

# Spherical camera sampling around pallet centre (0, 0.072, 0)
PALLET_CENTRE: Tuple[float, float, float]     = (0.0, 0.072, 0.0)

CAM_DIST_MIN: float    = 1.3             # metres — closest the camera gets
CAM_DIST_MAX: float    = 2.3             # metres — furthest the camera gets
CAM_ELEV_MIN: float                           = 15.0      # degrees above horizon — avoids ground-level shots
CAM_ELEV_MAX: float                           = 75.0      # degrees — avoids pure top-down shots

# Light randomization
KEY_INT_MIN: float                            = 1000.0
KEY_INT_MAX: float                            = 7000.0
FILL_INT_MIN: float                           = 300.0
FILL_INT_MAX: float                           = 800.0
DOME_INT_MIN: float                           = 300.0
DOME_INT_MAX: float                           = 800.0
# Pallet rotation
PALLET_ROTATIONS: List[Tuple[int, int, int]]  = [(0,0,0), (0,90,0), (0,180,0), (0,270,0)]

# Spherical Camera Placement

def sample_camera_positions(
    n: int,
    centre: Tuple[float, float, float],
    dist_min: float,
    dist_max: float,
    elev_min_deg: float,
    elev_max_deg: float,
) -> List[Tuple[float, float, float]]:
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
 
 
# Pre-sample all camera positions for the run
camera_positions: List[Tuple[float, float, float]] = sample_camera_positions(
    n=NUM_FRAMES,
    centre=PALLET_CENTRE,
    dist_min=CAM_DIST_MIN,
    dist_max=CAM_DIST_MAX,
    elev_min_deg=CAM_ELEV_MIN,
    elev_max_deg=CAM_ELEV_MAX,
)

# Replicator

with rep.new_layer():

    # Define scene objects
    pallet = rep.get.prim_at_path(PALLET_PATH)

    # Define camera
    camera = rep.create.camera(focal_length=FOCAL_LENGTH, horizontal_aperture=H_APERTURE,
    clipping_range=(0.1, 100.0), name="SDGCamera") # in m
    
    render_product: HydraTexture = rep.create.render_product(camera, RESOLUTION)
    
    # Create lights
    key_light = rep.create.light(light_type="Distant", intensity=600, color=(1.0, 0.97, 0.9), rotation=(225, 0, 0), name="KeyLight" )
    fill_light = rep.create.light(light_type="Sphere",  intensity=400,  color=(0.8, 0.85, 1.0), position=(-3.0, 2.0, 1.0),   name="FillLight") # pos in m
    dome_light = rep.create.light(light_type="Dome",    intensity=300, name="DomeLight")

    # Dfine randomizations - So far: Pose and Light
    with rep.trigger.on_frame(max_execs=NUM_FRAMES):

        # Camera: spherical sampling, always aimed at pallet
        with camera:
            rep.modify.pose(
                position=rep.distribution.choice(camera_positions),
                look_at=PALLET_PATH
            )

        with pallet:
            rep.modify.pose(
                rotation=rep.distribution.choice(PALLET_ROTATIONS)
            )

        with key_light:
            rep.modify.attribute("inputs:intensity", rep.distribution.uniform(KEY_INT_MIN, KEY_INT_MAX))
            rep.modify.attribute("inputs:color",     rep.distribution.uniform((0.85,0.75,0.6), (1.0,1.0,1.0)))
            rep.modify.pose(rotation=rep.distribution.uniform((200,-30,0), (260,30,0)))

        with fill_light:
            rep.modify.pose(position=rep.distribution.uniform((-4.0, 1.0, -3.0), (4.0, 4.0, 3.0)))
            rep.modify.attribute("inputs:intensity", rep.distribution.uniform(FILL_INT_MIN, FILL_INT_MAX))

        with dome_light:
            rep.modify.attribute("inputs:intensity", rep.distribution.uniform(DOME_INT_MIN, DOME_INT_MAX))


    # 4. Define what to capture
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=OUTPUT_DIR,
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
    
rep.orchestrator.run()
print(f"[Replicator] Done. {NUM_FRAMES} frames written to {OUTPUT_DIR}")

