import glob, sys
from typing import List
import asyncio
 
import omni.replicator.core as rep

MASK_DIR    = "C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/Isaac-sims/liquid_generation/masks/"
SHADER_PATH = "/PalletStack/TopPalletNLP/Looks/LiquidDecalMat/Shader"

PALLET_PATH = "/PalletStack/TopPalletNLP"

WAREHOUSE_LIGHT = "/Root/RectLight_02" # Only randomizing the one above the pallet
#the other has little impact, need to change path if pallets moved to other side of warehouse

OUTPUT_DIR   = r"C:\Users\snook\Desktop\Uni_Stuff\NTNU\Thesis\SDG_output\water_test"

NUM_FRAMES = 10

# Camera Settings and Pose
RESOLUTION   = (1224, 1048)
FOCAL_LENGTH = 5.94
H_APERTURE   = 6.4
CAMERA_LOOKAT = (-5.0, 0.0, 1.0) 
CAMERA_POS    = (-5.0, -1.5, 1.8)
CAM_POS_MIN = (-5.1, -1.6, 1.75)
CAM_POS_MAX = (-4.9, -1.4, 1.85)

# Max and minimum intensities for light randomization
WARE_LIGHT_MIN = 3000
WARE_LIGHT_MAX = 7000
SPOTLIGHT_MIN = 500
SPOTLIGHT_MAX = 2000

# Light colour range
LIGHT_COLOUR_MIN = (0.875,0.845,0.675)
LIGHT_COLOUR_MAX = (1.0,1.0,1.0)

# Spotlight POSE
SPOT_1_POS_MIN = (-5.4, -1.5, 1.8)
SPOT_1_POS_MAX = (-4.6, -0.9, 2.5)
SPOT_1_ROT_MIN = (15, -10, 0)
SPOT_1_ROT_MAX = (30, 10, 0)

SPOT_2_POS_MIN = (-5.4, 0.9, 1.8)
SPOT_2_POS_MAX = (-4.6, 1.5, 2.5)
SPOT_2_ROT_MIN = (-30, -10, 0)
SPOT_2_ROT_MAX = (-15, 10, 0)

# Contaminant Characteristics 
MIN_DIFF_COLOUR = (0.005, 0.010, 0.030)
MAX_DIFF_COLOUR = (0.080, 0.060, 0.020)
ROUGHNESS_MIN = 0.15
ROUGHNESS_MAX = 0.4

# If using more colours for contam
COLOURFUL = True
COLOURFUL_LOW = (0.002, 0.002, 0.002)
COLOURFUL_HIGH = (0.35, 0.35, 0.35)

def establish_masks(mask_dir: str) -> List[str]:
    """Glob all liquid_mask_*.png files from mask_dir. Exits if none found."""
    mask_paths = sorted(glob.glob(mask_dir + "liquid_mask_*.png"))
    if not mask_paths:
        sys.exit(f"Error: no liquid_mask_*.png files found in {mask_dir}")
    return mask_paths

# create "spotlights" that shine on the pallet
def create_spotlights():
    spot1 = rep.create.light(
        light_type = "disk",
        position   = (-5.0, -1.2, 2.2),
        rotation   = (20, 0, 0),
        scale      = (0.2, 0.2, 0.2),
        intensity  = 1000.0,
        color      = (1.0, 1.0, 1.0),
        name       = "Spotlight1")
    
    spot2 = rep.create.light(
        light_type = "disk",
        position   = (-5.0, 1.2, 2.2),
        rotation   = (-20, 0, 0),
        scale      = (0.2, 0.2, 0.2),
        intensity  = 1000.0,
        color      = (1.0, 1.0, 1.0),
        name       = "Spotlight2")
    
    return spot1, spot2

# vary scene lighting  - all at once
def randomize_lights(warehouse_light, spot1, spot2) -> None:
    with warehouse_light:
        rep.modify.attribute("inputs:intensity", rep.distribution.uniform(WARE_LIGHT_MIN, WARE_LIGHT_MAX))
        rep.modify.attribute("inputs:color",     rep.distribution.uniform(LIGHT_COLOUR_MIN, LIGHT_COLOUR_MAX))
    with spot1:
        rep.modify.attribute("inputs:intensity", rep.distribution.uniform(SPOTLIGHT_MIN, SPOTLIGHT_MAX))
        rep.modify.attribute("inputs:color", rep.distribution.uniform(LIGHT_COLOUR_MIN, LIGHT_COLOUR_MAX))

        rep.modify.pose(position=rep.distribution.uniform(SPOT_1_POS_MIN, SPOT_1_POS_MAX),
                        rotation=rep.distribution.uniform(SPOT_1_ROT_MIN, SPOT_1_ROT_MAX))
    with spot2:
        rep.modify.attribute("inputs:intensity", rep.distribution.uniform(SPOTLIGHT_MIN, SPOTLIGHT_MIN))
        rep.modify.attribute("inputs:color", rep.distribution.uniform(LIGHT_COLOUR_MIN, LIGHT_COLOUR_MAX))
        rep.modify.pose(position=rep.distribution.uniform(SPOT_2_POS_MIN, SPOT_2_POS_MAX),
                        rotation=rep.distribution.uniform(SPOT_2_ROT_MIN, SPOT_2_ROT_MAX))

# change decal mask and vary colour & roughness
def randomize_decal(shader, mask_paths: List[str]) -> None:
    with shader:
        rep.modify.attribute("inputs:opacity_texture",
            rep.distribution.choice(mask_paths))
        if COLOURFUL:
            rep.modify.attribute("inputs:diffuse_color_constant", rep.distribution.uniform(COLOURFUL_LOW, COLOURFUL_HIGH))
        else:
            rep.modify.attribute("inputs:diffuse_color_constant",
                rep.distribution.uniform(MIN_DIFF_COLOUR, MAX_DIFF_COLOUR))
        rep.modify.attribute("inputs:reflection_roughness_constant",
            rep.distribution.uniform(ROUGHNESS_MIN, ROUGHNESS_MAX))

# Small position changes in camera        
def randomize_camera(camera) -> None:
    with camera:
        rep.modify.pose(position=rep.distribution.uniform((CAM_POS_MIN),(CAM_POS_MAX)),
                        look_at=CAMERA_LOOKAT)

with rep.new_layer():
 
    # Camera
    camera = rep.create.camera(
        focal_length      = FOCAL_LENGTH,
        horizontal_aperture = H_APERTURE,
        clipping_range    = (0.1, 100.0),
        position          = CAMERA_POS,
        look_at           = CAMERA_LOOKAT,
        name              = "TestCamera"
    )
    render_product = rep.create.render_product(camera, RESOLUTION)

    # contam mask
    mask_paths = establish_masks(MASK_DIR)
    shader = rep.get.prim_at_path(SHADER_PATH)

    # Lights
    warehouse_light = rep.get.prim_at_path(WAREHOUSE_LIGHT)
    spotlight1, spotlight2 = create_spotlights()

    # Writer
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir            = OUTPUT_DIR,
        rgb                   = True,
        bounding_box_2d_tight = False,
        semantic_segmentation = False,
    )
    writer.attach([render_product])
 
    with rep.trigger.on_frame(max_execs=NUM_FRAMES+1):
        randomize_lights(warehouse_light, spotlight1, spotlight2)
        randomize_decal(shader, mask_paths)
        randomize_camera(camera)

async def run():
    await rep.orchestrator.run_async(num_frames=NUM_FRAMES+1)
    await rep.orchestrator.wait_until_complete_async()
    print("[replicator_main] Done. Output: " + OUTPUT_DIR)
 
asyncio.ensure_future(run())