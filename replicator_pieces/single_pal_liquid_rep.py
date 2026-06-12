import glob, sys
from typing import List
import asyncio
 
import omni.replicator.core as rep

MASK_DIR    = "C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/Isaac-sims/liquid_generation/masks/"
SHADER_PATH = "/scene/Meshes/NLP___Oliviers_Model/Looks/LiquidDecalMat/Shader"

PALLET_PATH = "/scene/Meshes" # Path to the pallet within Isaac Sim - Set up by stage-setup.py

OUTPUT_DIR   = r"C:\Users\snook\Desktop\Uni_Stuff\NTNU\Thesis\SDG_output\block_test"
RESOLUTION   = (1224, 1048)
FOCAL_LENGTH = 5.94
H_APERTURE   = 6.4
CAMERA_POS   = (1.0, 1.0, 0.0) # need to adjust
CAMERA_LOOKAT= (0.0, 0.075, 0.0)

NUM_FRAMES = 1

def establish_masks(mask_dir: str) -> List[str]:
    """Glob all liquid_mask_*.png files from mask_dir. Exits if none found."""
    mask_paths = sorted(glob.glob(mask_dir + "liquid_mask_*.png"))
    if not mask_paths:
        sys.exit(f"Error: no liquid_mask_*.png files found in {mask_dir}")
    return mask_paths

def randomize_decal(shader, mask_paths: List[str]) -> None:
    """Pick a random mask texture and assign it to the decal shader's opacity input."""
    with shader:
        rep.modify.attribute(
            "inputs:opacity_texture",
            rep.distribution.choice(mask_paths)
        )

mask_paths = establish_masks(MASK_DIR)
shader = rep.get.prim_at_path(SHADER_PATH)
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
        randomize_decal(shader, mask_paths)

async def run():
    await rep.orchestrator.run_async(num_frames=NUM_FRAMES+1)
    await rep.orchestrator.wait_until_complete_async()
    print("[replicator_main] Done. Output: " + OUTPUT_DIR)
 
asyncio.ensure_future(run())
