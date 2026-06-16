import asyncio
import json
from pathlib import Path
import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom, Gf

_REPO_ROOT    = Path(__file__).parent.parent.resolve()
SCHEDULE_PATH = str(Path(__file__).parent / "rotation_schedule.json")

PALLET_BASE = (
    "/PalletStack/TopPalletEPAL/scene/Meshes/Sketchfab_model"
    "/_53432bb09b84172864175516b644c7a_fbx"
    "/RootNode/Pallet_Blocks/Block_"
)

WAREHOUSE_LIGHT = "/Root/RectLight_02" # Only randomizing the one above the pallet
#the other has little impact, need to change path if pallets moved to other side of warehouse

OUTPUT_DIR   = str(_REPO_ROOT.parent / "SDG_output" / "block_test")
RESOLUTION   = (1224, 1048)
FOCAL_LENGTH = 5.94
H_APERTURE   = 6.4
CAMERA_LOOKAT = (-5.0, 0.0, 1.0) 
CAMERA_POS    = (-5.0, -1.5, 1.8)

# For randomizing camera on both sides of the pallet
CAMERA_POSITIONS = [
    (-5.0, -1.5, 1.8),   # side a, centre
    (-5.1, -1.4, 1.9),   # side a, left
    (-4.9, -1.4, 1.9),   # side a, right
    (-5.0, -1.6, 1.7),   # side a, low
    (-5.0,  1.5, 1.8),   # side b, centre
    (-5.1, 1.4, 1.9),   # side b, left
    (-4.9, 1.4, 1.9),   # side b, right
    (-5.0,  1.6, 1.7),   # side b, low
]

# Max and minimum intensities for light randomization
WARE_LIGHT_MIN = 500
WARE_LIGHT_MAX = 8000
SPOTLIGHT_MIN = 20
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

def load_schedule(path):
    """Read the JSON schedule file and return (meta, schedule, focus_blocks)."""
    with open(path, "r") as f:
        data = json.load(f)
    meta         = data["meta"]
    schedule     = data["schedule"]
    focus_blocks = meta["focus_blocks"]
    return meta, schedule, focus_blocks

def extract_angle_sequences(focus_blocks, schedule):
    angle_sequences = {}
    for block_idx in focus_blocks:
        angle_sequences[block_idx] = [entry.get(str(block_idx), 0.0) for entry in schedule]
    return angle_sequences

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
        
# Small position changes in camera        
def randomize_camera(camera) -> None:
    with camera:
        rep.modify.pose(
            position=rep.distribution.choice(CAMERA_POSITIONS),
            look_at=CAMERA_LOOKAT
        )


meta, schedule, focus_blocks = load_schedule(SCHEDULE_PATH)
angle_sequences = extract_angle_sequences(focus_blocks, schedule)
num_frames = len(schedule)
stage = omni.usd.get_context().get_stage()
 
with rep.new_layer():
 
    # Camera
    camera = rep.create.camera(
        focal_length      = FOCAL_LENGTH,
        horizontal_aperture = H_APERTURE,
        clipping_range    = (0.1, 100.0),
        position          = CAMERA_POS,
        look_at           = CAMERA_LOOKAT,
        name              = "BlockTestCamera"
    ) 
    render_product = rep.create.render_product(camera, RESOLUTION)
 
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
        semantic_filter_predicate="class:focus_pallet"
    )
    writer.attach([render_product])
 
    # Per-frame rotation — one rep.randomizer call per focus block
    # each driven by a sequence that advances one step per frame
    # focus blocks are those in view of camera, other blocks are skipped
    with rep.trigger.on_frame(max_execs=num_frames+1):
        randomize_lights(warehouse_light, spotlight1, spotlight2)
        randomize_camera(camera)
        for block_idx in focus_blocks:
            # prim path to each block in isaac sim
            prim_path = PALLET_BASE + str(block_idx)
            block_prim = rep.get.prim_at_path(prim_path)
            with block_prim:
                # modify block rotation (around centerpoints), no translation applied
                rep.modify.pose(
                    rotation=rep.distribution.sequence(
                        # sequence of (x, y, z) rotation tuples — only Z varies
                        # sequence extracted from rotation_schedule
                        [(0.0, 0.0, a) for a in angle_sequences[block_idx]]
                    )
                )

async def run():
    await rep.orchestrator.run_async(num_frames=num_frames+1)
    await rep.orchestrator.wait_until_complete_async()
    print("[replicator_main] Done. Output: " + OUTPUT_DIR)
 
asyncio.ensure_future(run())
