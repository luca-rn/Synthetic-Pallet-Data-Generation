import asyncio
import omni.replicator.core as rep
import omni.usd
import json
from pxr import Usd, UsdGeom, Gf

SCHEDULE_PATH = r"C:\Users\snook\Desktop\Uni_Stuff\NTNU\Thesis\Isaac-sims\replicator_pieces\rotation_schedule.json"

PALLET_BASE = (
    "/PalletStack/TopPalletEPAL/scene/Meshes/Sketchfab_model"
    "/_53432bb09b84172864175516b644c7a_fbx"
    "/RootNode/Pallet_Blocks/Block_"
)

OUTPUT_DIR   = r"C:\Users\snook\Desktop\Uni_Stuff\NTNU\Thesis\SDG_output\block_test"
RESOLUTION   = (1224, 1048)
FOCAL_LENGTH = 5.94
H_APERTURE   = 6.4
CAMERA_POS   = (1.4, 1.0, 0.0)
CAMERA_LOOKAT= (0.0, 0.072, 0.0)

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
 
    # Writer
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir            = OUTPUT_DIR,
        rgb                   = True,
        bounding_box_2d_tight = False,
        semantic_segmentation = False,
    )
    writer.attach([render_product])
 
    # Per-frame rotation — one rep.randomizer call per focus block
    # each driven by a sequence that advances one step per frame
    # focus blocks are those in view of camera, other blocks are skipped
    with rep.trigger.on_frame(max_execs=num_frames+1):
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
