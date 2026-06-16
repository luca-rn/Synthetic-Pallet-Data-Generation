# Sets up the Replicator render pipeline and runs it

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(Path(__file__).parent))

import omni.replicator.core as rep
import rep_rotate
import asyncio

OUTPUT_DIR = str(_REPO_ROOT.parent / "SDG_output" / "block_test")

RESOLUTION      = (1224, 1048)
FOCAL_LENGTH    = 5.94
H_APERTURE      = 6.4
CAMERA_POSITION = (1.4, 1.0, 0.0)
CAMERA_LOOK_AT  = (0.0, 0.072, 0.0)

num_frames = rep_rotate.setup()

async def run(num_frames):
    await rep.orchestrator.run_async(num_frames=num_frames+1)
    await rep.orchestrator.wait_until_complete_async()
    print("Done.")

with rep.new_layer():
    camera = rep.create.camera(
        focal_length=FOCAL_LENGTH,
        horizontal_aperture=H_APERTURE,
        clipping_range=(0.1, 100.0),
        position=CAMERA_POSITION,
        look_at=CAMERA_LOOK_AT,
        name="BlockTestCamera"
    )
    render_product = rep.create.render_product(camera, RESOLUTION)

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir            = OUTPUT_DIR,
        rgb                   = True,
        bounding_box_2d_tight = False,
        semantic_segmentation = False,
    )
    writer.attach([render_product])

    with rep.trigger.on_frame(num_frames=num_frames+1):
        rep.utils.send_og_event(event_name="pallet_rotate")

asyncio.ensure_future(run(num_frames))
