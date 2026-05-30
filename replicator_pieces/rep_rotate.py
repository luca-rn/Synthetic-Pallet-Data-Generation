# Loads a pre-generated rotation schedule (from rotated_blocks.py)
# and applies it to pallet blocks each Replicator frame.

import json
import omni.usd
import omni.replicator.core as rep
import omni.kit.app
import carb.events
from pxr import UsdGeom, Gf, Usd

# Set this before exec()-ing, or edit it directly here
SCHEDULE_PATH = r"C:\Users\snook\Desktop\Uni_Stuff\NTNU\Thesis\Isaac-sims\replicator_pieces\rotation_schedule.json"

# Specific to the right USD (blocks_pallet.usd)
PALLET_BASE = (
    "/scene/Meshes/Sketchfab_model"
    "/_53432bb09b84172864175516b644c7a_fbx"
    "/RootNode/Pallet_Blocks/Block_"
)

def load_schedule(path):
    """Read the JSON schedule file and return (meta, schedule, focus_blocks)."""
    with open(path, "r") as f:
        data = json.load(f)
    meta         = data["meta"]
    schedule     = data["schedule"]
    focus_blocks = meta["focus_blocks"]
    return meta, schedule, focus_blocks

def get_or_add_rot_op(prim):
    #Return the existing rotateZYX xform op, or add one if missing
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateZYX:
            return op
    return xform.AddRotateZYXOp()

def set_block_rotation(stage, block_idx, angle_degrees):
    #Write a Z-axis rotation to a single block prim
    path = PALLET_BASE + str(block_idx)
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        print("[pallet_rotate] WARNING: prim not found: " + path)
        return
    
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        rot_op = get_or_add_rot_op(prim)
        rot_op.Set(Gf.Vec3f(float(angle_degrees), 0.0, 0.0))  # ZYX: (z, y, x)

def reset_blocks(stage, block_indices):
    #Set a list of blocks back to 0 deg rotation
    for idx in block_indices:
        set_block_rotation(stage, idx, 0.0)
 
def make_frame_callback(schedule, focus_blocks):
    """
    Returns the per-frame callback closure.
    Captures schedule and focus_blocks so the callback is self-contained.
    """
    frame_index = [0]
 
    def on_frame(e=None):
        idx = frame_index[0]
        if idx >= len(schedule):
            print("[pallet_rotate] Schedule exhausted at frame " + str(idx))
            return
        
        stage  = omni.usd.get_context().get_stage()
        entry  = schedule[idx]
 
        # Apply rotations to whichever blocks are listed in this frame's entry
        for block_idx_str, angle in entry.items():
            set_block_rotation(stage, int(block_idx_str), angle)
 
        # Reset in-focus blocks that aren't rotating this frame
        rotated_this_frame = set(int(k) for k in entry.keys())
        idle_blocks = [i for i in focus_blocks if i not in rotated_this_frame]
        reset_blocks(stage, idle_blocks)
 
        print("[pallet_rotate] Frame {} applied. ({}/{} focus blocks rotated)".format(
            idx, len(entry), len(focus_blocks)))
 
        frame_index[0] = idx + 1
 
    return on_frame

def register_with_replicator(num_frames, callback):
    #Register event subscription
    subscription = omni.kit.app.get_app().get_message_bus_event_stream()\
        .create_subscription_to_pop_by_type(
            carb.events.type_from_string("pallet_rotate"),
            callback
        )
 
    return subscription # caller must hold a reference to keep subscription alive

def log_summary(meta, focus_blocks):
    """Print a summary of the loaded schedule."""
    all_blocks   = list(range(9))
    out_of_focus = [i for i in all_blocks if i not in focus_blocks]
    print("[pallet_rotate] Schedule loaded from: " + SCHEDULE_PATH)
    print("  Mode         : " + meta["mode"])
    print("  Frames       : " + str(meta["frames"]))
    print("  Focus blocks : " + str(focus_blocks))
    print("  Out of focus : " + str(out_of_focus) + "  (reset to 0 deg at startup)")

def setup():
    meta, schedule, focus_blocks = load_schedule(SCHEDULE_PATH)
 
    log_summary(meta, focus_blocks)
 
    # Reset out-of-focus blocks once at startup so no stale rotations carry over
    all_blocks   = list(range(9))
    out_of_focus = [i for i in all_blocks if i not in focus_blocks]
    stage        = omni.usd.get_context().get_stage()
    reset_blocks(stage, out_of_focus)
 
    # Build the per-frame callback and register it with Replicator
    callback = make_frame_callback(schedule, focus_blocks)
 
    # Store subscription in a global so it isn't garbage collected mid-run
    global _pallet_sub
    _pallet_sub = register_with_replicator(len(schedule), callback)
 
    #print("\n[pallet_rotate] Registered. Run rep.orchestrator.run() to start.")

    return len(schedule)

