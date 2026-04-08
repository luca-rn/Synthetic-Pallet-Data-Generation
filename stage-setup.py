"""
stage_setup.py — Pallet SDG Stage Setup
Run this ONCE at the start of a session, or whenever you want to reset
the scene to a clean known state.

Usage: paste into Isaac Sim Script Editor and run.
"""

import omni.usd
from pxr import UsdGeom, UsdShade, Sdf

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

USD_PATH    = "C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/Isaac-sims/Euro-pal.usd"
PALLET_PATH = "/scene/Meshes"   # root of all pallet geometry

# ---------------------------------------------------------------------------
# 1. OPEN THE STAGE
# ---------------------------------------------------------------------------

omni.usd.get_context().open_stage(USD_PATH)
import omni.kit.app
app = omni.kit.app.get_app()
for _ in range(5):
    app.update()

stage = omni.usd.get_context().get_stage()
print("[Setup] Stage opened")

# ---------------------------------------------------------------------------
# 2. VERIFY SCALE AND POSITION  (already baked in the USD, just confirming)
# ---------------------------------------------------------------------------

scene = stage.GetPrimAtPath("/scene")
xform = UsdGeom.Xformable(scene)
ops = {op.GetOpName(): op.Get() for op in xform.GetOrderedXformOps()}
print(f"[Setup] Scale:     {ops.get('xformOp:scale')}")
print(f"[Setup] Translate: {ops.get('xformOp:translate')}")

# ---------------------------------------------------------------------------
# 3. SEMANTIC LABEL
# ---------------------------------------------------------------------------

meshes = stage.GetPrimAtPath(PALLET_PATH)
if not meshes.IsValid():
    print(f"[Setup] ERROR: prim not found at {PALLET_PATH}")
else:
    meshes.CreateAttribute(
        "semantics:params:semanticType", Sdf.ValueTypeNames.String, custom=True
    ).Set("class")
    meshes.CreateAttribute(
        "semantics:params:semanticData", Sdf.ValueTypeNames.String, custom=True
    ).Set("pallet")
    print(f"[Setup] Semantic label 'pallet' applied to {PALLET_PATH}")

# ---------------------------------------------------------------------------
# 4. SAVE
# ---------------------------------------------------------------------------

omni.usd.get_context().save_stage()
print(f"[Setup] Stage saved to {USD_PATH}")
print("[Setup] Done — ready to run replicator.py")