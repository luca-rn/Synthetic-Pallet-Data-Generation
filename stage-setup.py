"""
stage_setup.py — Pallet SDG Stage Setup
Run this at the start of a session, or to reset the scene to a clean known state.

Stage units: metres (meters_per_unit = 1.0), Y-up
Pallet real-world size: 1.2 x 0.8 x 0.144m (EUR pallet)

run in Isaac Sim Script Editor.
"""

import omni.usd
import omni.kit.app
from pxr import UsdGeom, Sdf, Usd

# config

USD_PATH: str    = "C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/Isaac-sims/Euro-pal-floor.usd"
PALLET_PATH: str = "/scene/Meshes" # root of all geometry

# open stage

omni.usd.get_context().open_stage(USD_PATH)

app: omni.kit.app.IApp = omni.kit.app.get_app()
for _ in range(5):
    app.update()
 
stage: Usd.Stage = omni.usd.get_context().get_stage()
print("[Setup] Stage opened")
print(f"[Setup] meters_per_unit: {UsdGeom.GetStageMetersPerUnit(stage)}")
print(f"[Setup] up_axis: {UsdGeom.GetStageUpAxis(stage)}")

# verifying scale and position  (already in the USD, just confirming)

scene: Usd.Prim = stage.GetPrimAtPath("/scene")
xform: UsdGeom.Xformable = UsdGeom.Xformable(scene)
op: UsdGeom.XformOp
for op in xform.GetOrderedXformOps():
    print(f"[Setup] {op.GetOpName()} = {op.Get()}")

# Attaching semantic label

meshes: Usd.Prim = stage.GetPrimAtPath(PALLET_PATH)
if not meshes.IsValid():
    print(f"[Setup] ERROR: prim not found at {PALLET_PATH}")
else:
    rep.utils._set_semantics_legacy(meshes, [("class", "pallet")])
    print(f"[Setup] Semantic label 'pallet' applied to {PALLET_PATH}")

# SAVE

omni.usd.get_context().save_stage()
print(f"[Setup] Stage saved to {USD_PATH}")
print("[Setup] Done — ready to run replicator.py")
