"""
stage_setup.py — Pallet SDG Stage Setup
Run this at the start of a session, or to reset the scene to a clean known state.
 
Stage units: metres (meters_per_unit = 1.0), Y-up
Pallet real-world size: 1.2 x 0.8 x 0.144m (EUR pallet)
 
GUI usage - Run in Isaac Sim Script Editor
 
Headless usage:
    ./isaac-sim.headless.bat --/omni/replicator/script="stage_setup.py"
        --usd-path "C:/path/to/scene.usd"
        --pallet-path "/scene/Meshes"
"""
 
import argparse
import sys
import glob
import omni.usd
import omni.kit.app
import omni.replicator.core as rep
from pxr import UsdGeom, UsdShade, Usd, Sdf
 
DEFAULTS = {
    "usd_path":    "C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/Isaac-sims/usd_files/warehouse_nlp.usd",
    "pallet_path": "/scene/Meshes",
    "mask_dir": "C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/Isaac-sims/liquid_generation/masks/",
}

DECAL_PATH: str = "/scene/Meshes/NLP___Oliviers_Model/LiquidDecal"
MTL_PATH: str = "/scene/Meshes/NLP___Oliviers_Model/Looks/LiquidDecalMat"
PALLET_X    = 0.5999   # half-width of pallet top surface
PALLET_Y    = 0.3999   # half-depth of pallet top surface
DECAL_Y     = 0.152    # just above pallet top

 
def parse_args() -> argparse.Namespace:
    #Parse CLI arguments, falling back to defaults if not provided
    parser = argparse.ArgumentParser(description="Pallet SDG Stage Setup")
    parser.add_argument("--usd-path",    type=str, default=DEFAULTS["usd_path"])
    parser.add_argument("--pallet-path", type=str, default=DEFAULTS["pallet_path"])
    parser.add_argument("--gen-liquid", type=bool, default=False)
    parser.add_argument("--mask-dir", type=str, default=DEFAULTS["mask_dir"])
    # parse_known_args ignores Isaac Sim's own args so no errors in GUI mode
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args
 
def open_stage(usd_path: str) -> Usd.Stage:
    #Open the USD stage and wait for it to load
    omni.usd.get_context().open_stage(usd_path)
    app: omni.kit.app.IApp = omni.kit.app.get_app()
    for _ in range(5):
        app.update()
    stage: Usd.Stage = omni.usd.get_context().get_stage()
    print(f"[Setup] Stage opened: {usd_path}")
    print(f"[Setup] meters_per_unit: {UsdGeom.GetStageMetersPerUnit(stage)}")
    print(f"[Setup] up_axis: {UsdGeom.GetStageUpAxis(stage)}")
    return stage
 
def verify_scale(stage: Usd.Stage) -> None:
    #Print the scale and translate ops on /scene to confirm they are correct
    scene: Usd.Prim = stage.GetPrimAtPath("/scene")
    if not scene.IsValid():
        print(f"[Setup] WARNING: /scene prim not found — skipping scale verification")
        return
    xform: UsdGeom.Xformable = UsdGeom.Xformable(scene)
    op: UsdGeom.XformOp
    for op in xform.GetOrderedXformOps():
        print(f"[Setup] {op.GetOpName()} = {op.Get()}")
 
def apply_semantic_label(stage: Usd.Stage, pallet_path: str) -> None:
    #Apply semantic label to the pallet prim via Replicator's SemanticsAPI
    meshes: Usd.Prim = stage.GetPrimAtPath(pallet_path)
    if not meshes.IsValid():
        print(f"[Setup] ERROR: prim not found at {pallet_path}")
        return
    rep.utils._set_semantics_legacy(meshes, [("class", "pallet")])
    print(f"[Setup] Semantic label 'pallet' applied to {pallet_path}")

def liquid_decal(stage: Usd.Stage, mask_dir: str) -> list:
    mask_paths = sorted(glob.glob(mask_dir + "liquid_mask_*.png"))
    if not mask_paths:
        mask_paths = [mask_dir + "liquid_opacity_only.png"]
    print(f"[Setup] Found {len(mask_paths)} liquid mask(s)")

    shader = UsdShade.Shader(stage.GetPrimAtPath(f"{MTL_PATH}/Shader"))
    shader.GetInput("opacity_texture").Set(Sdf.AssetPath(mask_paths[0]))
    print(f"[Setup] Decal texture set to: {mask_paths[0]}")
    return mask_paths
 
def save_stage(usd_path: str) -> None:
    #Save the current stage
    omni.usd.get_context().save_stage()
    print(f"[Setup] Stage saved to {usd_path}")
 
def main() -> None:
    args = parse_args()
 
    stage: Usd.Stage = open_stage(args.usd_path)
    verify_scale(stage)
    apply_semantic_label(stage, args.pallet_path)
    if args.gen_liquid : mask_paths = liquid_decal(stage, args.mask_dir)
    save_stage(args.usd_path)
 
    print("[Setup] Done — ready to run replicator_script_1.py")
 
if __name__ == "__main__":
    main()

