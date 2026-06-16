import argparse
import sys
import glob
from pathlib import Path
import omni.usd
import omni.kit.app
import omni.replicator.core as rep
from pxr import UsdGeom, UsdShade, Usd, Sdf, Gf, Vt

_REPO_ROOT = Path(__file__).parent.parent.resolve()

DEFAULTS = {
    "usd_path":    str(_REPO_ROOT / "usd_files" / "pallet_stack.usd"),
    "pallet_path": "/PalletStack/TopPalletEPAL"
}

PALLET_PATH = "/PalletStack/TopPalletEPAL"
PALLET_TO_DEACTIVATE: str = "/PalletStack/TopPalletNLP"

BLOCKS_PATH = (
    "/PalletStack/TopPalletEPAL/scene/Meshes/Sketchfab_model"
    "/_53432bb09b84172864175516b644c7a_fbx"
    "/RootNode/Pallet_Blocks/Block_"
)

OUTPUT_DIR   = str(_REPO_ROOT.parent / "SDG_output" / "block_test")
RESOLUTION   = (1224, 1048)
FOCAL_LENGTH = 5.94
H_APERTURE   = 6.4
CAMERA_LOOKAT = (-5.0, 0.0, 1.0) 
CAMERA_POS    = (-5.0, -1.5, 1.8)

DECAL_CENTRE:  tuple = (0.0, 0.0, 0.151)  # local coords, 0cm above top surface
DECAL_HALF_W:  float = 0.6
DECAL_HALF_D:  float = 0.4  

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pallet SDG Stage Setup")
    parser.add_argument("--usd-path",    type=str,  default=DEFAULTS["usd_path"])
    parser.add_argument("--pallet-path", type=str,  default=DEFAULTS["pallet_path"])
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args

def open_stage(usd_path: str) -> Usd.Stage:
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
    scene: Usd.Prim = stage.GetPrimAtPath("/scene")
    if not scene.IsValid():
        print(f"[Setup] WARNING: /scene prim not found — skipping scale verification")
        return
    xform: UsdGeom.Xformable = UsdGeom.Xformable(scene)
    for op in xform.GetOrderedXformOps():
        print(f"[Setup] {op.GetOpName()} = {op.Get()}")

def apply_semantic_label(stage: Usd.Stage, pallet_path: str) -> None:
    prim: Usd.Prim = stage.GetPrimAtPath(pallet_path)
    if not prim.IsValid():
        print(f"[Setup] ERROR: prim not found at {pallet_path}")
        return
    rep.utils._set_semantics_legacy(prim, [("class", "focus_pallet")])
    print(f"[Setup] Semantic label 'focus_pallet' applied to {pallet_path}")
    
def configure_top_pallet(stage: Usd.Stage) -> None:
    """Deactivate and active"""
    nlp = stage.GetPrimAtPath(PALLET_TO_DEACTIVATE)
    if nlp.IsValid():
        nlp.SetActive(False)
        print(f"[Setup] Deactivated  {PALLET_TO_DEACTIVATE}")
    else:
        print(f"[Setup] {PALLET_TO_DEACTIVATE} not found — skipping")
 
    epal = stage.GetPrimAtPath(PALLET_PATH)
    if epal.IsValid():
        epal.SetActive(True)
        UsdGeom.Imageable(epal).MakeVisible()
        print(f"[Setup] Activated {PALLET_PATH}")
    else:
        print(f"[Setup] WARNING: {PALLET_PATH} not found")

"""
def save_stage(usd_path: str) -> None:
    omni.usd.get_context().save_stage()
    print(f"[Setup] Stage saved to {usd_path}")
"""

def main() -> None:
    args = parse_args()
 
    stage: Usd.Stage = open_stage(args.usd_path)
    configure_top_pallet(stage)
    verify_scale(stage)
    apply_semantic_label(stage, args.pallet_path)
 
    # save_stage(args.usd_path) # not gonna save cause overwrites all USDs in chain, annoying
    print(f"[Setup] Done — ready to run replicator script")


if __name__ == "__main__":
    main()
