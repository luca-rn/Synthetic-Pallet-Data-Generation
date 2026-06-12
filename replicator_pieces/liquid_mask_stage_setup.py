import argparse
import sys
import glob
import omni.usd
import omni.kit.app
import omni.replicator.core as rep
from pxr import UsdGeom, UsdShade, Usd, Sdf, Gf, Vt

DEFAULTS = {
    "usd_path":    "C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/Isaac-sims/usd_files/plastic_pallet_stack.usd",
    "pallet_path": "/PalletStack/TopPalletNLP",
    "mask_dir":    "C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/Isaac-sims/liquid_generation/masks/",
}

DECAL_PATH:  str = "/PalletStack/TopPalletNLP/LiquidDecal"
SHADER_PATH = "/PalletStack/TopPalletNLP/Looks/LiquidDecalMat/Shader"
PALLET_PATH = "/PalletStack/TopPalletNLP"
PALLET_TO_DEACTIVATE: str = "/PalletStack/TopPalletEPAL"
MTL_PATH:    str = "/PalletStack/TopPalletNLP/Looks/LiquidDecalMat"

OUTPUT_DIR   = r"C:\Users\snook\Desktop\Uni_Stuff\NTNU\Thesis\SDG_output\block_test"
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
    parser.add_argument("--mask-dir",    type=str,  default=DEFAULTS["mask_dir"])
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
    rep.utils._set_semantics_legacy(prim, [("class", "pallet")])
    print(f"[Setup] Semantic label 'pallet' applied to {pallet_path}")
    
def configure_top_pallet(stage: Usd.Stage) -> None:
    """Deactivate TopPalletEpal and activate TopPalletNLP."""
    epal = stage.GetPrimAtPath(PALLET_TO_DEACTIVATE)
    if epal.IsValid():
        epal.SetActive(False)
        print(f"[Setup] Deactivated  {PALLET_TO_DEACTIVATE}")
    else:
        print(f"[Setup] {PALLET_TO_DEACTIVATE} not found — skipping")
 
    nlp = stage.GetPrimAtPath(PALLET_PATH)
    if nlp.IsValid():
        nlp.SetActive(True)
        UsdGeom.Imageable(nlp).MakeVisible()
        print(f"[Setup] Activated {PALLET_PATH}")
    else:
        print(f"[Setup] WARNING: {PALLET_PATH} not found")

def create_decal_plane(stage: Usd.Stage) -> UsdGeom.Mesh:
    """
    Flat quad mesh sitting on the TopPallet top surface.
    Authored in TopPallet local space — inherits the parent 90deg Z rotation.
    subdivisionScheme=none is critical: catmullClark collapses a 4-vert quad to a point.
    """
    cx, cy, cz = DECAL_CENTRE
    hw, hd = DECAL_HALF_W, DECAL_HALF_D
 
    mesh = UsdGeom.Mesh.Define(stage, DECAL_PATH)
 
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(cx - hw, cy - hd, cz),
        Gf.Vec3f(cx + hw, cy - hd, cz),
        Gf.Vec3f(cx + hw, cy + hd, cz),
        Gf.Vec3f(cx - hw, cy + hd, cz),
    ]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))

    # Critical: without this a 4-vert quad under catmullClark collapses to a point
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
 
    # UVs mapped 0-1 across the quad
    texCoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
    )
    texCoords.Set(Vt.Vec2fArray([
        Gf.Vec2f(0, 0), Gf.Vec2f(1, 0),
        Gf.Vec2f(1, 1), Gf.Vec2f(0, 1),
    ]))
 
    mesh.GetNormalsAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1),
        Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1),
    ]))

    # Double-sided so it renders from any camera angle
    #lowkey unnecessary i think but oh well
    mesh.GetDoubleSidedAttr().Set(True)
 
    print(f"[Setup] Decal plane created at {DECAL_PATH}")
    print(f"[Setup]   Local: {hw*2}m(X) x {hd*2}m(Y) at Z={cz}")
    print(f"[Setup]   World (after 90deg parent rot): 0.8m(X) x 1.2m(Y)")
    return mesh
 
def create_liquid_decal_material(stage: Usd.Stage, initial_mask: str) -> None:
    """
    OmniPBR material with mask-driven alpha-blend opacity.
    Bound to the decal plane only — pallet DefaultMaterial is untouched.
    """
    material = UsdShade.Material.Define(stage, MTL_PATH)
    shader   = UsdShade.Shader.Define(stage, SHADER_PATH)
    shader.SetSourceAsset("OmniPBR.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateIdAttr("OmniPBR")

    # Liquid appearance (tuned in-scene)
    shader.CreateInput("diffuse_color_constant",        Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.008, 0.015, 0.045))
    shader.CreateInput("diffuse_tint",                  Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.277, 0.245, 0.245))
    shader.CreateInput("albedo_brightness",             Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.25)
    shader.CreateInput("metallic_constant",             Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("specular_level",                Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("enable_emission",               Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateInput("emissive_color",                Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.1, 0.1))
    shader.CreateInput("emissive_intensity",            Sdf.ValueTypeNames.Float).Set(40.0)
 
    # Mask-driven alpha blend opacity (mode 0 = blend, softer edges than cutout)
    shader.CreateInput("enable_opacity",         Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("enable_opacity_texture", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("opacity_constant",       Sdf.ValueTypeNames.Float).Set(0.63)
    shader.CreateInput("opacity_mode",           Sdf.ValueTypeNames.Int).Set(0)      # 0 = alpha blend
    shader.CreateInput("opacity_threshold",      Sdf.ValueTypeNames.Float).Set(0.96)
    shader.CreateInput("opacity_texture",        Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(initial_mask))
 
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    # RTX reads mdl:surface/displacement/volume — not the generic surface output
    material.CreateOutput("mdl:surface",      Sdf.ValueTypeNames.Token).ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateOutput("mdl:displacement", Sdf.ValueTypeNames.Token).ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateOutput("mdl:volume",       Sdf.ValueTypeNames.Token).ConnectToSource(shader.ConnectableAPI(), "out")
 
    print(f"[Setup] LiquidDecalMat created at {MTL_PATH}")
    print(f"[Setup] Initial opacity texture: {initial_mask}")

def liquid_decal(stage: Usd.Stage, mask_dir: str) -> list:
    mask_paths = sorted(glob.glob(mask_dir + "liquid_mask_*.png"))
    if not mask_paths:
        sys.exit(f"[Setup] Error: no liquid_mask_*.png files found in {mask_dir}")
    print(f"[Setup] Found {len(mask_paths)} liquid mask(s)")
 
    decal_mesh = create_decal_plane(stage)
    create_liquid_decal_material(stage, initial_mask=mask_paths[0])

    # Bind to the decal plane only — NOT the pallet mesh
    material = UsdShade.Material(stage.GetPrimAtPath(MTL_PATH))
    UsdShade.MaterialBindingAPI(decal_mesh).Bind(material)
    print(f"[Setup] LiquidDecalMat bound to decal plane (pallet material unchanged)")
 
    return mask_paths

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
 
    liquid_decal(stage, args.mask_dir)
 
    # save_stage(args.usd_path) # not gonna save cause overwrites all USDs in chain, annoying
    print(f"[Setup] Done — ready to run replicator script")
    print(f"[Setup] Use SHADER_PATH = \"{SHADER_PATH}\" in replicator script")


if __name__ == "__main__":
    main()
