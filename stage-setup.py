"""
stage_setup.py — Pallet Stage Setup (PalletStack)

Prepares the Isaac Sim USD stage for synthetic data generation.  Handles both
EPAL (wooden, with rotatable blocks) and NLP (plastic, with liquid-contaminant
decal) pallet types from a single entry point.

Stage units : metres (meters_per_unit = 1.0), Y-up
Pallet size : 1.2 × 0.8 × 0.144 m (EUR pallet)

Headless
--------
    ./isaac-sim.headless.bat \\
        --/omni/replicator/script="stage_setup.py" \\
        -- --pallet-type nlp --mask-dir "C:/path/to/masks/"

GUI
---
    Run directly in the Isaac Sim Script Editor (uses defaults).
"""

from __future__ import annotations

import argparse
import glob
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import omni.kit.app
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Vec3f = Tuple[float, float, float]

# ---------------------------------------------------------------------------
# PalletStack prim paths
# ---------------------------------------------------------------------------

ACCEPTED_PALLET_TYPES: List[str] = ["epal", "nlp"]

PALLET_PRIM_PATHS: Dict[str, str] = {
    "epal": "/PalletStack/TopPalletEPAL",
    "nlp":  "/PalletStack/TopPalletNLP",
}

_REPO_ROOT = Path(__file__).parent.resolve()

DEFAULT_USD_PATHS: Dict[str, str] = {
    "epal": str(_REPO_ROOT / "usd_files" / "pallet_stack.usd"),
    "nlp":  str(_REPO_ROOT / "usd_files" / "plastic_pallet_stack.usd"),
}

DEFAULT_MASK_DIR: str = str(_REPO_ROOT / "liquid_generation" / "masks")

# ---------------------------------------------------------------------------
# Liquid-decal geometry & material paths (NLP only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecalGeometry:
    """Position and size of the decal quad in pallet-local coordinates."""

    centre: Vec3f = (0.0, 0.0, 0.151)   # just above the top surface
    half_w: float = 0.6                  # X half-extent
    half_d: float = 0.4                  # Y half-extent


@dataclass(frozen=True)
class DecalPaths:
    """USD prim paths for the decal mesh, material, and shader."""

    mesh: str     = "/PalletStack/TopPalletNLP/LiquidDecal"
    material: str = "/PalletStack/TopPalletNLP/Looks/LiquidDecalMat"
    shader: str   = "/PalletStack/TopPalletNLP/Looks/LiquidDecalMat/Shader"


DECAL_GEOM  = DecalGeometry()
DECAL_PATHS = DecalPaths()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI flags; unknown flags silently ignored for Isaac Sim compat."""
    p = argparse.ArgumentParser(
        description="Pallet stage setup (PalletStack)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pallet-type", type=str, default="epal",
        choices=ACCEPTED_PALLET_TYPES,
        help="Which pallet variant to activate.",
    )
    p.add_argument(
        "--usd-path", type=str, default=None,
        help="Path to the .usd stage file (per-type default when omitted).",
    )
    p.add_argument(
        "--pallet-path", type=str, default=None,
        help="Override the USD prim path for the active pallet.",
    )
    p.add_argument(
        "--mask-dir", type=str, default=DEFAULT_MASK_DIR,
        help="Directory with liquid_mask_*.png files (NLP only).",
    )
    args, _ = p.parse_known_args(sys.argv[1:])

    # Resolve per-type defaults
    if args.usd_path is None:
        args.usd_path = DEFAULT_USD_PATHS[args.pallet_type]
    if args.pallet_path is None:
        args.pallet_path = PALLET_PRIM_PATHS[args.pallet_type]

    return args


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def open_stage(usd_path: str) -> Usd.Stage:
    """Open a USD stage, pump the app loop to let it load, and return it."""
    omni.usd.get_context().open_stage(usd_path)
    app: omni.kit.app.IApp = omni.kit.app.get_app()
    for _ in range(5):
        app.update()

    stage: Usd.Stage = omni.usd.get_context().get_stage()
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up  = UsdGeom.GetStageUpAxis(stage)
    print(f"[setup] Stage opened: {usd_path}")
    print(f"[setup]   metres_per_unit = {mpu},  up_axis = {up}")
    return stage


def verify_scale(stage: Usd.Stage) -> None:
    """Log xform ops on /scene so scale problems are immediately visible."""
    scene: Usd.Prim = stage.GetPrimAtPath("/scene")
    if not scene.IsValid():
        print("[setup] WARNING: /scene prim not found — skipping scale check")
        return
    xform = UsdGeom.Xformable(scene)
    for op in xform.GetOrderedXformOps():
        print(f"[setup]   {op.GetOpName()} = {op.Get()}")


def apply_semantic_label(stage: Usd.Stage, pallet_path: str) -> None:
    """Tag the pallet prim with the ``focus_pallet`` semantic class."""
    prim: Usd.Prim = stage.GetPrimAtPath(pallet_path)
    if not prim.IsValid():
        sys.exit(f"[setup] ERROR: prim not found at {pallet_path}")
    rep.utils._set_semantics_legacy(prim, [("class", "focus_pallet")])
    print(f"[setup] Semantic label 'focus_pallet' → {pallet_path}")


# ---------------------------------------------------------------------------
# Pallet activation / deactivation
# ---------------------------------------------------------------------------

def configure_pallets(stage: Usd.Stage, active_type: str) -> None:
    """Activate the chosen pallet variant and deactivate the other(s).

    Parameters
    ----------
    stage : Usd.Stage
        The current USD stage.
    active_type : str
        ``"epal"`` or ``"nlp"`` — the variant to keep active.
    """
    for ptype, ppath in PALLET_PRIM_PATHS.items():
        prim: Usd.Prim = stage.GetPrimAtPath(ppath)
        if not prim.IsValid():
            print(f"[setup] WARNING: {ppath} not found on stage — skipping")
            continue

        if ptype == active_type:
            prim.SetActive(True)
            UsdGeom.Imageable(prim).MakeVisible()
            print(f"[setup] Activated   {ppath}")
        else:
            prim.SetActive(False)
            print(f"[setup] Deactivated {ppath}")


# ---------------------------------------------------------------------------
# Liquid-contaminant decal (NLP only)
# ---------------------------------------------------------------------------

def _create_decal_plane(
    stage: Usd.Stage,
    geom: DecalGeometry = DECAL_GEOM,
    mesh_path: str = DECAL_PATHS.mesh,
) -> UsdGeom.Mesh:
    """Author a flat quad on the pallet top surface.

    ``subdivisionScheme = none`` is critical — catmullClark collapses a
    four-vertex quad to a point.
    """
    cx, cy, cz = geom.centre
    hw, hd = geom.half_w, geom.half_d

    mesh = UsdGeom.Mesh.Define(stage, mesh_path)

    mesh.GetPointsAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(cx - hw, cy - hd, cz),
        Gf.Vec3f(cx + hw, cy - hd, cz),
        Gf.Vec3f(cx + hw, cy + hd, cz),
        Gf.Vec3f(cx - hw, cy + hd, cz),
    ]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    # UVs spanning 0 → 1 across the quad
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying,
    )
    st.Set(Vt.Vec2fArray([
        Gf.Vec2f(0, 0), Gf.Vec2f(1, 0),
        Gf.Vec2f(1, 1), Gf.Vec2f(0, 1),
    ]))

    mesh.GetNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * 4))
    mesh.GetDoubleSidedAttr().Set(True)

    print(f"[setup] Decal plane created at {mesh_path}")
    print(f"[setup]   Local: {hw * 2:.1f} m × {hd * 2:.1f} m  at Z = {cz}")
    return mesh


def _create_decal_material(
    stage: Usd.Stage,
    initial_mask: str,
    mtl_path: str = DECAL_PATHS.material,
    shader_path: str = DECAL_PATHS.shader,
) -> None:
    """Create an OmniPBR material with mask-driven alpha-blend opacity.

    Bound to the decal plane only — the pallet DefaultMaterial is untouched.
    """
    material = UsdShade.Material.Define(stage, mtl_path)
    shader   = UsdShade.Shader.Define(stage, shader_path)

    shader.SetSourceAsset("OmniPBR.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateIdAttr("OmniPBR")

    # --- Surface appearance ---
    _inp = shader.CreateInput
    _inp("diffuse_color_constant",        Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.008, 0.015, 0.045))
    _inp("diffuse_tint",                  Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.277, 0.245, 0.245))
    _inp("albedo_brightness",             Sdf.ValueTypeNames.Float).Set(1.0)
    _inp("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.25)
    _inp("metallic_constant",             Sdf.ValueTypeNames.Float).Set(0.0)
    _inp("specular_level",                Sdf.ValueTypeNames.Float).Set(1.0)
    _inp("enable_emission",               Sdf.ValueTypeNames.Bool).Set(False)
    _inp("emissive_color",                Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.1, 0.1))
    _inp("emissive_intensity",            Sdf.ValueTypeNames.Float).Set(40.0)

    # --- Mask-driven opacity (mode 0 = alpha blend) ---
    _inp("enable_opacity",         Sdf.ValueTypeNames.Bool).Set(True)
    _inp("enable_opacity_texture", Sdf.ValueTypeNames.Bool).Set(True)
    _inp("opacity_constant",       Sdf.ValueTypeNames.Float).Set(0.63)
    _inp("opacity_mode",           Sdf.ValueTypeNames.Int).Set(0)
    _inp("opacity_threshold",      Sdf.ValueTypeNames.Float).Set(0.96)
    _inp("opacity_texture",        Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(initial_mask))

    # --- MDL outputs ---
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    connectable = shader.ConnectableAPI()
    material.CreateOutput("mdl:surface",      Sdf.ValueTypeNames.Token).ConnectToSource(connectable, "out")
    material.CreateOutput("mdl:displacement", Sdf.ValueTypeNames.Token).ConnectToSource(connectable, "out")
    material.CreateOutput("mdl:volume",       Sdf.ValueTypeNames.Token).ConnectToSource(connectable, "out")

    print(f"[setup] LiquidDecalMat created at {mtl_path}")
    print(f"[setup]   Initial mask: {initial_mask}")


def setup_liquid_decal(stage: Usd.Stage, mask_dir: str) -> List[str]:
    """Create the decal plane, material, and bind them together.

    Returns the sorted list of mask file paths found in *mask_dir*.
    """
    mask_paths: List[str] = sorted(glob.glob(
        mask_dir.rstrip("/\\") + "/liquid_mask_*.png"
    ))
    if not mask_paths:
        sys.exit(f"[setup] ERROR: no liquid_mask_*.png found in {mask_dir}")
    print(f"[setup] Found {len(mask_paths)} liquid mask(s)")

    decal_mesh = _create_decal_plane(stage)
    _create_decal_material(stage, initial_mask=mask_paths[0])

    material = UsdShade.Material(stage.GetPrimAtPath(DECAL_PATHS.material))
    UsdShade.MaterialBindingAPI(decal_mesh).Bind(material)
    print("[setup] LiquidDecalMat bound to decal plane (pallet material unchanged)")

    return mask_paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.pallet_type not in ACCEPTED_PALLET_TYPES:
        sys.exit(
            f"[setup] ERROR: unknown pallet type '{args.pallet_type}'. "
            f"Expected one of {ACCEPTED_PALLET_TYPES}."
        )

    stage: Usd.Stage = open_stage(args.usd_path)
    configure_pallets(stage, active_type=args.pallet_type)
    verify_scale(stage)
    apply_semantic_label(stage, args.pallet_path)

    # NLP: create liquid-contaminant decal geometry + material
    if args.pallet_type == "nlp":
        setup_liquid_decal(stage, args.mask_dir)

    print(f"[setup] Done — pallet type '{args.pallet_type}' is ready")
    if args.pallet_type == "nlp":
        print(f"[setup]   Shader path for replicator: \"{DECAL_PATHS.shader}\"")


if __name__ == "__main__":
    main()