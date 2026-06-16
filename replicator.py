"""
replicator.py — Pallet Synthetic Data Generation (PalletStack)

Unified replicator combining liquid-contaminant decal randomisation and
schedule-driven block rotation for the ``pallet_stack.usd`` scene.

Headless
--------
    ./isaac-sim.headless.bat \\
        --/omni/replicator/script="replicator.py" \\
        -- \\
        --pallet-type epal \\
        --num-frames 500 \\
        --output-dir "C:/my_output" \\
        --gen-liquid \\
        --rotation-schedule "rotation_schedule.json"

GUI
---
    Run in the Isaac Sim Script Editor (uses defaults).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import glob
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import carb
import omni.replicator.core as rep
import omni.usd
from pathlib import Path
from pxr import Usd

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Vec3  = Tuple[float, float, float]
Vec3i = Tuple[int, int, int]

# ---------------------------------------------------------------------------
# PalletStack prim paths  (must match stage_setup.py)
# ---------------------------------------------------------------------------

ACCEPTED_PALLET_TYPES: List[str] = ["epal", "nlp"]

PALLET_PRIM_PATHS: Dict[str, str] = {
    "epal": "/PalletStack/TopPalletEPAL",
    "nlp":  "/PalletStack/TopPalletNLP",
}

BLOCK_BASE_PATH: str = (
    "/PalletStack/TopPalletEPAL/scene/Meshes/Sketchfab_model"
    "/_53432bb09b84172864175516b644c7a_fbx"
    "/RootNode/Pallet_Blocks/Block_"
)

DECAL_SHADER_PATH: str = (
    "/PalletStack/TopPalletNLP/Looks/LiquidDecalMat/Shader"
)

# Existing warehouse light already in pallet_stack.usd
WAREHOUSE_LIGHT_PATH: str = "/Root/RectLight_02"

TOTAL_BLOCKS: int = 9

# ---------------------------------------------------------------------------
# Camera defaults (tuned for the PalletStack warehouse scene)
# ---------------------------------------------------------------------------

# The pallet sits near world (-5, 0, 1) in the warehouse
CAMERA_LOOKAT: Vec3 = (-5.0, 0.0, 1.0)

# Eight hand-tuned viewpoints covering both sides of the pallet
DEFAULT_CAMERA_POSITIONS: List[Vec3] = [
    (-5.0, -1.5, 1.8),   # side A, centre
    (-5.1, -1.4, 1.9),   # side A, left
    (-4.9, -1.4, 1.9),   # side A, right
    (-5.0, -1.6, 1.7),   # side A, low
    (-5.0,  1.5, 1.8),   # side B, centre
    (-5.1,  1.4, 1.9),   # side B, left
    (-4.9,  1.4, 1.9),   # side B, right
    (-5.0,  1.6, 1.7),   # side B, low
]

PALLET_ROTATIONS: List[Vec3i] = [
    (0, 0, 0), (0, 90, 0), (0, 180, 0), (0, 270, 0),
]

_REPO_ROOT = Path(__file__).parent.resolve()

DEFAULT_TEXTURES: List[str] = [
    str(_REPO_ROOT / "textures" / "Material_003_baseColor.jpg"),
]

DEFAULT_MASK_DIR: str = str(_REPO_ROOT / "liquid_generation" / "masks")

DEFAULT_OUTPUT_DIR: str = str(_REPO_ROOT.parent / "SDG_output")

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CameraConfig:
    """Intrinsics for the synthetic camera."""

    resolution: Tuple[int, int] = (1224, 1048)
    focal_length: float = 5.94
    h_aperture: float = 6.4
    clip_near: float = 0.1
    clip_far: float = 100.0


@dataclass
class LightConfig:
    """Intensity and pose bounds for the warehouse light and spotlights."""

    # Existing warehouse RectLight
    ware_int_min: float = 500.0
    ware_int_max: float = 8000.0

    # Spotlight (disk lights)
    spot_int_min: float = 20.0
    spot_int_max: float = 2000.0

    # Light colour range (shared by warehouse + spots)
    colour_min: Vec3 = (0.875, 0.845, 0.675)
    colour_max: Vec3 = (1.0, 1.0, 1.0)

    # Spotlight 1 — side A of the pallet
    spot1_pos_min: Vec3 = (-5.4, -1.5, 1.8)
    spot1_pos_max: Vec3 = (-4.6, -0.9, 2.5)
    spot1_rot_min: Vec3 = (15.0, -10.0, 0.0)
    spot1_rot_max: Vec3 = (30.0, 10.0, 0.0)

    # Spotlight 2 — side B of the pallet
    spot2_pos_min: Vec3 = (-5.4, 0.9, 1.8)
    spot2_pos_max: Vec3 = (-4.6, 1.5, 2.5)
    spot2_rot_min: Vec3 = (-30.0, -10.0, 0.0)
    spot2_rot_max: Vec3 = (-15.0, 10.0, 0.0)


@dataclass
class DecalConfig:
    """Liquid-contaminant decal appearance ranges."""

    mask_dir: str = DEFAULT_MASK_DIR
    shader_path: str = DECAL_SHADER_PATH

    # Standard brownish contaminant colour
    diff_colour_min: Vec3 = (0.005, 0.010, 0.030)
    diff_colour_max: Vec3 = (0.080, 0.060, 0.020)

    # Wider colour gamut when --colourful is set
    colourful: bool = True
    colourful_low: Vec3 = (0.002, 0.002, 0.002)
    colourful_high: Vec3 = (0.35, 0.35, 0.35)

    roughness_min: float = 0.15
    roughness_max: float = 0.4


@dataclass
class BlockConfig:
    """Pallet-block rotation and visibility settings (EPAL only)."""

    enabled: bool = True
    rot_max_deg: float = 90.0
    num_rotated: int = 1
    num_hidden: int = 0
    schedule_path: Optional[str] = None


@dataclass
class RenderConfig:
    """Path-tracing vs real-time render settings."""

    use_path_tracing: bool = False
    spp: int = 32
    total_spp: int = 64


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Build the argument parser and return parsed CLI flags."""
    p = argparse.ArgumentParser(
        description="Pallet SDG replicator (PalletStack)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Scene
    p.add_argument("--pallet-type", type=str, default="epal",
                    choices=ACCEPTED_PALLET_TYPES)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-frames", type=int, default=5,
                    help="Frames to generate (keep low for test runs)")

    # Lighting
    p.add_argument("--ware-int-min", type=float, default=500.0)
    p.add_argument("--ware-int-max", type=float, default=8000.0)
    p.add_argument("--spot-int-min", type=float, default=20.0)
    p.add_argument("--spot-int-max", type=float, default=2000.0)

    # Textures (EPAL only)
    p.add_argument("--textures", type=str, default=None,
                    help="Glob pattern for wood textures")

    # Liquid decal (NLP, or EPAL if explicitly enabled)
    p.add_argument("--gen-liquid", action="store_true", default=False,
                    help="Enable liquid-contaminant decal randomisation")
    p.add_argument("--mask-dir", type=str, default=DEFAULT_MASK_DIR)
    p.add_argument("--shader-path", type=str, default=DECAL_SHADER_PATH)
    p.add_argument("--colourful", action="store_true", default=True,
                    help="Use wider colour gamut for decal")

    # Block rotation (EPAL only)
    p.add_argument("--no-block-rot", action="store_true", default=False)
    p.add_argument("--block-rot-max", type=float, default=90.0)
    p.add_argument("--num-block-rot", type=int, default=1)
    p.add_argument("--num-blocks-hidden", type=int, default=0)
    p.add_argument("--rotation-schedule", type=str, default=None,
                    help="JSON rotation schedule (overrides random block rotation)")
    p.add_argument("--block-base-path", type=str, default=BLOCK_BASE_PATH)

    # Render
    p.add_argument("--path-tracing", action="store_true", default=False)
    p.add_argument("--spp", type=int, default=32)
    p.add_argument("--total-spp", type=int, default=64)

    args, _ = p.parse_known_args(sys.argv[1:])
    return args


def build_configs(
    args: argparse.Namespace,
) -> Tuple[CameraConfig, LightConfig, DecalConfig, BlockConfig, RenderConfig]:
    """Translate flat CLI namespace into typed config dataclasses."""
    cam = CameraConfig()
    light = LightConfig(
        ware_int_min=args.ware_int_min,
        ware_int_max=args.ware_int_max,
        spot_int_min=args.spot_int_min,
        spot_int_max=args.spot_int_max,
    )
    decal = DecalConfig(
        mask_dir=args.mask_dir,
        shader_path=args.shader_path,
        colourful=args.colourful,
    )
    block = BlockConfig(
        enabled=not args.no_block_rot,
        rot_max_deg=args.block_rot_max,
        num_rotated=args.num_block_rot,
        num_hidden=args.num_blocks_hidden,
        schedule_path=args.rotation_schedule,
    )
    render = RenderConfig(
        use_path_tracing=args.path_tracing,
        spp=args.spp,
        total_spp=args.total_spp,
    )
    return cam, light, decal, block, render


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def block_paths(base: str, count: int = TOTAL_BLOCKS) -> List[str]:
    """Return USD prim paths for each pallet block (EPAL)."""
    return [f"{base}{i}" for i in range(count)]


def find_textures(pattern: Optional[str]) -> List[str]:
    """Resolve texture file paths from a glob or fall back to defaults."""
    if pattern is None:
        return DEFAULT_TEXTURES
    found = sorted(glob.glob(pattern))
    if not found:
        print(f"[warn] No textures matched '{pattern}'; using defaults.")
        return DEFAULT_TEXTURES
    return found


def load_masks(mask_dir: str) -> List[str]:
    """Glob liquid_mask_*.png files; exit if none found."""
    paths = sorted(glob.glob(
        os.path.join(mask_dir.rstrip("/\\"), "liquid_mask_*.png")
    ))
    if not paths:
        sys.exit(f"[error] --gen-liquid set but no liquid_mask_*.png in {mask_dir}")
    return paths


# ---------------------------------------------------------------------------
# Rotation schedule
# ---------------------------------------------------------------------------


@dataclass
class RotationSchedule:
    """Parsed rotation schedule for deterministic block animation."""

    focus_blocks: List[int]
    num_frames: int
    angle_sequences: Dict[int, List[float]]


def load_rotation_schedule(path: str) -> RotationSchedule:
    """Read a JSON rotation schedule produced by generate_block_rotations.py.

    Expected structure::

        {
            "meta": { "focus_blocks": [0, 3, 6], ... },
            "schedule": [ { "0": 5.0, "3": 0.0, "6": -2.0 }, ... ]
        }
    """
    with open(path, "r", encoding="utf-8") as fh:
        data: dict = json.load(fh)

    meta: dict = data["meta"]
    schedule: List[dict] = data["schedule"]
    focus_blocks: List[int] = meta["focus_blocks"]

    angle_sequences: Dict[int, List[float]] = {
        idx: [entry.get(str(idx), 0.0) for entry in schedule]
        for idx in focus_blocks
    }
    return RotationSchedule(
        focus_blocks=focus_blocks,
        num_frames=len(schedule),
        angle_sequences=angle_sequences,
    )


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def create_camera(cfg: CameraConfig) -> Tuple:
    """Create the SDG camera and its render product."""
    camera = rep.create.camera(
        focal_length=cfg.focal_length,
        horizontal_aperture=cfg.h_aperture,
        clipping_range=(cfg.clip_near, cfg.clip_far),
        position=DEFAULT_CAMERA_POSITIONS[0],
        look_at=CAMERA_LOOKAT,
        name="SDGCamera",
    )
    rp = rep.create.render_product(camera, cfg.resolution)
    return camera, rp


# ---------------------------------------------------------------------------
# Lights
# ---------------------------------------------------------------------------


def create_spotlights() -> Tuple:
    """Create two disk lights aimed at opposite sides of the pallet."""
    spot1 = rep.create.light(
        light_type="disk",
        position=(-5.0, -1.2, 2.2), rotation=(20, 0, 0),
        scale=(0.2, 0.2, 0.2), intensity=1000.0,
        color=(1.0, 1.0, 1.0), name="Spotlight1",
    )
    spot2 = rep.create.light(
        light_type="disk",
        position=(-5.0, 1.2, 2.2), rotation=(-20, 0, 0),
        scale=(0.2, 0.2, 0.2), intensity=1000.0,
        color=(1.0, 1.0, 1.0), name="Spotlight2",
    )
    return spot1, spot2


# ---------------------------------------------------------------------------
# Per-frame randomisers
# ---------------------------------------------------------------------------


def randomize_camera(camera: rep.create.camera) -> None:
    """Pick a random pre-tuned position and orient toward the pallet."""
    with camera:
        rep.modify.pose(
            position=rep.distribution.choice(DEFAULT_CAMERA_POSITIONS),
            look_at=CAMERA_LOOKAT,
        )


def randomize_pallet(pallet: rep.get.prim_at_path) -> None:
    """Rotate the pallet to one of four cardinal orientations."""
    with pallet:
        rep.modify.pose(rotation=rep.distribution.choice(PALLET_ROTATIONS))


def randomize_lights(
    warehouse_light: rep.get.prim_at_path,
    spot1: rep.create.light,
    spot2: rep.create.light,
    cfg: LightConfig,
) -> None:
    """Randomise intensity, colour, and pose of all scene lights."""
    with warehouse_light:
        rep.modify.attribute(
            "inputs:intensity",
            rep.distribution.uniform(cfg.ware_int_min, cfg.ware_int_max),
        )
        rep.modify.attribute(
            "inputs:color",
            rep.distribution.uniform(cfg.colour_min, cfg.colour_max),
        )

    with spot1:
        rep.modify.attribute(
            "inputs:intensity",
            rep.distribution.uniform(cfg.spot_int_min, cfg.spot_int_max),
        )
        rep.modify.attribute(
            "inputs:color",
            rep.distribution.uniform(cfg.colour_min, cfg.colour_max),
        )
        rep.modify.pose(
            position=rep.distribution.uniform(cfg.spot1_pos_min, cfg.spot1_pos_max),
            rotation=rep.distribution.uniform(cfg.spot1_rot_min, cfg.spot1_rot_max),
        )

    with spot2:
        rep.modify.attribute(
            "inputs:intensity",
            rep.distribution.uniform(cfg.spot_int_min, cfg.spot_int_max),
        )
        rep.modify.attribute(
            "inputs:color",
            rep.distribution.uniform(cfg.colour_min, cfg.colour_max),
        )
        rep.modify.pose(
            position=rep.distribution.uniform(cfg.spot2_pos_min, cfg.spot2_pos_max),
            rotation=rep.distribution.uniform(cfg.spot2_rot_min, cfg.spot2_rot_max),
        )


def randomize_texture(textures: List[str]) -> None:
    """Swap the base-colour texture on pallet materials (EPAL)."""
    with rep.get.prims(semantics=[("class", "pallet")]):
        rep.randomizer.texture(
            textures=rep.distribution.choice(textures),
            per_sub_mesh=False,
        )


def randomize_decal(
    shader: rep.get.prim_at_path,
    mask_paths: List[str],
    cfg: DecalConfig,
) -> None:
    """Randomise contaminant mask, colour, and roughness on the decal shader."""
    with shader:
        rep.modify.attribute(
            "inputs:opacity_texture",
            rep.distribution.choice(mask_paths),
        )
        if cfg.colourful:
            rep.modify.attribute(
                "inputs:diffuse_color_constant",
                rep.distribution.uniform(cfg.colourful_low, cfg.colourful_high),
            )
        else:
            rep.modify.attribute(
                "inputs:diffuse_color_constant",
                rep.distribution.uniform(cfg.diff_colour_min, cfg.diff_colour_max),
            )
        rep.modify.attribute(
            "inputs:reflection_roughness_constant",
            rep.distribution.uniform(cfg.roughness_min, cfg.roughness_max),
        )


def randomize_blocks_random(paths: List[str], cfg: BlockConfig) -> None:
    """Randomly twist a subset of pallet blocks around the Z axis."""
    indices = random.sample(range(len(paths)), min(cfg.num_rotated, len(paths)))
    for i, bp in enumerate(paths):
        block = rep.get.prim_at_path(bp)
        with block:
            if i in indices:
                twist = random.uniform(-cfg.rot_max_deg, cfg.rot_max_deg)
                rep.modify.pose(rotation=(0.0, 0.0, twist))
            else:
                rep.modify.pose(rotation=(0.0, 0.0, 0.0))


def apply_scheduled_rotations(
    schedule: RotationSchedule,
    base_path: str,
) -> None:
    """Wire deterministic per-block rotation sequences from a JSON schedule.

    Uses ``rep.distribution.sequence`` so each frame advances one step.
    """
    for block_idx in schedule.focus_blocks:
        prim_path = f"{base_path}{block_idx}"
        block = rep.get.prim_at_path(prim_path)
        with block:
            rep.modify.pose(
                rotation=rep.distribution.sequence(
                    [(0.0, 0.0, a) for a in schedule.angle_sequences[block_idx]]
                ),
            )


def randomize_block_visibility(paths: List[str], num_hidden: int) -> None:
    """Hide a random subset of blocks to simulate missing blocks."""
    hide_indices = random.sample(range(len(paths)), min(num_hidden, len(paths)))
    for i, bp in enumerate(paths):
        block = rep.get.prim_at_path(bp)
        with block:
            rep.modify.visibility(visible=(i not in hide_indices))


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def attach_writer(render_product, output_dir: str) -> rep.WriterRegistry:
    """Initialise BasicWriter and attach to the render product."""
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=output_dir,
        rgb=True,
        bounding_box_2d_tight=False,
        semantic_segmentation=False,
        semantic_filter_predicate="class:focus_pallet",
    )
    writer.attach([render_product])
    return writer


# ---------------------------------------------------------------------------
# Render mode
# ---------------------------------------------------------------------------


def configure_render(cfg: RenderConfig) -> int:
    """Apply path-tracing or real-time settings; return sub-frame count."""
    settings = carb.settings.get_settings()
    if cfg.use_path_tracing:
        settings.set("/rtx/rendermode", "PathTracing")
        settings.set_int("/rtx/pathtracing/spp", cfg.spp)
        settings.set_int("/rtx/pathtracing/totalSpp", cfg.total_spp)
        return cfg.total_spp // cfg.spp
    settings.set("/rtx/rendermode", "RaytracedLighting")
    return 1


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------


def write_run_summary(
    args: argparse.Namespace,
    cam_cfg: CameraConfig,
    light_cfg: LightConfig,
    block_cfg: BlockConfig,
    render_cfg: RenderConfig,
    num_frames: int,
) -> None:
    """Persist a human-readable summary of the generation run."""
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.output_dir, f"run_summary_{ts}.txt")

    render_mode = carb.settings.get_settings().get("/rtx/rendermode")

    lines = [
        "=== SDG Run Summary ===",
        f"Timestamp:           {ts}",
        "",
        "--- Scene ---",
        f"Pallet type:         {args.pallet_type}",
        f"Pallet prim:         {PALLET_PRIM_PATHS[args.pallet_type]}",
        f"Liquid decal:        {args.gen_liquid}",
        f"Rotation schedule:   {args.rotation_schedule or 'none (random)'}",
        "",
        "--- Output ---",
        f"Output dir:          {args.output_dir}",
        f"Num frames:          {num_frames}",
        f"Resolution:          {cam_cfg.resolution[0]}×{cam_cfg.resolution[1]}",
        "",
        "--- Camera ---",
        f"Focal length:        {cam_cfg.focal_length} mm",
        f"Horizontal aperture: {cam_cfg.h_aperture} mm",
        f"Look-at:             {CAMERA_LOOKAT}",
        f"Positions:           {len(DEFAULT_CAMERA_POSITIONS)} viewpoints",
        "",
        "--- Lighting ---",
        f"Warehouse intensity: {light_cfg.ware_int_min} – {light_cfg.ware_int_max}",
        f"Spotlight intensity: {light_cfg.spot_int_min} – {light_cfg.spot_int_max}",
        "",
        "--- Block Rotation ---",
        f"Enabled:             {block_cfg.enabled}",
        f"Schedule:            {block_cfg.schedule_path or 'random'}",
        f"Random blocks/frame: {block_cfg.num_rotated} / {TOTAL_BLOCKS}",
        f"Max twist:           ±{block_cfg.rot_max_deg}°",
        f"Blocks hidden:       {block_cfg.num_hidden} / {TOTAL_BLOCKS}",
        "",
        "--- Render ---",
        f"Render mode:         {render_mode}",
        f"Path tracing:        {render_cfg.use_path_tracing}",
        (f"SPP / Total SPP:     {render_cfg.spp} / {render_cfg.total_spp}"
         if render_cfg.use_path_tracing else "SPP / Total SPP:     N/A (realtime)"),
    ]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[summary] Written to {path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def run_replicator(
    num_frames: int,
    args: argparse.Namespace,
    cam_cfg: CameraConfig,
    light_cfg: LightConfig,
    block_cfg: BlockConfig,
    render_cfg: RenderConfig,
) -> None:
    """Run the replicator async loop, then write a summary."""
    await rep.orchestrator.run_async(num_frames=num_frames + 1)
    await rep.orchestrator.wait_until_complete_async()
    write_run_summary(args, cam_cfg, light_cfg, block_cfg, render_cfg, num_frames)
    print(f"[replicator] Done — {num_frames} frames → {args.output_dir}")


def main() -> None:
    """Entry point: parse args, build the replicator graph, launch async."""
    args = parse_args()

    if args.pallet_type not in ACCEPTED_PALLET_TYPES:
        sys.exit(
            f"[error] Unknown pallet type '{args.pallet_type}'. "
            f"Expected one of {ACCEPTED_PALLET_TYPES}."
        )

    cam_cfg, light_cfg, decal_cfg, block_cfg, render_cfg = build_configs(args)

    # Render settings
    subframes: int = configure_render(render_cfg)

    # Rotation schedule overrides --num-frames when provided
    rot_schedule: Optional[RotationSchedule] = None
    if block_cfg.schedule_path:
        rot_schedule = load_rotation_schedule(block_cfg.schedule_path)
        num_frames = rot_schedule.num_frames
        print(
            f"[info] Rotation schedule loaded: {num_frames} frames, "
            f"blocks {rot_schedule.focus_blocks}"
        )
    else:
        num_frames = args.num_frames

    # Resolve textures (EPAL) and masks (liquid decal)
    textures: List[str] = find_textures(args.textures)
    mask_paths: Optional[List[str]] = None
    if args.gen_liquid:
        mask_paths = load_masks(decal_cfg.mask_dir)

    pallet_path: str = PALLET_PRIM_PATHS[args.pallet_type]
    bp: List[str] = block_paths(args.block_base_path)

    # ── Build Replicator graph ──────────────────────────────────────────
    with rep.new_layer():
        pallet = rep.get.prim_at_path(pallet_path)
        camera, render_product = create_camera(cam_cfg)

        # Existing warehouse light + two new spotlights
        warehouse_light = rep.get.prim_at_path(WAREHOUSE_LIGHT_PATH)
        spot1, spot2 = create_spotlights()

        # Decal shader handle (only when liquid generation is enabled)
        shader = (
            rep.get.prim_at_path(decal_cfg.shader_path)
            if args.gen_liquid else None
        )

        with rep.trigger.on_frame(max_execs=num_frames + 1, rt_subframes=subframes):
            # Camera & pallet
            randomize_camera(camera)
            randomize_pallet(pallet)

            # Lights
            randomize_lights(warehouse_light, spot1, spot2, light_cfg)

            # EPAL-specific: texture swap and block rotation
            if args.pallet_type == "epal":
                randomize_texture(textures)

                if rot_schedule is not None:
                    apply_scheduled_rotations(rot_schedule, args.block_base_path)
                elif block_cfg.enabled:
                    randomize_blocks_random(bp, block_cfg)

                if block_cfg.num_hidden > 0:
                    randomize_block_visibility(bp, block_cfg.num_hidden)

            # Liquid-contaminant decal
            if args.gen_liquid and shader is not None and mask_paths is not None:
                randomize_decal(shader, mask_paths, decal_cfg)

        attach_writer(render_product, args.output_dir)

    # Launch async
    asyncio.ensure_future(
        run_replicator(num_frames, args, cam_cfg, light_cfg, block_cfg, render_cfg)
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()