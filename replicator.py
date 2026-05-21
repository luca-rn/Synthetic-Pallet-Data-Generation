"""
replicator_script_1.py — Pallet Synthetic Data Generation
Run after stage_setup.py

GUI usage - run in Isaac Sim Script Editor.

Headless usage:
    ./isaac-sim.headless.bat --/omni/replicator/script="replicator.py"
        --output-dir "C:/my_output"
        --num-frames 500
        --cam-dist-min 1.3
        --cam-dist-max 2.3
        --textures "/path/to/textures/*"
"""
import argparse, math, random, sys, os
import asyncio, carb, glob
from typing import List, Tuple
import datetime

import omni.hydra.engine.stats as hstats
import omni.kit.app
import omni.replicator.core as rep
from omni.replicator.core import Writer
from omni.replicator.core.scripts.utils.viewport_manager import HydraTexture


print("Running replicator script...")

# Config

DEFAULTS = {
    "pallet_path":  "/scene/Meshes", # Path to the pallet within Isaac Sim - Set up by stage-setup.py
    "output_dir":   "C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/SDG_output",
    # Liquid Decals
    "mask_dir" : "C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/Isaac-sims/liquid_generation/masks/",
    "shader_path" : "/scene/Meshes/NLP___Oliviers_Model/Looks/LiquidDecalMat/Shader",

    "num_frames":   5, # Set low to avoid accidental large runs
    # How close to pallet
    "cam_dist_min": 1.3,
    "cam_dist_max": 2.3,
    # degrees above base plane
    "cam_elev_min": 15.0,
    "cam_elev_max": 35.0,
    # Light randomization limits
    "key_int_min":  3000.0,
    "key_int_max":  8000.0,
    "fill_int_min": 300.0,
    "fill_int_max": 800.0,
    "dome_int_min": 300.0,
    "dome_int_max": 800.0,
    
    # Block rotation limits (degrees, applied around Z/vertical axis)
    "block_rot_max": 15.0,
    # Number of blocks rotated (0-9)
    "num_block_rot": 1,
}

USE_PATH_TRACING = False  # false for real-time
SPP = 32
TOTAL_SPP = 64

ACCEPTED_PALLET_TYPES: List[str] = ["epal","nlp"]

TEXTURES: List[str] = [#"C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/Isaac-sims/textures/plywood_diff_4k.jpg",
       "C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/Isaac-sims/textures/Material_003_baseColor.jpg"
        #"C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/Isaac-sims/textures/Texturelabs_Wood_266L.jpg",
       # "C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/Isaac-sims/textures/Texturelabs_Wood_267L.jpg",
        #"C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/Isaac-sims/textures/Texturelabs_Wood_268L.jpg"
        ]

# Camera intrinsics - need to match to real camera later
#RESOLUTION: Tuple[int, int]      = (2448, 2048)    # Zivid 2 M70 resolution
RESOLUTION: Tuple[int, int]      = (1224, 1048)  # Zivid 2 M70 resolution
FOCAL_LENGTH: float    = 5.94            # mm, derived from FOV
H_APERTURE: float     = 6.4             # mm, 1/2" sensor

PALLET_CENTRE: dict[str, Tuple[float, float, float]] = {
    "epal": (0.0, 0.072, 0.0),
    "nlp":  (0.0, 0.075, 0.0),
}
PALLET_ROTATIONS: List[Tuple[int, int, int]]  = [(0,0,0), (0,90,0), (0,180,0), (0,270,0)]

# Pallet_Blocks prim paths - Block_0 through Block_8
BLOCK_PATHS: List[str] = [
    "/scene/Meshes/Sketchfab_model/_53432bb09b84172864175516b644c7a_fbx/RootNode/Pallet_Blocks/Block_{}".format(i)
    for i in range(9)
]

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments, falling back to defaults if not provided."""
    parser = argparse.ArgumentParser(description="Pallet SDG Replicator")
    parser.add_argument("--pallet-type",  type=str, default="epal")
    parser.add_argument("--gen-liquid", type=int, default=None)
    #parser.add_argument("--gen-liquid", action="store_true", default=False)
    parser.add_argument("--pallet-path",  type=str,   default=DEFAULTS["pallet_path"],
                        help="Path to the pallet within Isaac Sim (Default: /scene/Meshes)")
    parser.add_argument("--mask-dir",     type=str, default=DEFAULTS["mask_dir"],
                        help="Path to masks for NLP pal contaminants")
    parser.add_argument("--shader-path",  type=str,   default=DEFAULTS["shader_path"],
                        help="Path to shader for NLP pallet (in sim)")
    parser.add_argument("--output-dir",   type=str,   default=DEFAULTS["output_dir"])
    parser.add_argument("--num-frames",   type=int,   default=DEFAULTS["num_frames"],
                        help="Number of frames for replicator to generate")
    
    #Lights and Camera Variables
    parser.add_argument("--cam-dist-min", type=float, default=DEFAULTS["cam_dist_min"])
    parser.add_argument("--cam-dist-max", type=float, default=DEFAULTS["cam_dist_max"])
    parser.add_argument("--cam-elev-min", type=float, default=DEFAULTS["cam_elev_min"])
    parser.add_argument("--cam-elev-max", type=float, default=DEFAULTS["cam_elev_max"])
    parser.add_argument("--key-int-min",  type=float, default=DEFAULTS["key_int_min"])
    parser.add_argument("--key-int-max",  type=float, default=DEFAULTS["key_int_max"])
    parser.add_argument("--fill-int-min", type=float, default=DEFAULTS["fill_int_min"])
    parser.add_argument("--fill-int-max", type=float, default=DEFAULTS["fill_int_max"])
    parser.add_argument("--dome-int-min", type=float, default=DEFAULTS["dome_int_min"])
    parser.add_argument("--dome-int-max", type=float, default=DEFAULTS["dome_int_max"])

    parser.add_argument("--textures",     type=str)
    # Block rotation args
    parser.add_argument("--block-rot-max",  type=float, default=DEFAULTS["block_rot_max"],
                        help="Max twist angle per block in degrees (default: 15)")
    parser.add_argument("--num-block-rot", type=int, default=DEFAULTS["num_block_rot"],
                        help="Probability each block is rotated each frame (default: 1)")
    parser.add_argument("--no-block-rot",   action="store_true", default=False,
                        help="Disable block rotation entirely")
    
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args

def check_pallet_type(pal_type: str) -> str:
    if not pal_type in ACCEPTED_PALLET_TYPES:
        sys.exit(f"Error: Unacceptable pallet type: {pal_type}")
    return pal_type

def set_render_mode() -> int:
    #mode: 'PathTracing' or 'RaytracedLighting'"""
    s = carb.settings.get_settings()
    if USE_PATH_TRACING:
        s.set("/rtx/rendermode", "PathTracing")
        s.set_int("/rtx/pathtracing/spp", SPP)
        s.set_int("/rtx/pathtracing/totalSpp", TOTAL_SPP)  # MUST be > 0
        return TOTAL_SPP//SPP
    else:
        s.set("/rtx/rendermode", "RaytracedLighting")
        return 1


# Spherical Camera Placement
def sample_camera_positions(
    n: int,
    centre: Tuple[float, float, float],
    dist_min: float,
    dist_max: float,
    elev_min_deg: float,
    elev_max_deg: float) -> List[Tuple[float, float, float]]:
    """
    Sample n camera positions on a sphere around centre.
    elevation is clamped to avoid ground-level or pure top-down shots.

    Args:
        n:            Number of positions to sample.
        centre:       World-space target point (pallet centre).
        dist_min:     Minimum distance from centre in metres.
        dist_max:     Maximum distance from centre in metres.
        elev_min_deg: Minimum elevation angle in degrees (above horizon).
        elev_max_deg: Maximum elevation angle in degrees.

    Returns:
        List of (x, y, z) camera positions in world space.
    """
    positions: List[Tuple[float, float, float]] = []
    for _ in range(n):
        distance: float  = random.uniform(dist_min, dist_max)
        azimuth: float   = random.uniform(0.0, 360.0)          # degrees, full circle
        elevation: float = random.uniform(elev_min_deg, elev_max_deg)  # degrees

        az_rad: float    = math.radians(azimuth)
        el_rad: float    = math.radians(elevation)

        x: float = centre[0] + distance * math.cos(el_rad) * math.sin(az_rad)
        y: float = centre[1] + distance * math.sin(el_rad)
        z: float = centre[2] + distance * math.cos(el_rad) * math.cos(az_rad)

        positions.append((x, y, z))

    return positions

def find_textures(texture_dict: str) -> List[str]:
    if texture_dict is None:
        return TEXTURES
    else:
        return sorted(glob.glob(texture_dict + "texture_*.png"))

def establish_masks(mask_dir) -> List[str]:
    mask_paths = sorted(glob.glob(mask_dir + "liquid_mask_*.png"))
    if not mask_paths:
        sys.exit(f"Error: --gen-liquid requested but no liquid_mask_*.png files found in {mask_dir}")
    return mask_paths

def create_camera() -> HydraTexture:
    # Create the SDG camera and render product matching given intrinsics
    camera = rep.create.camera(
        focal_length=FOCAL_LENGTH,
        horizontal_aperture=H_APERTURE,
        clipping_range=(0.1, 100.0),
        name="Camera"
    )
    render_product: HydraTexture = rep.create.render_product(camera, RESOLUTION)
    return camera, render_product

def create_lights() -> Tuple:
    #Create lights with baseline intensities
    key_light = rep.create.light(
        light_type="Distant", intensity=600,
        color=(1.0, 0.97, 0.9), rotation=(225, 0, 0), name="KeyLight")

    fill_light = rep.create.light(
        light_type="Sphere", intensity=400,
        color=(0.8, 0.85, 1.0), position=(-3.0, 2.0, 1.0), name="FillLight")

    dome_light = rep.create.light(light_type="Dome", intensity=300, name="DomeLight")
    return key_light, fill_light, dome_light

def randomize_camera(
        camera,camera_positions: List[Tuple[float, float, float]], pallet_path: str) -> None:
    #Randomize camera position each frame, always looking at the pallet
    with camera:
        rep.modify.pose(position=rep.distribution.choice(camera_positions),look_at=pallet_path)

def randomize_pallet(pallet) -> None:
    #Rotate pallet each frame - maybe unnecessary given changing camera positions
    with pallet:
        rep.modify.pose(rotation=rep.distribution.choice(PALLET_ROTATIONS))

def randomize_lights(
    key_light, fill_light, dome_light,
    key_int_min: float, key_int_max: float,
    fill_int_min: float, fill_int_max: float,
    dome_int_min: float, dome_int_max: float) -> None:
    #Randomize intensity, colour, and position of all lights each frame
    with key_light:
        rep.modify.attribute("inputs:intensity", rep.distribution.uniform(key_int_min, key_int_max))
        rep.modify.attribute("inputs:color",     rep.distribution.uniform((0.85,0.75,0.6), (1.0,1.0,1.0)))
        rep.modify.pose(rotation=rep.distribution.uniform((225,-15,0), (245,15,0)))

    with fill_light:
        rep.modify.pose(position=rep.distribution.uniform((-4.0,1.0,-3.0), (4.0,4.0,3.0)))
        rep.modify.attribute("inputs:intensity", rep.distribution.uniform(fill_int_min, fill_int_max))

    with dome_light:
        rep.modify.attribute("inputs:intensity", rep.distribution.uniform(dome_int_min, dome_int_max))

def randomize_texture(pallet, textures: List[str]) -> None:
    #Randomize base colour texture on materials each frame
    # Just wood at the moment
    texture = rep.distribution.choice(textures)
    with rep.get.prims(semantics=[("class", "pallet")]):
        rep.randomizer.texture(
            textures=texture,
            per_sub_mesh=False, # False to ensure all wood gelements are same texture - unfortunately assigns texture to nails
        )

def randomize_blocks(block_rot_max: float, num_block_rot: int, block_paths: List[str]) -> None:
    #Randomly twist individual pallet blocks around the vertical (Y) axis each frame.
    total_blocks = 9
    indices_to_rotate = random.sample(range(total_blocks), min(num_block_rot, total_blocks))

    for i, block_path in enumerate(block_paths):
        block = rep.get.prim_at_path(block_path)
        with block:
            if i in indices_to_rotate:
                twist = random.uniform(-block_rot_max, block_rot_max)
                rep.modify.pose(rotation=(0.0, 0.0, twist))
            else:
                rep.modify.pose(rotation=(0.0, 0.0, 0.0))

def randomize_decal(shader, mask_paths: str) -> None:
    with shader:
        rep.modify.attribute(
            "inputs:opacity_texture",
            rep.distribution.choice(mask_paths)
        )

def attach_writer(render_product: HydraTexture, output_dir: str) -> Writer:
    #Initialise BasicWriter with all required annotators and attach to render product
    writer: Writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=output_dir,
        rgb=True,
        bounding_box_2d_tight=False,
        bounding_box_2d_loose=False,
        bounding_box_3d=False,
        instance_segmentation=False,
        semantic_segmentation=False,
        distance_to_camera=False,
        #pointcloud=True,
        #pointcloud_include_unlabelled=False,
        normals=False,
        camera_params=False,
    )
    writer.attach([render_product])
    return writer

def get_frame_stats():
    h = hstats.HydraEngineStats()
    result = h.get_gpu_profiler_result()
    if result and result[0]:
        frame_ms = result[0][0]["duration"]
        fps = 1000.0 / frame_ms if frame_ms > 0 else 0.0
        return fps, frame_ms
    return None, None
    
def write_run_summary(output_dir, num_frames, pal_type, gen_liquid, args):
    fps, frame_ms = get_frame_stats()
    render_mode = carb.settings.get_settings().get("/rtx/rendermode")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"run_summary_{timestamp}.txt")

    lines = [
        "=== SDG Run Summary ===",
        f"Timestamp:         {timestamp}",
        f"",
        f"--- Scene ---",
        f"Pallet type:       {pal_type}",
        f"Pallet path:       {args.pallet_path}",
        f"Contaminant decal: {gen_liquid}",
        f"",
        f"--- Output ---",
        f"Output dir:        {args.output_dir}",
        f"Num frames:        {num_frames}",
        f"Resolution:        {RESOLUTION[0]}x{RESOLUTION[1]}",
        f"",
        f"--- Camera ---",
        f"Focal length:      {FOCAL_LENGTH} mm",
        f"Horizontal apt:    {H_APERTURE} mm",
        f"Distance range:    {args.cam_dist_min} – {args.cam_dist_max} m",
        f"Elevation range:   {args.cam_elev_min} – {args.cam_elev_max} deg",
        f"",
        f"--- Lighting ---",
        f"Key intensity:     {args.key_int_min} – {args.key_int_max}",
        f"Fill intensity:    {args.fill_int_min} – {args.fill_int_max}",
        f"Dome intensity:    {args.dome_int_min} – {args.dome_int_max}",
        f"",
        f"--- Block Rotation ---",
        f"Enabled:           {not args.no_block_rot}",
        f"Blocks rotated:    {args.num_block_rot} / {len(BLOCK_PATHS)}",
        f"Max twist angle:   +/- {args.block_rot_max} deg",
        f"",
        f"--- Render ---",
        f"Render mode:       {render_mode}",
        f"Path tracing:      {USE_PATH_TRACING}",
        f"SPP / Total SPP:   {SPP} / {TOTAL_SPP}" if USE_PATH_TRACING else f"SPP / Total SPP:   N/A (realtime)",
        f"GPU frame time:    {frame_ms:.2f} ms" if frame_ms else f"GPU frame time:    N/A",
        f"FPS:               {fps:.1f}" if fps else f"FPS:               N/A",
    ]

    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[Summary] Written to {summary_path}")

    
async def run_replicator(num_frames: int, output_dir: str, pal_type:str, gen_liquid: bool, args) -> None:
    await rep.orchestrator.run_async(num_frames=num_frames+1)#+1 for silently consumed frame in startup 
    await rep.orchestrator.wait_until_complete_async()
    write_run_summary(output_dir, num_frames, pal_type, gen_liquid, args)

def main() -> None:
    args = parse_args()
    pal_type: str = check_pallet_type(args.pallet_type)

    subframes: int = set_render_mode()

    camera_positions: List[Tuple[float, float, float]] = sample_camera_positions(
        n=args.num_frames, 
        centre=PALLET_CENTRE[pal_type],
        dist_min=args.cam_dist_min,
        dist_max=args.cam_dist_max,
        elev_min_deg=args.cam_elev_min,
        elev_max_deg=args.cam_elev_max,
    )
    textures: List[str] = find_textures(args.textures)
    gen_liquid: bool = bool(args.gen_liquid)
    if gen_liquid: mask_paths: List[str] = establish_masks(args.mask_dir)

    with rep.new_layer():

        pallet = rep.get.prim_at_path(args.pallet_path)
        camera, render_product = create_camera()
        key_light, fill_light, dome_light = create_lights()
        if gen_liquid: shader = rep.get.prim_at_path(args.shader_path)

        with rep.trigger.on_frame(max_execs=args.num_frames+1, rt_subframes=subframes): #+1 for silently consumed frame in startup
            randomize_camera(camera, camera_positions, args.pallet_path)
            randomize_pallet(pallet)
            randomize_lights(
                key_light, fill_light, dome_light,
                args.key_int_min, args.key_int_max,
                args.fill_int_min, args.fill_int_max,
                args.dome_int_min, args.dome_int_max,
            )
            if pal_type == "epal":
                randomize_texture(pallet, textures)
                if not args.no_block_rot:
                    randomize_blocks(args.block_rot_max, args.num_block_rot, BLOCK_PATHS)
            if gen_liquid : randomize_decal(shader,mask_paths)

        attach_writer(render_product, args.output_dir)

    asyncio.ensure_future(run_replicator(args.num_frames, args.output_dir, pal_type, gen_liquid, args)) 

if __name__ == "__main__":
    main()

