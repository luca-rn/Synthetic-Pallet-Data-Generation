print("Running replicator script...")

import omni.replicator.core as rep

PALLET_PATH  = "/scene/Meshes"
OUTPUT_DIR   = "C:/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/SDG_output"
NUM_FRAMES   = 10

# Camera intrinsics - need to match to real camera later
RESOLUTION   = (1280, 720)
FOCAL_LENGTH = 24.0     # mm
H_APERTURE   = 20.955   # mm

# Light randomization
KEY_INT_MIN  = 300
KEY_INT_MAX  = 4000
FILL_INT_MIN = 100
FILL_INT_MAX = 800
DOME_INT_MIN = 100
DOME_INT_MAX = 800

# Pallet rotation -cardinal orientations only
PALLET_ROTATIONS = [(0,0,0), (0,90,0), (0,180,0), (0,270,0)]

with rep.new_layer():

    # Define scene objects
    pallet = rep.get.prim_at_path(PALLET_PATH)

    # Define camera
    camera = rep.create.camera(focal_length=FOCAL_LENGTH, horizontal_aperture=H_APERTURE,
    clipping_range=(0.5, 5000.0), name="SDGCamera") # in cm
    
    render_product = rep.create.render_product(camera, RESOLUTION)
    
    # Create lights
    key_light = rep.create.light(light_type="Distant", intensity=600, color=(1.0, 0.97, 0.9), rotation=(225, 0, 0), name="KeyLight" )
    fill_light = rep.create.light(light_type="Sphere",  intensity=400,  color=(0.8, 0.85, 1.0), position=(-300, 200, 100),   name="FillLight") # pos in cm
    dome_light = rep.create.light(light_type="Dome",    intensity=300, name="DomeLight")

    # Dfine randomizations - So far: Pose and Light
    with rep.trigger.on_frame(max_execs=NUM_FRAMES):

        with pallet:
            rep.modify.pose(
                rotation=rep.distribution.choice(PALLET_ROTATIONS)
            )

        with key_light:
            rep.modify.attribute("inputs:intensity", rep.distribution.uniform(KEY_INT_MIN, KEY_INT_MAX))
            rep.modify.attribute("inputs:color",     rep.distribution.uniform((0.85,0.75,0.6), (1.0,1.0,1.0)))
            rep.modify.pose(rotation=rep.distribution.uniform((200,-30,0), (260,30,0)))

        with fill_light:
            rep.modify.pose(position=rep.distribution.uniform((-400, 100, -300), (400, 400, 300)))
            rep.modify.attribute("inputs:intensity", rep.distribution.uniform(FILL_INT_MIN, FILL_INT_MAX))

        with dome_light:
            rep.modify.attribute("inputs:intensity", rep.distribution.uniform(DOME_INT_MIN, DOME_INT_MAX))


    # 4. Define what to capture
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=OUTPUT_DIR,
        rgb=True,
        bounding_box_2d_tight=True,
        bounding_box_2d_loose=True,
        bounding_box_3d=True,
        instance_segmentation=True,
        semantic_segmentation=True,
        distance_to_camera=True,
        normals=True,
        camera_params=True,
    )
    writer.attach([render_product])
    
rep.orchestrator.run()
print(f"[Replicator] Done. {NUM_FRAMES} frames written to {OUTPUT_DIR}")
