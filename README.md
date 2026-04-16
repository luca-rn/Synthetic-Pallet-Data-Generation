# Pallet SDG — Isaac Sim Synthetic Data Generation

Synthetic data generation (SDG) pipeline for a EUR pallet using NVIDIA Isaac Sim and Omniverse Replicator.

## Overview

This pipeline renders randomised camera views of a EUR pallet (1.2 × 0.8 × 0.144 m) with randomised lighting, producing annotated image datasets for training object detection/pose estimation models.

## Files

| File | Description |
|------|-------------|
| `stage-setup.py` | Opens the USD scene, verifies scale/axis, and attaches a `pallet` semantic label. Run once per session before the replicator. |
| `replicator.py` | Runs the Replicator loop — randomises camera pose, pallet rotation, and lighting, then writes annotated frames to disk. |
| `---.usd` | USD files used as the stage entry point |
| `example_output` | Contains the output of a test run of the replicator script - with NUM_FRAMES = 10 |

## IMPORTANT - files path
Path to the usd in `stage-setup` is hardcoded.
Output dict is for writing is hardcoded in `replicator_sript_1.py`.

## Usage

Both scripts are run inside the **Isaac Sim Script Editor**.

1. Open Isaac Sim.
2. Run `stage-setup.py` to load and label the scene.
3. Run `replicator_script_1.py` to generate the dataset.

## Running Headless Isaac Sim (Remote - No GUI)
Stage Setup:
`./isaac-sim.headless.bat --/omni/replicator/script="stage_setup.py" --usd-path "C:/path/to/scene.usd" --pallet-path "/scene/Meshes"`
Replicator:
`./isaac-sim.headless.bat --/omni/replicator/script="replicator.py" --output-dir "C:/my_output" --num-frames 100`

## Output Annotations

Each frame is saved with:
- RGB image
- 2D bounding boxes (tight & loose)
- 3D bounding box
- Instance & semantic segmentation
- Depth (distance to camera)
- Surface normals
- Camera parameters

## Camera Configuration

| Parameter | Value |
|-----------|-------|
| Resolution | 2448 × 2048 px (M70 sensor) |
| Focal length | 5.94 mm |
| Horizontal aperture | 6.4 mm (1/2" sensor) |
| Distance range | 1.3 – 2.3 m |
| Elevation range | 15° – 75° |

## Scene Units

- **Coordinate system:** Y-up  
- **Scale:** 1 unit = 1 metre

## Dependencies

- NVIDIA Isaac Sim (with Omniverse Replicator)
- `omni.replicator.core`
- `omni.usd`, `omni.kit.app`
- `pxr` (OpenUSD Python bindings)
