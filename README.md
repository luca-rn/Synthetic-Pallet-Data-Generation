# Pallet SDG — Isaac Sim Synthetic Data Generation

Synthetic data generation (SDG) pipeline for EUR pallets using NVIDIA Isaac Sim and Omniverse Replicator. Supports two pallet variants with randomised contamination and block configurations, plus a YOLO-based contamination classifier trained on the generated data.

## Overview

The pipeline renders randomised camera views of a EUR pallet (1.2 × 0.8 × 0.144 m) with randomised lighting, producing annotated image datasets for training object detection and classification models.

Two pallet types are supported:

| Type | Description |
|------|-------------|
| `epal` | Wooden EUR pallet with 9 rotatable blocks on top |
| `nlp` | Plastic NLP pallet with a liquid-contaminant decal |

## Repository Structure

```
Isaac-sims/
├── stage-setup.py               # Scene initialisation — run once per session
├── replicator.py                # Main SDG script — CLI-driven
├── replicator_pieces/           # Standalone sub-scripts (individual features)
│   ├── blocks_setup.py          # EPAL block visibility/rotation setup
│   ├── liquid_rep.py            # NLP liquid-contaminant replicator
│   ├── rotate_blocks.py         # Schedule-driven block rotation replicator
│   ├── liquid_mask_stage_setup.py
│   ├── replicator_main.py
│   ├── generate_block_rotations.py
│   └── single_pal_liquid_rep.py
├── liquid_generation/           # Liquid mask generation utilities
│   ├── generate_liquid_mask.py  # Generate a single procedural mask
│   └── gen_many_masks.py        # Batch mask generation
├── ML_training/
│   └── contam_classification.py # YOLO11/26 contamination classifier training
├── data_handling/               # Post-processing utilities
│   ├── pointcloud_viewer.py     # Open3D point cloud visualiser
│   ├── analyse_cases.py
│   ├── sort_analysed_cases.py
│   └── quicksort_single_cam.py
├── usd_files/                   # USD stage assets
├── textures/                    # Pallet surface material textures
├── pyproject.toml               # Project dependencies (uv)
└── requirements.txt             # Pip-compatible dependency list
```

## Usage

Both `stage-setup.py` and `replicator.py` are run inside the **Isaac Sim Script Editor** or headless via CLI.

### 1. Stage Setup

Loads the USD scene, deactivates the inactive pallet, and attaches semantic labels.

**GUI (Script Editor):** open and run `stage-setup.py` directly — uses defaults.

**Headless:**
```bash
./isaac-sim.headless.bat \
    --/omni/replicator/script="stage-setup.py" \
    -- \
    --pallet-type epal
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--pallet-type` | `epal` | `epal` or `nlp` |
| `--usd-path` | auto (repo-relative) | Path to the USD stage file |
| `--mask-dir` | auto (repo-relative) | Directory of liquid mask PNGs (NLP only) |

### 2. Replicator

Runs the randomisation loop and writes annotated frames to disk.

**GUI (Script Editor):** open and run `replicator.py` directly — uses defaults (100 frames, EPAL, no liquid).

**Headless:**
```bash
./isaac-sim.headless.bat \
    --/omni/replicator/script="replicator.py" \
    -- \
    --pallet-type epal \
    --num-frames 500 \
    --output-dir "C:/my_output" \
    --gen-liquid \
    --rotation-schedule "replicator_pieces/rotation_schedule.json"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--pallet-type` | `epal` | `epal` or `nlp` |
| `--num-frames` | `100` | Number of frames to render |
| `--output-dir` | auto | Output directory for annotated images |
| `--gen-liquid` | off | Enable liquid-contaminant decal (NLP) |
| `--mask-dir` | auto | Directory of liquid mask PNGs |
| `--rotation-schedule` | `None` | JSON schedule file for block rotations (EPAL) |
| `--colourful` | on | Widen decal colour gamut |
| `--path-trace` | off | Use path tracing instead of real-time renderer |

### 3. Liquid Mask Generation

Generate procedural liquid masks before running the NLP replicator:

```bash
python liquid_generation/gen_many_masks.py --count 50 --output-dir liquid_generation/masks/
```

### 4. ML Training

Train a YOLO contamination classifier on generated data. Configure paths at the top of the script before running:

```bash
python ML_training/contam_classification.py
```

Trains `yolo11s` and `yolo26s` classification models with 150 epochs at 224 × 224 px and produces comparison plots.

## Output Annotations

Each frame is written by `BasicWriter` with:

- RGB image
- 2D bounding boxes (tight & loose)
- 3D bounding box
- Instance & semantic segmentation
- Depth (distance to camera)
- Camera parameters (intrinsics + extrinsics)

## Camera Configuration

| Parameter | Value |
|-----------|-------|
| Resolution | 1224 × 1048 px |
| Focal length | 5.94 mm |
| Horizontal aperture | 6.4 mm (1/2" sensor) |
| Viewpoints | 8 fixed positions (4 per side) |
| Look-at | pallet centre |

## Scene Units

- **Coordinate system:** Y-up
- **Scale:** 1 unit = 1 metre

## Dataset

The Synthetic dataset generated for NLP contaminants is hosted on Zenodo:

**[https://zenodo.org/records/20723502](https://zenodo.org/records/20723502)**

The rest of the data cannot be made available due to an NDA.

## Dependencies

Install with [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

PyTorch with CUDA 12.6 is configured automatically via `pyproject.toml`.

The following modules are bundled with NVIDIA Isaac Sim and are **not** pip-installable — scripts using them must be run inside the Isaac Sim Python environment:

- `omni.replicator.core`
- `omni.usd`, `omni.kit.app`
- `pxr` (OpenUSD Python bindings)
- `carb`
