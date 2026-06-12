"""
generate_liquid_mask.py
-----------------------
Generates a transparent-background liquid splash mask PNG
from a fixed set of parameters.

Requirements:
    pip install numpy Pillow

Usage:
    python generate_liquid_mask.py

Output:
    liquid_mask.png  — RGBA PNG
"""

import argparse
import json
import numpy as np
from PIL import Image
import os


# ─────────────────────────────────────────────
# PARAMETERS — tune these (use the test_liquid_mask.html to check)
# ─────────────────────────────────────────────

DEFAULT_PARAMS = {
    "scale":           4.0,
    "octaves":         3,
    "persistence":     0.59,
    "warp":            0.3,
    "threshold":       0.49,
    "edge_soft":       0.04,
    "color":           (255, 255, 255),
    "seed":            None,
    "offset_x":        None,
    "offset_y":        None,
    "pallet_x_min":   -0.5999,
    "pallet_x_max":    0.5999,
    "pallet_y_min":   -0.3999,
    "pallet_y_max":    0.3999,
    "pallet_mask_png": "pallet_solid_mask.png",
    "base_width":      512,
}

# ─────────────────────────────────────────────────────────────────────────────


def _build_perm(seed_int: int) -> np.ndarray:
    """Build a 512-length permutation table from an integer seed (LCG shuffle)."""
    p = list(range(256))
    s = seed_int & 0xFFFFFFFF
    for i in range(255, 0, -1):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        j = s % (i + 1)
        p[i], p[j] = p[j], p[i]
    perm = np.array(p * 2, dtype=np.uint8)
    return perm


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6 - 15) + 10)


def _grad2(h: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h = h & 3
    u = np.where(h < 2, x, y)
    v = np.where(h < 2, y, x)
    return np.where(h & 1, -u, u) + np.where(h & 2, -v, v)


def _perlin_grid(xs: np.ndarray, ys: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Vectorised 2-D Perlin noise over a 2-D grid of (xs, ys)."""
    xi = np.floor(xs).astype(np.int32) & 255
    yi = np.floor(ys).astype(np.int32) & 255
    xf = xs - np.floor(xs)
    yf = ys - np.floor(ys)
    u = _fade(xf)
    v = _fade(yf)

    aa = perm[perm[xi]     + yi    ]
    ab = perm[perm[xi]     + yi + 1]
    ba = perm[perm[xi + 1] + yi    ]
    bb = perm[perm[xi + 1] + yi + 1]

    x1 = xf - 1
    y1 = yf - 1
    n00 = _grad2(aa, xf, yf)
    n10 = _grad2(ba, x1, yf)
    n01 = _grad2(ab, xf, y1)
    n11 = _grad2(bb, x1, y1)

    ix0 = n00 + u * (n10 - n00)
    ix1 = n01 + u * (n11 - n01)
    return ix0 + v * (ix1 - ix0)


def _fbm(xs: np.ndarray, ys: np.ndarray,
         octaves: int, persistence: float,
         perm: np.ndarray) -> np.ndarray:
    """Fractional Brownian motion — sum of Perlin octaves."""
    value = np.zeros_like(xs)
    amplitude = 1.0
    frequency = 1.0
    max_val = 0.0
    for _ in range(octaves):
        value += _perlin_grid(xs * frequency, ys * frequency, perm) * amplitude
        max_val += amplitude
        amplitude *= persistence
        frequency *= 2.0
    return value / max_val

def build_pallet_mask(params: dict, W: int, H: int) -> np.ndarray:
    """
    Load the pre-baked pallet solid-surface mask PNG and resize it to W x H.
 
    The mask PNG is generated once from Isaac Sim using extract_pallet_mask.py
    and lives alongside this script — no Isaac Sim needed at runtime.
 
    Returns a uint8 numpy array (H, W): 255 = solid surface, 0 = hole/background.
    """
    mask_path = params.get("pallet_mask_png")
    if not mask_path:
        mask_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "pallet_solid_mask.png"
        )
    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Pallet mask not found at '{mask_path}'."
            "Run extract_pallet_mask.py inside Isaac Sim once to generate it."
        )
    mask = Image.open(mask_path).convert("L").resize((W, H), Image.LANCZOS)
    return np.array(mask)

def generate(params: dict,  output: str) -> Image.Image:
    # Derive W×H from pallet aspect ratio so texture maps 1-to-1, no stretching
    pallet_w = params["pallet_x_max"] - params["pallet_x_min"]
    pallet_h = params["pallet_y_max"] - params["pallet_y_min"]
    W = params["base_width"]
    H = int(round(W * pallet_h / pallet_w))
 
    scale       = params["scale"]
    octaves     = params["octaves"]
    persistence = params["persistence"]
    warp_str    = params["warp"]
    threshold   = params["threshold"]
    edge_soft   = params["edge_soft"]
    color       = params["color"]
 
    rng = np.random.default_rng(params.get("seed"))
    seed_int = int(rng.integers(0, 2**31))
    warp_seed_int = int(rng.integers(0, 2**31))
 
    ox = params.get("offset_x") or rng.random() * 10000
    oy = params.get("offset_y") or rng.random() * 10000
 
    perm      = _build_perm(seed_int)
    warp_perm = _build_perm(warp_seed_int)
 
    # Build coordinate grids
    gx, gy = np.meshgrid(
        np.linspace(0, scale, W, endpoint=False),
        np.linspace(0, scale, H, endpoint=False),
    )
    gx += ox
    gy += oy
 
    warp_oct = min(octaves, 4)
    wx = _fbm(gx + 1.7, gy + 9.2, warp_oct, persistence, warp_perm) * warp_str
    wy = _fbm(gx + 8.3, gy + 2.8, warp_oct, persistence, warp_perm) * warp_str
 
    noise = _fbm(gx + wx, gy + wy, octaves, persistence, perm)
    n01 = (noise + 1.0) * 0.5   # remap [-1,1] → [0,1]
 
    lo = threshold - edge_soft * 0.5
    hi = threshold + edge_soft * 0.5
 
    alpha = np.where(
        n01 <= lo, 255.0,
        np.where(n01 >= hi, 0.0,
                 255.0 * (1.0 - (n01 - lo) / (hi - lo)))
    ).astype(np.uint8)
 
    # Apply pallet solid-surface mask — zeros out holes and background.
    # Requires Isaac Sim; skipped gracefully if unavailable.
    try:
        solid = build_pallet_mask(params, W, H)
        alpha = ((alpha.astype(np.float32) * solid.astype(np.float32)) / 255.0).astype(np.uint8)
    except Exception as e:
        print(f"  [pallet mask skipped: {e}]")

    r, g, b = color
    rgba = np.stack([
        np.full((H, W), r, dtype=np.uint8),
        np.full((H, W), g, dtype=np.uint8),
        np.full((H, W), b, dtype=np.uint8),
        alpha,
    ], axis=-1)

    img = Image.fromarray(rgba, mode="RGBA")
    img.save(output)
    return img

def _parse_args():
    parser = argparse.ArgumentParser(description="Generate a liquid mask PNG.")
    parser.add_argument("--output", default="liquid_mask.png",
                        help="Output file path (default: liquid_mask.png)")
    parser.add_argument("--params", default=None,
                        help="JSON string or path to a JSON file of param overrides")
    # Individual param overrides
    parser.add_argument("--scale",       type=float)
    parser.add_argument("--octaves",     type=int)
    parser.add_argument("--persistence", type=float)
    parser.add_argument("--warp",        type=float)
    parser.add_argument("--threshold",   type=float)
    parser.add_argument("--edge_soft",   type=float)
    parser.add_argument("--seed",        type=int)
    parser.add_argument("--base_width",  type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
 
    # Start from defaults
    params = dict(DEFAULT_PARAMS)
 
    # Apply JSON overrides if provided
    if args.params:
        if os.path.isfile(args.params):
            with open(args.params) as f:
                overrides = json.load(f)
        else:
            overrides = json.loads(args.params)
        params.update(overrides)
 
    # Apply individual CLI overrides
    for key in ("scale", "octaves", "persistence", "warp",
                "threshold", "edge_soft", "seed", "base_width"):
        val = getattr(args, key)
        if val is not None:
            params[key] = val

    output = args.output
 
    print(f"Generating → {output}")
    img = generate(params, output)
    print(f"Saved → {output}  ({img.width}×{img.height})")