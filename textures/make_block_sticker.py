#!/usr/bin/env python3
"""
Overlays EPAL block markings onto a JPG texture
Output is always 1024x1024 px to match original Material.002 dimensions.
Usage:
    uv run overlay_epal_markings.py --input concrete.jpg --markings markings.png --output concrete_epal.jpg
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

TARGET_SIZE = (1024, 1024)

def parse_args():
    parser = argparse.ArgumentParser(description="Overlay EPAL markings onto a JPG texture.")
    parser.add_argument("--input",    required=True,  help="Path to input JPG texture")
    parser.add_argument("--markings", required=True,   help="Path to EPAL_markings.png")
    parser.add_argument("--output",   default="marked_block.jpg", help="Output JPG filename")
    return parser.parse_args()

def validate_args(args):
    """Validate argument values and exit with a clear message on failure."""
    for filepath in [args.input, args.markings]:
        if not os.path.isfile(filepath):
            sys.exit(f"Error: file not found: {filepath}")

def load_background(path):
    #Load a JPG as RGB and scale+centre-crop it to TARGET_SIZE
    bg = Image.open(path).convert("RGB")
    if bg.size == TARGET_SIZE:
        return bg
    return crop_to_target(bg)

def crop_to_target(image):
    #Scale an image so its short side fills TARGET_SIZE, then centre-crop
    scale = max(TARGET_SIZE[0] / image.width, TARGET_SIZE[1] / image.height)
    new_w = int(image.width  * scale)
    new_h = int(image.height * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - TARGET_SIZE[0]) // 2
    top  = (new_h - TARGET_SIZE[1]) // 2
    return resized.crop((left, top, left + TARGET_SIZE[0], top + TARGET_SIZE[1]))

def load_alpha_mask(markings_path):
    #Load the EPAL markings PNG and return its alpha channel.
    markings = Image.open(markings_path).convert("RGBA")
    if markings.size != TARGET_SIZE:
        sys.exit("Error: Markings input of unexpected size")
    return np.array(markings)[:, :, 3].astype(np.float32) / 255.0

def composite(background, alpha_mask):
    #Alpha-blend ink_color over background using alpha_mask
    alpha3 = alpha_mask[:, :, np.newaxis]
    blended = (1.0 - alpha3) * np.array(background, dtype=np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)

def save_result(array, output_path):
    #Save a uint8 RGB array as a JPEG file
    Image.fromarray(array, "RGB").save(output_path, format="JPEG", quality=95)

def main():
    args          = parse_args()
    validate_args(args)

    background    = load_background(args.input)
    alpha_mask    = load_alpha_mask(args.markings)
    result    = composite(background, alpha_mask)
    save_result(result, args.output)

if __name__ == "__main__":
    main()