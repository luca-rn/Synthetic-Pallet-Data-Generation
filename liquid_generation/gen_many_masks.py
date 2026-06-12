import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_liquid_mask import generate, DEFAULT_PARAMS

# PARAMETERS — tune these (use the test_liquid_mask.html to check)
# Examples
# 4.0,3,0.59,0.30,0.49,0.04,512
# 2.5,6,0.49,0.12,0.57,0.08,512
# 4.0,4,0.30,0.12,0.47,0.08,512
# 2.0,3,0.59,0.06,0.42,0.08,512

PARAMS = {
    **DEFAULT_PARAMS,
    "scale":       6.0,
    "octaves":     4,
    "persistence": 0.50,
    "warp":        0.15,
    "threshold":   0.49,
    "edge_soft":   0.06,
    "base_width":  512,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Batch-generate N liquid mask PNGs.")
    parser.add_argument("--count",   type=int, default=10,
                        help="Number of masks to generate (default: 10)")
    parser.add_argument("--out_dir", default="masks",
                        help="Output directory (default: liquid_masks/)")
    parser.add_argument("--prefix",  default="liquid_mask",
                        help="Filename prefix (default: liquid_mask)")
    return parser.parse_args()

def main():
    args = parse_args()
 
    os.makedirs(args.out_dir, exist_ok=True)
 
    print(f"Generating {args.count} mask(s) → {os.path.abspath(args.out_dir)}/")
    print()

    for i in range(args.count):
        run_params = dict(PARAMS)
        run_params["seed"] = None   # random each time
 
        filename = f"{args.prefix}_{i:04d}.png"
        output_path = os.path.join(args.out_dir, filename)
 
        img = generate(run_params, output_path)
        print(f"[{i+1:>{len(str(args.count))}}/{args.count}] {filename}  ({img.width}×{img.height})")

if __name__ == "__main__":
    main()