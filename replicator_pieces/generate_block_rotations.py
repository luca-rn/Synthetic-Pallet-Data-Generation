# Generates a schedule for block rotation to use in isaac sim
# randomised because that gives wider distribution of data
# alternative is a procedural list (incrementally increasing), but randomised is likely better for more robust learning on less data
# incremental can be implemented is needed

import argparse
import json
import random, math
import sys
from pathlib import Path

# Block definitions - according to USD file "blocks_pallet"
ALL_BLOCKS    = [0, 1, 2, 3, 4, 5, 6, 7, 8]
EDGE_BLOCKS   = [0, 2, 3, 5, 6, 8]
MIDDLE_BLOCKS = [1, 4, 7]

# Named focus presets — add more here as you add camera positions
FOCUS_PRESETS = {
    "all": ALL_BLOCKS,
    "front": [0, 1, 2], # Camera at the front end of the pallet
    "back" : [6, 7, 8], # Camera at the far end of the pallet
}

# Deciding how likely one or more blocks are to rotate
#                  0   1   2   3   4  5  6  7  8  9
COUNT_WEIGHTS   = [0, 85, 10, 4, 1, 0, 0, 0, 0, 0]

RANGES = {
    "edge": {
        "normal": (0.0,   8.5),   # nearly always near 0
        "defect": (8.5,  160.0), # tends towards lower range
    },
    "middle": {
        "normal": (0.0,  20.0),  # still near 0
        "defect": (20.0, 160.0), # tends towards lower range
    },
}

WEIBULL_K = 0.8 # Higher k puts distribution peak further towards higher range. 
                # K<1 creates spike at 9, K=1 is exponential from 9, K>1 creates hump
WEIBULL_LAMBDA = 70 # Lower lambda means less higher range cases


def weibull_sample(low, high):
    while True:
        v = low + WEIBULL_LAMBDA * (-math.log(random.random())) ** (1/WEIBULL_K)
        if v <= high: return v

def sample_angle(block_idx, defect_mode):
    # Return a rotation angle for a given block index."""
    key = "middle" if block_idx in MIDDLE_BLOCKS else "edge"
    lo, hi = RANGES[key]["defect" if defect_mode else "normal"]
    return round(weibull_sample(lo, hi), 4)

def sample_block_count(max_blocks):
    # pick how many blocks to rotate, capped at max (number in focus)
    valid_counts = list(range(len(COUNT_WEIGHTS)))
    valid_weights = list(COUNT_WEIGHTS)
 
    # Zero out counts that exceed the number of in-focus blocks
    for i in range(max_blocks + 1, len(valid_counts)):
        valid_weights[i] = 0
 
    return random.choices(valid_counts, weights=valid_weights, k=1)[0]

def build_schedule(num_frames, defect_mode, focus_blocks): 
    # Build list of which blocks to rotate and how much per frame
    schedule = []
    for _ in range(num_frames):
        count = sample_block_count(len(focus_blocks))
        chosen = random.sample(focus_blocks, count)
 
        entry = {}
        for idx in chosen:
            entry[str(idx)] = sample_angle(idx, defect_mode)
 
        schedule.append(entry)
 
    return schedule

def parse_focus(raw):
    # Accept a named preset ("all", "end") or a comma-separated index list
    # ("5,6,7,8" or "0,1,2,6,7,8"). Returns a sorted list of ints.
    raw = raw.strip().lower()
 
    if raw in FOCUS_PRESETS:
        return FOCUS_PRESETS[raw]
 
    try:
        indices = [int(x.strip()) for x in raw.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid --focus value '{}'. Use a preset ({}) "
            "or a comma-separated list of block indices (e.g. 0,1,2).".format(
                raw, ", ".join(FOCUS_PRESETS.keys())
            )
        )
 
    invalid = [i for i in indices if i not in ALL_BLOCKS]
    if invalid:
        raise argparse.ArgumentTypeError(
            "Block indices {} are out of range. Valid range is 0-8.".format(invalid)
        )
 
    return sorted(set(indices))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a pallet block rotation schedule."
    )
    parser.add_argument(
        "--defect", action="store_true",
        help="Use defect rotation ranges."
    )
    parser.add_argument(
        "--focus", type=str, default="all", metavar="GROUP",
        help=(
            "Which blocks are in-camera focus. "
            "Presets: all (0-8), front (0,1,2), back (6,7,8). "
            "Or pass a custom list: '0,1,2,3'. Default: all."
        )
    )
    parser.add_argument(
        "--frames", type=int, default=50, metavar="N",
        help="Number of frames to generate (default: 50)."
    )
    parser.add_argument(
        "--out", type=str, default="rotation_schedule.json", metavar="PATH",
        help="Output JSON file path (default: rotation_schedule.json)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
 
    try:
        focus_blocks = parse_focus(args.focus)
    except argparse.ArgumentTypeError as e:
        print("Error: " + str(e))
        raise SystemExit(1)
    
    print("Generating rotation schedule...")
    print("  Mode        : {}".format("DEFECT" if args.defect else "NORMAL"))
    print("  Focus blocks: {}".format(focus_blocks))
    print("  Frames      : {}".format(args.frames))
    print("  Output      : {}".format(args.out))

    schedule = build_schedule(args.frames, args.defect, focus_blocks)

    output = {
        "meta": {
            "mode":           "defect" if args.defect else "normal",
            "focus_blocks":   focus_blocks,
            "frames":         args.frames,
            "edge_blocks":    EDGE_BLOCKS,
            "middle_blocks":  MIDDLE_BLOCKS,
            "count_weights":  COUNT_WEIGHTS,
            "ranges":         RANGES,
        },
        "schedule": schedule,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(output, indent=2))

    print("\nSaved {} frames to: {}".format(len(schedule), out_path.resolve()))

if __name__ == "__main__":
    main()
