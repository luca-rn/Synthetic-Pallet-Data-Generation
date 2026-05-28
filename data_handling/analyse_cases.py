#NOTE: THIS IS A SCRIPT TO RUN TOOLS.PY (SOLWR's case analysis) IN LINUX

import pathlib
import subprocess
import argparse
from tqdm import tqdm
from datetime import datetime

DEFAULT_DATA_DIR = pathlib.Path("/mnt/c/Users/snook/Desktop/Uni_Stuff/NTNU/Thesis/data")
DEFAULT_BATCH_SIZE = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Batch update case folders using tools.py")
    parser.add_argument("data_dir", type=pathlib.Path, nargs="?", default=DEFAULT_DATA_DIR, help=f"Path to the directory containing case folders (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Number of cases per batch (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of cases to process (default: all)")
    parser.add_argument("--output", type=pathlib.Path, default=None, help="Path for the output txt file (default: <data_dir>/run_<timestamp>.txt)")
    return parser.parse_args()


def run_update(data_dir, batch_size, limit=None, output=None):
    if not data_dir.is_dir():
        raise NotADirectoryError(f"'{data_dir}' is not a valid directory.")

    case_folders = [f for f in data_dir.iterdir() if f.is_dir()]
    if limit is not None:
        case_folders = case_folders[:limit]

    print(f"Found {len(case_folders)} potential cases. Starting update...")

    output_path = output or data_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    succeeded, failed = [], []

    batches = [case_folders[i:i + batch_size] for i in range(0, len(case_folders), batch_size)]

    for batch in tqdm(batches):
        cmd = ["uv", "run", "tools.py", "update"] + [str(f) for f in batch] + ["--repetitions", "1", "-y"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"\n[ERROR] Failed on batch: {[f.name for f in batch]}")
                print(f"Details: {result.stderr}")
                failed.extend(f.name for f in batch)
            else:
                for folder in batch:
                    print(f"Successfully updated: {folder.name}")
                    succeeded.append(folder.name)

        except Exception as e:
            print(f"System error during batch: {e}")
            failed.extend(f.name for f in batch)

    with open(output_path, "w") as f:
        f.write(f"Run completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Batch size: {batch_size} | Limit: {limit or 'none'}\n")
        f.write(f"\n--- Succeeded ({len(succeeded)}) ---\n")
        f.write("\n".join(succeeded) or "None")
        f.write(f"\n\n--- Failed ({len(failed)}) ---\n")
        f.write("\n".join(failed) or "None")
        f.write("\n")

    print(f"\nRun log saved to: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    run_update(args.data_dir, args.batch_size, args.limit, args.output)