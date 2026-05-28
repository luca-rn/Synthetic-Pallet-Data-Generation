#!/usr/bin/env python3
"""
move_small_folders.py

Scans a source folder and moves every direct subfolder that contains
4 or fewer items (files + subfolders combined) to a destination folder.

Usage:
    python move_small_folders.py <source_folder> <destination_folder> [--dry-run] [--threshold N]

Examples:
    python move_small_folders.py ~/Downloads/projects ~/Archive
    python move_small_folders.py ~/Downloads/projects ~/Archive --dry-run
    python move_small_folders.py ~/Downloads/projects ~/Archive --threshold 6
"""

import argparse
import os
import shutil
import sys

DEST : str = "D:\ThesisData\single_cam"
SOURCE : str = "D:\ThesisData\sort_agder"


def count_items(folder_path: str) -> int:
    """Return the number of direct children (files + subdirs) in a folder."""
    try:
        return len(os.listdir(folder_path))
    except PermissionError:
        print(f"  [WARNING] Permission denied reading: {folder_path}")
        return None


def move_small_subfolders(source: str, destination: str, threshold: int, dry_run: bool) -> None:
    source = os.path.abspath(source)
    destination = os.path.abspath(destination)

    # Validate source
    if not os.path.isdir(source):
        print(f"[ERROR] Source folder does not exist or is not a directory:\n  {source}")
        sys.exit(1)

    # Prevent moving source into itself
    if destination.startswith(source + os.sep) or destination == source:
        print("[ERROR] Destination cannot be inside the source folder.")
        sys.exit(1)

    # Create destination if needed
    if not dry_run:
        os.makedirs(destination, exist_ok=True)
    else:
        print(f"[DRY RUN] Would ensure destination exists: {destination}\n")

    print(f"Source      : {source}")
    print(f"Destination : {destination}")
    print(f"Threshold   : {threshold} items or fewer")
    print(f"Mode        : {'DRY RUN (nothing will be moved)' if dry_run else 'LIVE'}")
    print("-" * 60)

    moved = 0
    skipped = 0
    errors = 0

    entries = sorted(os.scandir(source), key=lambda e: e.name)
    subfolders = [e for e in entries if e.is_dir()]

    if not subfolders:
        print("No subfolders found in source.")
        return

    for entry in subfolders:
        item_count = count_items(entry.path)

        if item_count is None:
            errors += 1
            continue

        if item_count <= threshold:
            dest_path = os.path.join(destination, entry.name)
            status = "MOVE" if not dry_run else "WOULD MOVE"

            # Handle name collision
            if os.path.exists(dest_path):
                base = entry.name
                dest_path = os.path.join(destination, f"{base}_conflict")
                print(f"  [{status}] {entry.name}/ ({item_count} items)  →  {dest_path}  [name collision, renamed]")
            else:
                print(f"  [{status}] {entry.name}/ ({item_count} items)")

            if not dry_run:
                try:
                    shutil.move(entry.path, dest_path)
                    moved += 1
                except Exception as exc:
                    print(f"    [ERROR] Could not move {entry.name}: {exc}")
                    errors += 1
            else:
                moved += 1
        else:
            print(f"  [SKIP]   {entry.name}/ ({item_count} items)")
            skipped += 1

    print("-" * 60)
    action = "Would move" if dry_run else "Moved"
    print(f"{action}: {moved}  |  Skipped: {skipped}  |  Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(
        description="Move subfolders with N or fewer items to a destination folder."
    )
    parser.add_argument("source", type=str, default = SOURCE, help="Path to the folder to scan")
    parser.add_argument("destination", type = str,default = DEST, help="Path to move matching subfolders into")
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=4,
        help="Maximum number of items a subfolder may contain to be moved (default: 4)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview what would be moved without actually moving anything",
    )

    args = parser.parse_args()

    if args.threshold < 0:
        print("[ERROR] Threshold must be 0 or greater.")
        sys.exit(1)

    move_small_subfolders(args.source, args.destination, args.threshold, args.dry_run)


if __name__ == "__main__":
    main()