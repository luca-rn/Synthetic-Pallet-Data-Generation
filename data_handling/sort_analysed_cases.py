"""
sort_cases.py

Sorts case folders into output categories based on type detection and results JSON files.
Before sorting, reads any run_*.txt files (produced by sorting_folders.py) from the input
directory to determine which cases have been analysed:
  - Cases in "Succeeded": sorted normally.
  - Cases in "Failed":    copied/moved to 1_no_json.
  - Cases in neither:     skipped silently.

Output folders:
  1_no_json                     - No type/results JSON files found, or analysis failed
  2_eur_passing                 - EUR type, all thresholds passed
  3_eur_failing/
      single_defect/
          missing_block         - Only defect: a block is missing
          missing_plank         - Only defect: a plank is missing
          rotation_block        - Only defect: block rotation out of bounds
          rotation_plank        - Only defect: plank angle out of bounds
          dislocation_block     - Only defect: block dislocation out of bounds
          volume_block          - Only defect: block volume too low
          area_block            - Only defect: block area too low
          chunk_plank           - Only defect: plank chunk too high
          width_plank           - Only defect: plank width too low
          unreadable_labels     - Only defect: too many unreadable labels
          wood_quality          - Only defect: lightness too low
      multiple_defects          - Two or more distinct defect categories
  4_nlp_passing                 - NLP type, all thresholds passed
  5_nlp_failing/
      single_defect/
          crack                 - Only defect: crack above threshold
          damage                - Only defect: damage above threshold
          hole                  - Only defect: hole above threshold
          dirt                  - Only defect: dirt level above threshold
      multiple_defects          - Two or more distinct defect categories

A summary report (sorting_summary.txt) is written to the output root.

Usage:
  python sort_cases.py --input /path/to/case/folders --output /path/to/output
  python sort_cases.py --input /path/to/case/folders --output /path/to/output --move
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

# ── EUR Thresholds ────────────────────────────────────────────────────────────

EUR_LIMIT_MISSING_OBJECT      = False   # False = any missing block/plank is a failure
EUR_MAX_UNREADABLE_LABELS     = 5       # unreadable_labels_entire_pallet
EUR_MIN_WOOD_QUALITY          = 0.185   # lightness_entire_pallet
EUR_MAX_ROTATION_EDGE_BLOCK   = 8.50    # rotation_block_{front/back/mid}_{left/right}
EUR_MAX_ROTATION_MIDDLE_BLOCK = 20.0    # rotation_block_{front/back/mid}_middle
EUR_MAX_PLANK_ANGLE           = 1.0     # rotation_plank_*

EUR_PREFIX_THRESHOLDS: dict[str, tuple[float | None, float | None]] = {
    "dislocation_block_":   (None, 0.022),
    "volume_block_":        (0.80, None),
    "area_block_":          (0.60, None),
    "chunk_plank_":         (None, 0.030),
    "width_plank_":         (0.80, None),
}

EDGE_BLOCK_ROTATION_FIELDS = {
    "rotation_block_front_left",
    "rotation_block_front_right",
    "rotation_block_back_left",
    "rotation_block_back_right",
    "rotation_block_mid_left",
    "rotation_block_mid_right",
}

MIDDLE_BLOCK_ROTATION_FIELDS = {
    "rotation_block_front_middle",
    "rotation_block_back_middle",
    "rotation_block_mid_middle",
}

EUR_DEFECT_DIRS = [
    "missing_block",
    "missing_plank",
    "rotation_block",
    "rotation_plank",
    "dislocation_block",
    "volume_block",
    "area_block",
    "chunk_plank",
    "width_plank",
    "unreadable_labels",
    "wood_quality",
]

# ── NLP Thresholds ────────────────────────────────────────────────────────────

NLP_MAX_CRACK  = 0.01   # crack_block_* / crack_entire_pallet
NLP_MAX_DAMAGE = 0.01   # damage_block_* / damage_entire_pallet
NLP_MAX_HOLE   = 0.01   # hole_block_* / hole_entire_pallet
NLP_MAX_DIRT   = 0.08   # dirt_level_top_plate

NLP_DEFECT_DIRS = [
    "crack",
    "damage",
    "hole",
    "dirt",
]

# ── Top-level output folders ──────────────────────────────────────────────────

OUTPUT_DIRS = {
    "no_json":      "1_no_json",
    "eur_passing":  "2_eur_passing",
    "eur_failing":  "3_eur_failing",
    "nlp_passing":  "4_nlp_passing",
    "nlp_failing":  "5_nlp_failing",
}


# ── Run-log parsing ───────────────────────────────────────────────────────────

def parse_run_logs(input_root: Path) -> tuple[set[str], set[str]]:
    """
    Scan input_root for run_*.txt files produced by sorting_folders.py.
    Returns (succeeded, failed) as sets of case folder names.
    The union across all log files is used so any case that succeeded in
    any run is considered analysed.
    """
    succeeded: set[str] = set()
    failed: set[str]    = set()

    log_files = sorted(input_root.glob("run_*.txt"))
    if not log_files:
        return succeeded, failed

    print(f"Found {len(log_files)} run log(s): {[f.name for f in log_files]}")

    for log_file in log_files:
        text = log_file.read_text(errors="replace")

        # Split on the section headers written by sorting_folders.py
        succeeded_match = re.search(r"--- Succeeded \(\d+\) ---\n(.*?)(?=\n--- |\Z)", text, re.DOTALL)
        failed_match    = re.search(r"--- Failed \(\d+\) ---\n(.*?)(?=\n--- |\Z)",    text, re.DOTALL)

        if succeeded_match:
            for line in succeeded_match.group(1).splitlines():
                name = line.strip()
                if name and name != "None":
                    succeeded.add(name)

        if failed_match:
            for line in failed_match.group(1).splitlines():
                name = line.strip()
                if name and name != "None":
                    failed.add(name)

    # A case that failed in one run but succeeded in a later one is considered OK
    failed -= succeeded

    print(f"  → {len(succeeded)} succeeded, {len(failed)} failed across all logs\n")
    return succeeded, failed


# ── JSON helpers ──────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict | list | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return None


# ── EUR helpers ───────────────────────────────────────────────────────────────

def check_prefix_thresholds(results: dict, prefix_thresholds: dict) -> list[str]:
    failures = []
    for field, value in results.items():
        for prefix, (lo, hi) in prefix_thresholds.items():
            if field.startswith(prefix):
                if lo is not None and value < lo:
                    failures.append(f"{field}={value:.4f} < min {lo}")
                if hi is not None and value > hi:
                    failures.append(f"{field}={value:.4f} > max {hi}")
    return failures


def eur_field_to_defect(reason: str) -> str:
    field = reason.split("=")[0].strip()
    if field.startswith("missing_block_"):         return "missing_block"
    if field.startswith("missing_plank_"):         return "missing_plank"
    if field in EDGE_BLOCK_ROTATION_FIELDS \
            or field in MIDDLE_BLOCK_ROTATION_FIELDS: return "rotation_block"
    if field.startswith("rotation_plank_"):        return "rotation_plank"
    if field.startswith("dislocation_block_"):     return "dislocation_block"
    if field.startswith("volume_block_"):          return "volume_block"
    if field.startswith("area_block_"):            return "area_block"
    if field.startswith("chunk_plank_"):           return "chunk_plank"
    if field.startswith("width_plank_"):           return "width_plank"
    if field.startswith("unreadable_labels"):      return "unreadable_labels"
    if field.startswith("lightness"):              return "wood_quality"
    return "unknown"


def passes_eur(results: dict) -> tuple[bool, list[str]]:
    failures = []

    failures += check_prefix_thresholds(results, EUR_PREFIX_THRESHOLDS)

    for field in EDGE_BLOCK_ROTATION_FIELDS:
        value = results.get(field)
        if value is not None and value > EUR_MAX_ROTATION_EDGE_BLOCK:
            failures.append(f"{field}={value:.4f} > max_rotation_edge {EUR_MAX_ROTATION_EDGE_BLOCK}")

    for field in MIDDLE_BLOCK_ROTATION_FIELDS:
        value = results.get(field)
        if value is not None and value > EUR_MAX_ROTATION_MIDDLE_BLOCK:
            failures.append(f"{field}={value:.4f} > max_rotation_middle {EUR_MAX_ROTATION_MIDDLE_BLOCK}")

    for field, value in results.items():
        if field.startswith("rotation_plank_") and value > EUR_MAX_PLANK_ANGLE:
            failures.append(f"{field}={value:.4f} > max_plank_angle {EUR_MAX_PLANK_ANGLE}")

    if not EUR_LIMIT_MISSING_OBJECT:
        for field, value in results.items():
            if (field.startswith("missing_block_") or field.startswith("missing_plank_")) and value is True:
                failures.append(f"{field} is True (missing not allowed)")

    unreadable = results.get("unreadable_labels_entire_pallet")
    if unreadable is not None and unreadable > EUR_MAX_UNREADABLE_LABELS:
        failures.append(f"unreadable_labels_entire_pallet={unreadable} > max {EUR_MAX_UNREADABLE_LABELS}")

    lightness = results.get("lightness_entire_pallet")
    if lightness is not None and lightness < EUR_MIN_WOOD_QUALITY:
        failures.append(f"lightness_entire_pallet={lightness:.4f} < min_wood_quality {EUR_MIN_WOOD_QUALITY}")

    return (len(failures) == 0), failures


# ── NLP helpers ───────────────────────────────────────────────────────────────

def nlp_field_to_defect(reason: str) -> str:
    field = reason.split("=")[0].strip()
    if field.startswith("crack"):  return "crack"
    if field.startswith("damage"): return "damage"
    if field.startswith("hole"):   return "hole"
    if field.startswith("dirt"):   return "dirt"
    return "unknown"


def passes_nlp(results: dict) -> tuple[bool, list[str]]:
    failures = []
    for field, value in results.items():
        if field.startswith("crack"):
            if value > NLP_MAX_CRACK:
                failures.append(f"{field}={value:.6f} > max_crack {NLP_MAX_CRACK}")
        elif field.startswith("damage"):
            if value > NLP_MAX_DAMAGE:
                failures.append(f"{field}={value:.6f} > max_damage {NLP_MAX_DAMAGE}")
        elif field.startswith("hole"):
            if value > NLP_MAX_HOLE:
                failures.append(f"{field}={value:.6f} > max_hole {NLP_MAX_HOLE}")
        elif field.startswith("dirt"):
            if value > NLP_MAX_DIRT:
                failures.append(f"{field}={value:.6f} > max_dirt {NLP_MAX_DIRT}")
    return (len(failures) == 0), failures


# ── Classification ────────────────────────────────────────────────────────────

def classify_case(case_dir: Path) -> tuple[str, list[str], list[str]]:
    """Returns (category_key, [failure_reasons], [defect_categories])."""
    type_file    = case_dir / "expected_type_detection.json"
    results_file = case_dir / "results.json"

    if not type_file.exists() or not results_file.exists():
        return "no_json", [], []

    type_data    = load_json(type_file)
    results_data = load_json(results_file)

    if type_data is None or results_data is None:
        return "no_json", [], []

    try:
        pallet_type = type_data["result"][0].upper()
    except (KeyError, IndexError, TypeError):
        print(f"  [WARN] Unexpected type format in {type_file}")
        return "no_json", [], []

    if not isinstance(results_data, dict):
        print(f"  [WARN] results.json is not a dict in {case_dir}")
        return "no_json", [], []

    if pallet_type == "EUR":
        passed, failures = passes_eur(results_data)
        if passed:
            return "eur_passing", [], []
        defect_categories = sorted(set(eur_field_to_defect(r) for r in failures))
        return "eur_failing", failures, defect_categories

    elif pallet_type == "NLP":
        passed, failures = passes_nlp(results_data)
        if passed:
            return "nlp_passing", [], []
        defect_categories = sorted(set(nlp_field_to_defect(r) for r in failures))
        return "nlp_failing", failures, defect_categories

    else:
        print(f"  [WARN] Unknown pallet type '{pallet_type}' in {case_dir}")
        return "no_json", [], []


# ── Directory setup ───────────────────────────────────────────────────────────

def ensure_dirs(output_root: Path) -> None:
    for dirname in OUTPUT_DIRS.values():
        (output_root / dirname).mkdir(parents=True, exist_ok=True)

    for failing_key, defect_dirs in [("eur_failing", EUR_DEFECT_DIRS),
                                      ("nlp_failing", NLP_DEFECT_DIRS)]:
        failing_root = output_root / OUTPUT_DIRS[failing_key]
        (failing_root / "multiple_defects").mkdir(parents=True, exist_ok=True)
        for defect in defect_dirs:
            (failing_root / "single_defect" / defect).mkdir(parents=True, exist_ok=True)


def copy_or_move(src: Path, dest: Path, copy: bool) -> None:
    action = shutil.copytree if copy else shutil.move
    action(str(src), str(dest))


# ── Main sort logic ───────────────────────────────────────────────────────────

def sort_cases(input_root: Path, output_root: Path, copy: bool = True) -> None:
    ensure_dirs(output_root)

    # ── Load run logs ─────────────────────────────────────────────────────────
    logs_present = any(input_root.glob("run_*.txt"))
    succeeded, run_failed = parse_run_logs(input_root)

    counts: dict[str, int]             = defaultdict(int)
    eur_defect_counts: dict[str, int]  = defaultdict(int)
    nlp_defect_counts: dict[str, int]  = defaultdict(int)
    eur_field_counts: dict[str, int]   = defaultdict(int)
    nlp_field_counts: dict[str, int]   = defaultdict(int)
    per_case_failures: dict[str, dict] = {}
    skipped_cases: list[str]           = []
    log_failed_cases: list[str]        = []

    case_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    print(f"Found {len(case_dirs)} case folder(s) in {input_root}\n")

    for case_dir in case_dirs:

        # ── Run-log filtering ─────────────────────────────────────────────────
        if logs_present:
            if case_dir.name in run_failed:
                # Analysis failed — send straight to no_json
                dest = output_root / OUTPUT_DIRS["no_json"] / case_dir.name
                try:
                    copy_or_move(case_dir, dest, copy)
                    counts["no_json"] += 1
                    log_failed_cases.append(case_dir.name)
                    print(f"  [LOG FAIL]  {case_dir.name}  →  {OUTPUT_DIRS['no_json']}  (analysis failed)")
                except FileExistsError:
                    print(f"  [SKIP] {dest} already exists — skipping.")
                continue

            if case_dir.name not in succeeded:
                # Not in any log — skip silently
                skipped_cases.append(case_dir.name)
                continue

        # ── Normal classification ─────────────────────────────────────────────
        category, failures, defect_categories = classify_case(case_dir)

        if category in ("eur_failing", "nlp_failing"):
            failing_root = output_root / OUTPUT_DIRS[category]
            if len(defect_categories) > 1:
                dest = failing_root / "multiple_defects" / case_dir.name
                dest_label = f"{OUTPUT_DIRS[category]}/multiple_defects"
            elif len(defect_categories) == 1:
                dest = failing_root / "single_defect" / defect_categories[0] / case_dir.name
                dest_label = f"{OUTPUT_DIRS[category]}/single_defect/{defect_categories[0]}"
            else:
                dest = failing_root / "multiple_defects" / case_dir.name
                dest_label = f"{OUTPUT_DIRS[category]}/multiple_defects"
        else:
            dest = output_root / OUTPUT_DIRS[category] / case_dir.name
            dest_label = OUTPUT_DIRS[category]

        if failures:
            per_case_failures[case_dir.name] = {
                "type": "EUR" if category.startswith("eur") else "NLP",
                "defect_categories": defect_categories,
                "failures": failures,
                "destination": dest_label,
            }
            field_counts  = eur_field_counts  if category.startswith("eur") else nlp_field_counts
            defect_counts = eur_defect_counts if category.startswith("eur") else nlp_defect_counts
            for cat in defect_categories:
                defect_counts[cat] += 1
            for reason in failures:
                field_counts[reason.split("=")[0].strip()] += 1

        try:
            copy_or_move(case_dir, dest, copy)
            counts[category] += 1

            tag_map = {
                "eur_passing": "EUR PASS",
                "eur_failing": "EUR FAIL",
                "nlp_passing": "NLP PASS",
                "nlp_failing": "NLP FAIL",
                "no_json":     "        ",
            }
            tag = tag_map.get(category, "        ")
            print(f"  [{tag}]  {case_dir.name}  →  {dest_label}")
            for r in failures:
                print(f"             ↳ {r}")
        except FileExistsError:
            print(f"  [SKIP] {dest} already exists — skipping.")

    # ── Console summary ───────────────────────────────────────────────────────
    total = sum(counts.values())
    print("\n── Case counts ──────────────────────────────")
    for key, dirname in OUTPUT_DIRS.items():
        print(f"  {dirname}: {counts[key]}")
    print(f"  Total processed : {total}")
    if skipped_cases:
        print(f"  Skipped (not in any run log) : {len(skipped_cases)}")

    if eur_defect_counts:
        print("\n── EUR failing cases by defect category ──")
        for cat, n in sorted(eur_defect_counts.items(), key=lambda x: -x[1]):
            print(f"  {n:>4}x  {cat}")

    if nlp_defect_counts:
        print("\n── NLP failing cases by defect category ──")
        for cat, n in sorted(nlp_defect_counts.items(), key=lambda x: -x[1]):
            print(f"  {n:>4}x  {cat}")

    # ── Write TXT summary ─────────────────────────────────────────────────────
    report_path = output_root / "sorting_summary.txt"
    with open(report_path, "w") as f:

        f.write("SORTING SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        # ── Run-log status ────────────────────────────────────────────────────
        if logs_present:
            f.write("── Run log filtering ───────────────────────────────────────\n")
            f.write(f"  Logs found          : {sum(1 for _ in input_root.glob('run_*.txt'))}\n")
            f.write(f"  Succeeded (analysed): {len(succeeded)}\n")
            f.write(f"  Failed in logs      : {len(run_failed)}\n")
            f.write(f"  Skipped (not in any log): {len(skipped_cases)}\n")
            if log_failed_cases:
                f.write(f"\n  Cases sent to no_json due to analysis failure:\n")
                for name in sorted(log_failed_cases):
                    f.write(f"    - {name}\n")
            f.write("\n")
        else:
            f.write("── Run log filtering ───────────────────────────────────────\n")
            f.write("  No run_*.txt files found — all cases processed without filtering.\n\n")

        # ── Thresholds ────────────────────────────────────────────────────────
        f.write("── EUR thresholds ──────────────────────────────────────────\n")
        f.write(f"  missing_block / missing_plank : {'not allowed' if not EUR_LIMIT_MISSING_OBJECT else 'allowed'}\n")
        f.write(f"  unreadable_labels             : max {EUR_MAX_UNREADABLE_LABELS}\n")
        f.write(f"  wood_quality (lightness)      : min {EUR_MIN_WOOD_QUALITY}\n")
        f.write(f"  rotation edge blocks          : max {EUR_MAX_ROTATION_EDGE_BLOCK}\n")
        f.write(f"  rotation middle blocks        : max {EUR_MAX_ROTATION_MIDDLE_BLOCK}\n")
        f.write(f"  rotation planks               : max {EUR_MAX_PLANK_ANGLE}\n")
        for prefix, (lo, hi) in EUR_PREFIX_THRESHOLDS.items():
            parts = []
            if lo is not None: parts.append(f"min {lo}")
            if hi is not None: parts.append(f"max {hi}")
            f.write(f"  {prefix:<30}: {', '.join(parts)}\n")

        f.write("\n── NLP thresholds ──────────────────────────────────────────\n")
        f.write(f"  crack  (crack_block_* / crack_entire_pallet)   : max {NLP_MAX_CRACK}\n")
        f.write(f"  damage (damage_block_* / damage_entire_pallet) : max {NLP_MAX_DAMAGE}\n")
        f.write(f"  hole   (hole_block_* / hole_entire_pallet)     : max {NLP_MAX_HOLE}\n")
        f.write(f"  dirt   (dirt_level_top_plate)                  : max {NLP_MAX_DIRT}\n")

        # ── Case counts ───────────────────────────────────────────────────────
        f.write("\n── Case counts ─────────────────────────────────────────────\n")
        for key, dirname in OUTPUT_DIRS.items():
            f.write(f"  {dirname:<30}: {counts[key]:>4}\n")
        f.write(f"  {'Total processed':<30}: {total:>4}\n")
        if skipped_cases:
            f.write(f"  {'Skipped (not analysed)':<30}: {len(skipped_cases):>4}\n")

        eur_failing_cases = {k: v for k, v in per_case_failures.items() if v["type"] == "EUR"}
        nlp_failing_cases = {k: v for k, v in per_case_failures.items() if v["type"] == "NLP"}

        for label, failing_cases in [("EUR", eur_failing_cases), ("NLP", nlp_failing_cases)]:
            if not failing_cases:
                continue
            n_single = sum(1 for v in failing_cases.values() if len(v["defect_categories"]) == 1)
            n_multi  = sum(1 for v in failing_cases.values() if len(v["defect_categories"]) > 1)
            f.write(f"\n  {label} failing breakdown:\n")
            f.write(f"    Single-defect cases : {n_single}\n")
            f.write(f"    Multi-defect cases  : {n_multi}\n")

        # ── Defect category breakdowns ────────────────────────────────────────
        for label, defect_counts in [("EUR", eur_defect_counts), ("NLP", nlp_defect_counts)]:
            if not defect_counts:
                continue
            f.write(f"\n── {label} failing cases by defect category ─────────────────\n")
            f.write("  (a multi-defect case is counted once per category it triggered)\n")
            for cat, n in sorted(defect_counts.items(), key=lambda x: -x[1]):
                bar = "█" * n
                f.write(f"  {cat:<22}: {n:>4}  {bar}\n")

        # ── Field-level failure counts ────────────────────────────────────────
        for label, field_counts in [("EUR", eur_field_counts), ("NLP", nlp_field_counts)]:
            if not field_counts:
                continue
            f.write(f"\n── {label} individual field failures (most common first) ───────\n")
            for field, n in sorted(field_counts.items(), key=lambda x: -x[1]):
                f.write(f"  {n:>4}x  {field}\n")

        # ── Per-case breakdown ────────────────────────────────────────────────
        for label, failing_cases in [("EUR", eur_failing_cases), ("NLP", nlp_failing_cases)]:
            if not failing_cases:
                continue
            f.write(f"\n── {label} per-case breakdown ({len(failing_cases)} failing cases) ───────────\n")
            for case_name, info in sorted(failing_cases.items()):
                cats = ", ".join(info["defect_categories"])
                f.write(f"\n  {case_name}\n")
                f.write(f"    Destination : {info['destination']}\n")
                f.write(f"    Categories  : {cats}\n")
                f.write(f"    Failures    :\n")
                for r in info["failures"]:
                    f.write(f"      - {r}\n")

    print(f"\n  Summary report saved → {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Sort pallet case folders by type and defect category.")
    parser.add_argument("--input",  required=True, help="Root folder containing case sub-folders")
    parser.add_argument("--output", required=True, help="Root folder where sorted sub-folders will be created")
    parser.add_argument("--move",   action="store_true", help="Move folders instead of copying (default: copy)")
    args = parser.parse_args()

    input_root  = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    if not input_root.is_dir():
        raise SystemExit(f"Input path does not exist or is not a directory: {input_root}")

    sort_cases(input_root, output_root, copy=not args.move)


if __name__ == "__main__":
    main()