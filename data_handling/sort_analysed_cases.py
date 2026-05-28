"""
sort_cases.py

Sorts case folders into output categories based on type detection and results JSON files.

Output folders:
  1_no_json       - No type/results JSON files found
  2_eur_passing   - EUR type, results pass all thresholds
  3_eur_failing   - EUR type, results fail one or more thresholds
  4_nlp           - NLP type (no pass/fail limits defined yet)

A failure report (eur_failure_report.txt) is written to the output root,
showing which rules triggered failures and a per-case breakdown.

Usage:
  python sort_cases.py --input /path/to/case/folders --output /path/to/output
  python sort_cases.py --input /path/to/case/folders --output /path/to/output --move
"""

import argparse
import json
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

# Prefix-based rules applied to all matching fields: (min, max), None = no bound
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

# ── NLP Thresholds ────────────────────────────────────────────────────────────
# No limits defined yet — all NLP cases go into a single folder.

# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIRS = {
    "no_json":     "1_no_json",
    "eur_passing": "2_eur_passing",
    "eur_failing": "3_eur_failing",
    "nlp":         "4_nlp",
}


def load_json(path: Path) -> dict | list | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return None


def check_prefix_thresholds(results: dict, prefix_thresholds: dict) -> list[str]:
    """Return failure messages for any field violating a prefix-matched rule."""
    failures = []
    for field, value in results.items():
        for prefix, (lo, hi) in prefix_thresholds.items():
            if field.startswith(prefix):
                if lo is not None and value < lo:
                    failures.append(f"{field}={value:.4f} < min {lo}")
                if hi is not None and value > hi:
                    failures.append(f"{field}={value:.4f} > max {hi}")
    return failures


def passes_eur(results: dict) -> tuple[bool, list[str]]:
    """Check all EUR rules. Returns (passed, [failure_messages])."""
    failures = []

    # 1. Prefix-based numeric thresholds
    failures += check_prefix_thresholds(results, EUR_PREFIX_THRESHOLDS)

    # 2. Rotation — edge blocks
    for field in EDGE_BLOCK_ROTATION_FIELDS:
        value = results.get(field)
        if value is not None and value > EUR_MAX_ROTATION_EDGE_BLOCK:
            failures.append(f"{field}={value:.4f} > max_rotation_edge {EUR_MAX_ROTATION_EDGE_BLOCK}")

    # 3. Rotation — middle blocks
    for field in MIDDLE_BLOCK_ROTATION_FIELDS:
        value = results.get(field)
        if value is not None and value > EUR_MAX_ROTATION_MIDDLE_BLOCK:
            failures.append(f"{field}={value:.4f} > max_rotation_middle {EUR_MAX_ROTATION_MIDDLE_BLOCK}")

    # 4. Plank angle (rotation_plank_*)
    for field, value in results.items():
        if field.startswith("rotation_plank_"):
            if value > EUR_MAX_PLANK_ANGLE:
                failures.append(f"{field}={value:.4f} > max_plank_angle {EUR_MAX_PLANK_ANGLE}")

    # 5. Missing objects
    if not EUR_LIMIT_MISSING_OBJECT:
        for field, value in results.items():
            if (field.startswith("missing_block_") or field.startswith("missing_plank_")) and value is True:
                failures.append(f"{field} is True (missing not allowed)")

    # 6. Unreadable labels
    unreadable = results.get("unreadable_labels_entire_pallet")
    if unreadable is not None and unreadable > EUR_MAX_UNREADABLE_LABELS:
        failures.append(f"unreadable_labels_entire_pallet={unreadable} > max {EUR_MAX_UNREADABLE_LABELS}")

    # 7. Wood quality (lightness)
    lightness = results.get("lightness_entire_pallet")
    if lightness is not None and lightness < EUR_MIN_WOOD_QUALITY:
        failures.append(f"lightness_entire_pallet={lightness:.4f} < min_wood_quality {EUR_MIN_WOOD_QUALITY}")

    return (len(failures) == 0), failures


def classify_case(case_dir: Path) -> tuple[str, list[str]]:
    """Returns (category_key, [failure_reasons])."""
    type_file    = case_dir / "expected_type_detection.json"
    results_file = case_dir / "results.json"

    if not type_file.exists() or not results_file.exists():
        return "no_json", []

    type_data    = load_json(type_file)
    results_data = load_json(results_file)

    if type_data is None or results_data is None:
        return "no_json", []

    try:
        pallet_type = type_data["result"][0].upper()
    except (KeyError, IndexError, TypeError):
        print(f"  [WARN] Unexpected type format in {type_file}")
        return "no_json", []

    if not isinstance(results_data, dict):
        print(f"  [WARN] results.json is not a dict in {case_dir}")
        return "no_json", []

    if pallet_type == "EUR":
        passed, failures = passes_eur(results_data)
        return ("eur_passing" if passed else "eur_failing"), failures
    elif pallet_type == "NLP":
        return "nlp", []
    else:
        print(f"  [WARN] Unknown pallet type '{pallet_type}' in {case_dir}")
        return "no_json", []


def sort_cases(input_root: Path, output_root: Path, copy: bool = True) -> None:
    for dirname in OUTPUT_DIRS.values():
        (output_root / dirname).mkdir(parents=True, exist_ok=True)

    counts: dict[str, int]            = defaultdict(int)
    failure_tracker: dict[str, int]   = defaultdict(int)   # rule/field -> count
    per_case_failures: dict[str, list[str]] = {}            # case name -> reasons

    case_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    print(f"Found {len(case_dirs)} case folder(s) in {input_root}\n")

    for case_dir in case_dirs:
        category, failures = classify_case(case_dir)
        dest = output_root / OUTPUT_DIRS[category] / case_dir.name

        if failures:
            per_case_failures[case_dir.name] = failures
            for reason in failures:
                # Bucket by the field/rule name (part before '=')
                bucket = reason.split("=")[0].strip()
                failure_tracker[bucket] += 1

        action = shutil.copytree if copy else shutil.move
        try:
            action(str(case_dir), str(dest))
            counts[category] += 1
            if category == "eur_passing":
                tag = "PASS"
            elif category == "eur_failing":
                tag = "FAIL"
            elif category == "nlp":
                tag = "NLP "
            else:
                tag = "    "
            print(f"  [{tag}]  {case_dir.name}")
            for r in failures:
                print(f"          ↳ {r}")
        except FileExistsError:
            print(f"  [SKIP] {dest} already exists — skipping.")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n── Case counts ──────────────────────────────")
    for key, dirname in OUTPUT_DIRS.items():
        print(f"  {dirname}: {counts[key]}")
    print(f"  Total: {sum(counts.values())}")

    if failure_tracker:
        print("\n── EUR failure reasons (by field/rule, most common first) ──")
        for bucket, count in sorted(failure_tracker.items(), key=lambda x: -x[1]):
            print(f"  {count:>4}x  {bucket}")

    # ── Failure report file ───────────────────────────────────────────────────
    if per_case_failures:
        report_path = output_root / "eur_failure_report.txt"
        with open(report_path, "w") as f:
            f.write("EUR FAILURE REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("── Failures by field/rule (most common first) ──\n")
            for bucket, count in sorted(failure_tracker.items(), key=lambda x: -x[1]):
                f.write(f"  {count:>4}x  {bucket}\n")

            f.write(f"\n── Per-case breakdown ({len(per_case_failures)} failing cases) ──\n")
            for case_name, reasons in sorted(per_case_failures.items()):
                f.write(f"\n{case_name}:\n")
                for r in reasons:
                    f.write(f"  - {r}\n")

        print(f"\n  Failure report saved → {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Sort pallet case folders by type and pass/fail.")
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