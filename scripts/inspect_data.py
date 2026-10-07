#!/usr/bin/env python3
"""
Report what a downloaded dataset actually contains, before writing any converter.

Dataset archives lie in their descriptions. Run this first and read the output,
then write the conversion to YOLO format against what is really there.

    python scripts/inspect_data.py --root data/raw
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ANNOT_EXT = {".txt", ".xml", ".json", ".csv", ".yaml", ".yml"}


def tree(root: Path, max_depth=3, max_entries=12):
    """Print the directory structure without drowning in file lists."""
    def walk(d: Path, prefix="", depth=0):
        if depth > max_depth:
            return
        try:
            entries = sorted(d.iterdir())
        except PermissionError:
            return
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        for sub in dirs[:max_entries]:
            n = sum(1 for _ in sub.rglob("*") if _.is_file())
            print(f"{prefix}{sub.name}/  ({n} files)")
            walk(sub, prefix + "  ", depth + 1)
        if len(dirs) > max_entries:
            print(f"{prefix}... and {len(dirs) - max_entries} more directories")
        for f in files[:4]:
            print(f"{prefix}{f.name}")
        if len(files) > 4:
            print(f"{prefix}... and {len(files) - 4} more files")
    print(f"{root}/")
    walk(root, "  ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()
    root = args.root

    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    print("=" * 70)
    print("DIRECTORY STRUCTURE")
    print("=" * 70)
    tree(root)

    files = [p for p in root.rglob("*") if p.is_file()]
    ext_counts = Counter(p.suffix.lower() for p in files)

    print("\n" + "=" * 70)
    print("FILE TYPES")
    print("=" * 70)
    for ext, n in ext_counts.most_common(20):
        print(f"  {n:7d}  {ext or '(no extension)'}")

    images = [p for p in files if p.suffix.lower() in IMAGE_EXT]
    annots = [p for p in files if p.suffix.lower() in ANNOT_EXT]
    print(f"\n  {len(images)} images, {len(annots)} possible annotation files")

    # --- what do the annotations look like? ---
    print("\n" + "=" * 70)
    print("ANNOTATION SAMPLES")
    print("=" * 70)
    by_ext = defaultdict(list)
    for p in annots:
        by_ext[p.suffix.lower()].append(p)

    for ext, paths in by_ext.items():
        print(f"\n--- {ext}  ({len(paths)} files)  e.g. {paths[0].relative_to(root)}")
        try:
            text = paths[0].read_text(errors="replace")
        except Exception as exc:
            print(f"    unreadable: {exc}")
            continue
        if ext == ".json":
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    print(f"    top-level keys: {list(obj)[:12]}")
                    # COCO detection
                    if {"images", "annotations", "categories"} <= set(obj):
                        print("    -> looks like COCO format")
                        print(f"       {len(obj['images'])} images, "
                              f"{len(obj['annotations'])} annotations")
                        print(f"       categories: "
                              f"{[c.get('name') for c in obj['categories']]}")
                        continue
                else:
                    print(f"    top-level: list of {len(obj)}")
            except json.JSONDecodeError:
                pass
        for line in text.split("\n")[:6]:
            print(f"    {line[:110]}")

    # --- guess the format ---
    print("\n" + "=" * 70)
    print("LIKELY FORMAT")
    print("=" * 70)
    guesses = []
    if any(p.name == "data.yaml" for p in files):
        guesses.append("YOLO (data.yaml present) - usable directly")
    if ext_counts.get(".xml", 0) > 10:
        guesses.append("Pascal VOC (many .xml) - needs conversion")
    if any("annotations" in p.name.lower() and p.suffix == ".json" for p in files):
        guesses.append("COCO (annotations .json) - needs conversion")
    txt_near_images = sum(
        1 for p in images[:200] if p.with_suffix(".txt").exists()
    )
    if txt_near_images > 20:
        guesses.append(f"YOLO .txt beside images ({txt_near_images}/200 sampled) "
                       "- may just need restructuring into images/ labels/")
    if ext_counts.get(".csv", 0):
        guesses.append("CSV annotations - inspect columns above")
    if not guesses:
        guesses.append("UNCLEAR - images may be organised by class folder "
                       "(classification, not detection). Check the tree above.")
    for g in guesses:
        print(f"  - {g}")

    # --- class names from folder structure, if any ---
    subdirs = [d for d in root.rglob("*") if d.is_dir()]
    classlike = [d for d in subdirs
                 if sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXT) > 20]
    if classlike:
        print("\n  directories holding >20 images (possible class folders):")
        for d in sorted(classlike)[:25]:
            n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXT)
            print(f"    {n:6d}  {d.relative_to(root)}")

    print("\nNext: paste this output and we write the converter against it.")


if __name__ == "__main__":
    main()
