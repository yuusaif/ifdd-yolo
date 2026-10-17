"""Read-only audit of an Ultralytics detection dataset export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _split_paths(value, root: Path) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    paths = []
    for item in values:
        path = Path(item)
        paths.append((path if path.is_absolute() else root / path).resolve())
    return paths


def audit_dataset(yaml_path: Path) -> dict:
    yaml_path = yaml_path.resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Dataset YAML must contain a mapping")
    names = data.get("names")
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(names))]
    if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
        raise ValueError("Dataset YAML needs a complete names list or indexed names mapping")
    declared_root = Path(data.get("path", "."))
    root = (declared_root if declared_root.is_absolute() else yaml_path.parent / declared_root).resolve()
    report = {"yaml": str(yaml_path), "classes": names, "splits": {}, "issues": []}
    digest = hashlib.sha256()
    image_hash_splits: dict[str, str] = {}
    image_paths_seen: dict[str, str] = {}

    for split in ("train", "val", "test"):
        if split not in data:
            if split in ("train", "val"):
                report["issues"].append(f"Missing required {split} split in data.yaml")
            continue
        image_files: list[Path] = []
        for source in _split_paths(data[split], root):
            if source.is_dir():
                image_files.extend(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
            elif source.is_file() and source.suffix.lower() == ".txt":
                image_files.extend((source.parent / line.strip()).resolve() for line in source.read_text().splitlines() if line.strip())
            else:
                report["issues"].append(f"Missing {split} image directory/list: {source}")
        image_files = sorted(set(image_files))
        class_counts: Counter = Counter()
        image_sizes: Counter = Counter()
        image_formats: Counter = Counter()
        image_modes: Counter = Counter()
        missing_labels = 0
        empty_labels = 0
        for image_path in image_files:
            if not image_path.is_file():
                report["issues"].append(f"Missing image: {image_path}")
                continue
            try:
                with Image.open(image_path) as image:
                    image_sizes[f"{image.width}x{image.height}"] += 1
                    image_formats[str(image.format)] += 1
                    image_modes[image.mode] += 1
                    image.verify()
            except (UnidentifiedImageError, OSError) as exc:
                report["issues"].append(f"Unreadable image {image_path}: {exc}")
                continue
            key = str(image_path.resolve())
            if key in image_paths_seen and image_paths_seen[key] != split:
                report["issues"].append(f"Same image path in {image_paths_seen[key]} and {split}: {image_path}")
            image_paths_seen[key] = split
            image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if image_hash in image_hash_splits and image_hash_splits[image_hash] != split:
                report["issues"].append(f"Identical image bytes across {image_hash_splits[image_hash]} and {split}: {image_path}")
            image_hash_splits[image_hash] = split
            digest.update(f"{split}/{os.path.relpath(image_path, root)}:{image_hash}\n".encode())
            parts = list(image_path.parts)
            if "images" not in parts:
                report["issues"].append(f"Image path has no images/ component: {image_path}")
                continue
            parts[parts.index("images")] = "labels"
            label_path = Path(*parts).with_suffix(".txt")
            if not label_path.is_file():
                missing_labels += 1
                continue
            raw = label_path.read_bytes()
            if not raw.strip():
                empty_labels += 1
            digest.update(f"{split}/{os.path.relpath(label_path, root)}:{hashlib.sha256(raw).hexdigest()}\n".encode())
            for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
                fields = line.split()
                try:
                    if len(fields) != 5:
                        raise ValueError("expected five YOLO detection fields")
                    class_id = int(fields[0])
                    x, y, w, h = (float(v) for v in fields[1:])
                    if not 0 <= class_id < len(names):
                        raise ValueError("class ID outside names")
                    if w <= 0 or h <= 0 or min(x - w / 2, y - h / 2) < -1e-6 or max(x + w / 2, y + h / 2) > 1 + 1e-6:
                        raise ValueError("box is empty or outside normalized image")
                    class_counts[names[class_id]] += 1
                except ValueError as exc:
                    report["issues"].append(f"{label_path}:{line_no}: {exc}")
        report["splits"][split] = {
            "images": len(image_files),
            "instances": sum(class_counts.values()),
            "by_class": dict(class_counts),
            "image_sizes": dict(image_sizes),
            "image_formats": dict(image_formats),
            "image_modes": dict(image_modes),
            "images_without_label_file": missing_labels,
            "images_with_empty_label_file": empty_labels,
            "zero_instance_images": missing_labels + empty_labels,
        }
    report["dataset_sha256"] = digest.hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_yaml", type=Path)
    args = parser.parse_args()
    try:
        report = audit_dataset(args.data_yaml)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 1 if report["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

