"""Copy a YOLO dataset while excluding images with empty label files.

The source is left untouched. The destination must not already exist.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    from scripts.audit_dataset import audit_dataset
except ModuleNotFoundError:  # Support direct execution as `python scripts/...py`.
    from audit_dataset import audit_dataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _resolved_paths(value, root: Path) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    return [(Path(v) if Path(v).is_absolute() else root / v).resolve() for v in values]


def prepare(source: Path, destination: Path, expected: int) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    source_yaml = source / "data.yaml"
    if not source_yaml.is_file():
        raise FileNotFoundError(f"Source dataset YAML not found: {source_yaml}")
    if destination.exists():
        raise FileExistsError(f"Destination already exists; refusing to overwrite: {destination}")

    config = yaml.safe_load(source_yaml.read_text())
    if not isinstance(config, dict):
        raise ValueError("data.yaml must contain a mapping")
    root = Path(config.get("path", "."))
    root = (root if root.is_absolute() else source / root).resolve()
    label_pairs: list[tuple[str, Path, Path]] = []
    for split, yaml_key in (("train", "train"), ("valid", "val"), ("test", "test")):
        if yaml_key not in config:
            continue
        for image_dir in _resolved_paths(config[yaml_key], root):
            if not image_dir.is_dir():
                raise FileNotFoundError(f"Image directory not found: {image_dir}")
            if image_dir.name != "images":
                raise ValueError(f"Expected an images directory, got: {image_dir}")
            labels_dir = image_dir.parent / "labels"
            if not labels_dir.is_dir():
                raise FileNotFoundError(f"Label directory not found: {labels_dir}")
            for label in sorted(labels_dir.glob("*.txt")):
                if label.read_text().strip():
                    continue
                image_matches = [image_dir / f"{label.stem}{ext}" for ext in IMAGE_EXTENSIONS]
                images = [path for path in image_matches if path.is_file()]
                if len(images) != 1:
                    raise ValueError(f"Expected exactly one image for empty label {label}, found {len(images)}")
                label_pairs.append((split, label, images[0]))

    if len(label_pairs) != expected:
        raise ValueError(f"Expected {expected} empty-label images, found {len(label_pairs)}; no copy made")

    source_audit = audit_dataset(source_yaml)
    if source_audit["issues"]:
        raise ValueError("Source audit has issues; refusing to filter: " + "; ".join(source_audit["issues"][:10]))

    shutil.copytree(source, destination)
    removed = []
    try:
        for split, label, image in label_pairs:
            dest_label = destination / label.relative_to(source)
            dest_image = destination / image.relative_to(source)
            dest_label.unlink()
            dest_image.unlink()
            removed.append({"split": split, "image": str(image.relative_to(source)), "label": str(label.relative_to(source))})

        destination_yaml = destination / "data.yaml"
        filtered_audit = audit_dataset(destination_yaml)
        if filtered_audit["issues"]:
            raise ValueError("Filtered dataset audit has issues: " + "; ".join(filtered_audit["issues"][:10]))
        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(source),
            "source_dataset_sha256": source_audit["dataset_sha256"],
            "filtered_dataset_sha256": filtered_audit["dataset_sha256"],
            "rule": "Exclude image/label pairs whose label file contains no annotations",
            "removed_count": len(removed),
            "removed": removed,
            "source_audit": source_audit,
            "filtered_audit": filtered_audit,
        }
        (destination / "filter_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    except Exception:
        shutil.rmtree(destination)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--expected", type=int, default=10, help="Expected number of empty-label image pairs")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination, args.expected), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
