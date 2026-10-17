"""Derive axis-aligned YOLO detection boxes from YOLO polygon labels.

The source export is copied unchanged into a new destination first. Polygon
vertices are replaced by their enclosing normalized bounding boxes; existing
five-field detection rows are retained. The destination is refused if it
already exists so an audited dataset cannot be overwritten accidentally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(f"{path.relative_to(root)}:{_sha256_file(path)}\n".encode())
    return digest.hexdigest()


def _format(value: float) -> str:
    return f"{value:.9f}".rstrip("0").rstrip(".")


def _convert_label(source: Path, destination: Path, class_count: int, stats: Counter) -> None:
    converted: list[str] = []
    for line_number, raw_line in enumerate(source.read_text().splitlines(), 1):
        fields = raw_line.split()
        if not fields:
            continue
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"{source}:{line_number}: non-numeric label field") from exc
        if not 0 <= class_id < class_count:
            raise ValueError(f"{source}:{line_number}: class ID outside names")

        if len(fields) == 5:
            x, y, width, height = values
            if width <= 0 or height <= 0:
                raise ValueError(f"{source}:{line_number}: invalid detection box")
            if min(x - width / 2, y - height / 2) < -1e-6 or max(x + width / 2, y + height / 2) > 1 + 1e-6:
                raise ValueError(f"{source}:{line_number}: detection box outside normalized image")
            converted.append(" ".join(fields))
            stats["existing_detection_rows"] += 1
            continue

        if len(values) < 6 or len(values) % 2:
            raise ValueError(f"{source}:{line_number}: expected a five-field box or polygon coordinates")
        xs = values[0::2]
        ys = values[1::2]
        if min(xs + ys) < -1e-6 or max(xs + ys) > 1 + 1e-6:
            raise ValueError(f"{source}:{line_number}: polygon coordinate outside normalized image")
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        width = x_max - x_min
        height = y_max - y_min
        if width <= 0 or height <= 0:
            raise ValueError(f"{source}:{line_number}: degenerate polygon")
        converted.append(
            " ".join(
                (
                    str(class_id),
                    _format((x_min + x_max) / 2),
                    _format((y_min + y_max) / 2),
                    _format(width),
                    _format(height),
                )
            )
        )
        stats["converted_polygon_rows"] += 1

    destination.write_text("\n".join(converted) + ("\n" if converted else ""))
    if not converted:
        stats["empty_label_files"] += 1


def convert(source_root: Path, destination_root: Path) -> dict:
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    source_yaml = source_root / "data.yaml"
    if not source_yaml.is_file():
        raise FileNotFoundError(f"Source data.yaml not found: {source_yaml}")
    if destination_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {destination_root}")

    data = yaml.safe_load(source_yaml.read_text())
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(names))]
    if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
        raise ValueError("Source data.yaml needs a complete names list")

    source_fingerprint = _export_fingerprint(source_root)
    shutil.copytree(source_root, destination_root)
    stats_by_split: dict[str, Counter] = defaultdict(Counter)
    for split in ("train", "valid", "test"):
        source_labels = source_root / split / "labels"
        destination_labels = destination_root / split / "labels"
        for source_label in sorted(source_labels.glob("*.txt")):
            stats = stats_by_split[split]
            _convert_label(source_label, destination_labels / source_label.name, len(names), stats)

    data["train"] = "train/images"
    data["val"] = "valid/images"
    data["test"] = "test/images"
    data["conversion"] = {
        "source_annotation_format": "YOLO polygon/segmentation rows",
        "destination_annotation_format": "YOLO axis-aligned detection boxes",
        "method": "enclosing normalized min/max box for each polygon",
        "existing_five_field_rows": "preserved",
        "empty_label_files": "preserved",
    }
    (destination_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    derived_fingerprint = _export_fingerprint(destination_root)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "source_roboflow_version": data.get("roboflow", {}).get("version"),
        "source_export_sha256": source_fingerprint,
        "derived_export_sha256": derived_fingerprint,
        "classes": names,
        "conversion": data["conversion"],
        "splits": {split: dict(stats) for split, stats in stats_by_split.items()},
    }
    (destination_root / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    manifest = convert(args.source, args.destination)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
