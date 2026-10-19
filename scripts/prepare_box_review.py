"""Prepare a reproducible visual sample of source polygons and converted boxes."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path.home() / "Downloads" / "Manufacturing Industrial Robot.v4i.yolov8"
DEFAULT_DERIVED = ROOT / "data" / "ifdd_930_v4_boxes"
CLASS_NAMES = ["Clip", "Rivet", "Screw"]
COLORS = [(0, 255, 90), (255, 210, 0), (60, 180, 255)]


def parse_row(row: str, path: Path, line_no: int) -> tuple[int, list[float], bool]:
    fields = row.split()
    if len(fields) < 5:
        raise ValueError(f"Malformed label at {path}:{line_no}")
    cls = int(fields[0])
    values = list(map(float, fields[1:]))
    if len(values) == 4:
        xc, yc, w, h = values
        return cls, [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], False
    if len(values) < 6 or len(values) % 2:
        raise ValueError(f"Unexpected polygon/box row at {path}:{line_no}")
    xs, ys = values[0::2], values[1::2]
    return cls, [min(xs), min(ys), max(xs), max(ys)], True


def angle_score(row: str) -> float:
    fields = row.split()
    values = list(map(float, fields[1:]))
    if len(values) < 6:
        return 0.0
    points = list(zip(values[0::2], values[1::2]))
    edges = []
    for a, b in zip(points, points[1:] + points[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        degrees = abs(math.degrees(math.atan2(dy, dx))) % 90
        deviation = min(degrees, 90 - degrees)
        edges.append((length, deviation))
    return max(edges)[1] if edges else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--derived", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DERIVED / "box_spotcheck")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    try:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite review sample: {output}")
        source_root, derived_root = args.source.resolve(), args.derived.resolve()
        candidates = []
        for split in ("train", "valid", "test"):
            for image_path in sorted((source_root / split / "images").glob("*")):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                source_label = source_root / split / "labels" / f"{image_path.stem}.txt"
                converted_label = derived_root / split / "labels" / f"{image_path.stem}.txt"
                if not source_label.is_file() or not converted_label.is_file():
                    continue
                source_rows = source_label.read_text().splitlines()
                converted_rows = converted_label.read_text().splitlines()
                if len(source_rows) != len(converted_rows):
                    raise ValueError(f"Label row count changed for {image_path.name}")
                annotations = []
                for line_no, (source_row, converted_row) in enumerate(zip(source_rows, converted_rows), 1):
                    cls, box, polygon = parse_row(source_row, source_label, line_no)
                    converted_cls, converted_box, converted_polygon = parse_row(converted_row, converted_label, line_no)
                    if cls != converted_cls or converted_polygon:
                        raise ValueError(f"Class/order mismatch during conversion: {converted_label}:{line_no}")
                    if max(abs(a - b) for a, b in zip(box, converted_box)) > 1e-5:
                        raise ValueError(f"Converted box differs from source polygon bounds: {converted_label}:{line_no}")
                    if polygon:
                        annotations.append({
                            "class_id": cls, "class": CLASS_NAMES[cls], "polygon_row": source_row,
                            "box_xyxy_normalized": box,
                            "angled_score_degrees": angle_score(source_row) if cls == 0 else 0.0,
                        })
                if annotations:
                    candidates.append({
                        "image": image_path, "split": split,
                        "annotations": annotations,
                        "has_clip": any(x["class_id"] == 0 for x in annotations),
                        "clip_angle": max((x["angled_score_degrees"] for x in annotations), default=0.0),
                    })
        clips = [x for x in candidates if x["has_clip"] and x["clip_angle"] >= 10]
        clips.sort(key=lambda x: (-x["clip_angle"], x["image"].name))
        selected = clips[:10]
        selected_paths = {x["image"] for x in selected}
        remaining = [x for x in candidates if x["image"] not in selected_paths]
        random.Random(args.seed).shuffle(remaining)
        selected.extend(remaining[:30 - len(selected)])
        if len(selected) != 30:
            raise ValueError(f"Only found {len(selected)} eligible polygon images; expected 30")
        output.mkdir(parents=True)
        sheet_w, sheet_h, cols = 420, 455, 5
        rows = math.ceil(len(selected) / cols)
        sheet = Image.new("RGB", (cols * sheet_w, rows * sheet_h), "white")
        draw = ImageDraw.Draw(sheet)
        manifest = []
        for index, sample in enumerate(selected):
            with Image.open(sample["image"]) as source_image:
                image = source_image.convert("RGB")
            draw_tile = ImageDraw.Draw(image)
            width, height = image.size
            for annotation in sample["annotations"]:
                color = COLORS[annotation["class_id"]]
                fields = annotation["polygon_row"].split()
                coords = list(map(float, fields[1:]))
                points = [(coords[i] * width, coords[i + 1] * height) for i in range(0, len(coords), 2)]
                draw_tile.polygon(points, outline=color, width=3)
                x1, y1, x2, y2 = annotation["box_xyxy_normalized"]
                draw_tile.rectangle((x1 * width, y1 * height, x2 * width, y2 * height), outline=(255, 40, 40), width=3)
            image.thumbnail((sheet_w - 16, sheet_h - 72))
            x = (index % cols) * sheet_w + (sheet_w - image.width) // 2
            y = (index // cols) * sheet_h + 48
            sheet.paste(image, (x, y))
            title = f"{index + 1:02d} {sample['split']} | {sample['image'].name}"
            draw.text(((index % cols) * sheet_w + 7, (index // cols) * sheet_h + 6), title[:58], fill="black")
            manifest.append({
                "image": sample["image"].name,
                "split": sample["split"],
                "class": "Clip" if sample["has_clip"] else sample["annotations"][0]["class"],
                "angled_clip": bool(sample["clip_angle"] >= 10),
                "angled_score_degrees": sample["clip_angle"],
                "annotations": sample["annotations"],
            })
        sheet.save(output / "polygon_box_contact_sheet.jpg", quality=94)
        (output / "selected_samples.json").write_text(json.dumps(manifest, indent=2))
        print(f"Prepared {len(manifest)} samples at {output}; red=converted box, class color=source polygon")
    except (FileNotFoundError, FileExistsError, ValueError, IndexError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
