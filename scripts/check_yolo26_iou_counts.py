"""Compare YOLO26m prediction box counts at two NMS IoU values on validation images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs_compare/ifdd_930_v4_medium_seed42/yolo26m/weights/selected_by_val_map5095.pt"),
    )
    parser.add_argument("--data", type=Path, default=Path("data/ifdd_930_v4_boxes/data.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs_compare/yolo26_iou_check_84179"))
    parser.add_argument("--images", type=int, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if ultralytics.__version__ != "8.4.179":
        raise RuntimeError(f"Expected Ultralytics 8.4.179; found {ultralytics.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("This check requires CUDA on the RTX 3050 host")
    if args.device == "cpu":
        raise ValueError("Use the CUDA device for this check, for example --device 0")
    if args.images < 1:
        raise ValueError("--images must be positive")

    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    data_yaml = args.data if args.data.is_absolute() else ROOT / args.data
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")

    spec = yaml.safe_load(data_yaml.read_text())
    data_root = Path(spec.get("path", data_yaml.parent))
    if not data_root.is_absolute():
        data_root = (data_yaml.parent / data_root).resolve()
    val_dir = Path(spec["val"])
    if not val_dir.is_absolute():
        val_dir = data_root / val_dir
    images = sorted(
        p for p in val_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[: args.images]
    if len(images) != args.images:
        raise RuntimeError(f"Requested {args.images} images, found {len(images)} under {val_dir}")

    device_index = int(args.device) if args.device.isdigit() else 0
    gpu_name = torch.cuda.get_device_name(device_index)
    model = YOLO(str(checkpoint))
    rows = []
    for image in images:
        counts = []
        for iou in (0.5, 0.9):
            results = model.predict(
                source=str(image),
                imgsz=args.imgsz,
                batch=1,
                device=args.device,
                conf=args.conf,
                iou=iou,
                quantize=16,
                max_det=300,
                augment=False,
                verbose=False,
            )
            if len(results) != 1:
                raise RuntimeError(f"Expected one result for {image.name}, received {len(results)}")
            counts.append(len(results[0].boxes) if results[0].boxes is not None else 0)
        rows.append(
            {
                "image": image.name,
                "boxes_iou_0_5": counts[0],
                "boxes_iou_0_9": counts[1],
                "count_changed": counts[0] != counts[1],
            }
        )

    output_dir.mkdir(parents=True)
    with (output_dir / "box_counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    changed = sum(row["count_changed"] for row in rows)
    summary = {
        "ultralytics": ultralytics.__version__,
        "pytorch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "gpu": gpu_name,
        "device": args.device,
        "precision": "FP16",
        "checkpoint": str(checkpoint),
        "validation_images": len(rows),
        "imgsz": args.imgsz,
        "confidence": args.conf,
        "iou_values": [0.5, 0.9],
        "images_with_changed_box_counts": changed,
        "total_boxes_iou_0_5": sum(row["boxes_iou_0_5"] for row in rows),
        "total_boxes_iou_0_9": sum(row["boxes_iou_0_9"] for row in rows),
        "box_counts_csv": "box_counts.csv",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Images with changed box counts: {changed}/{len(rows)}; "
        f"totals: IoU 0.5={summary['total_boxes_iou_0_5']}, "
        f"IoU 0.9={summary['total_boxes_iou_0_9']}"
    )
    print(f"Saved results under {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, KeyError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
