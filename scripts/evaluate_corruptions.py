"""Evaluate all selected checkpoints on generated corruption test copies."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MATRIX_DIR = ROOT / "config" / "run_matrix"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corruption-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix-dir", type=Path, default=MATRIX_DIR)
    args = parser.parse_args()
    corruption_dir = args.corruption_dir if args.corruption_dir.is_absolute() else ROOT / args.corruption_dir
    threshold_path = args.thresholds if args.thresholds.is_absolute() else ROOT / args.thresholds
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    matrix_dir = args.matrix_dir if args.matrix_dir.is_absolute() else ROOT / args.matrix_dir
    try:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite corruption evaluation: {output}")
        corruption_yamls = sorted(corruption_dir.glob("*/data.yaml"))
        configs = sorted(matrix_dir.glob("*.yaml"))
        if not corruption_yamls or len(configs) != 6:
            raise FileNotFoundError("Need corruption data.yaml files and all six run-matrix configs")
        selected = json.loads(threshold_path.read_text())["selected"]
        confidence, nms_iou = float(selected["confidence"]), float(selected["nms_iou"])
        from ultralytics import YOLO

        output.mkdir(parents=True)
        rows = []
        for config_path in configs:
            config = yaml.safe_load(config_path.read_text())
            for model_name in config["models"]:
                key = f"{config['experiment']}/{model_name}"
                checkpoint = (ROOT / "runs_compare" / config["experiment"] / model_name
                              / "weights" / "selected_by_val_map5095.pt")
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Missing selected checkpoint: {checkpoint}")
                for data_path in corruption_yamls:
                    corruption = data_path.parent.name
                    result_dir = output / config["experiment"] / model_name
                    result_dir.mkdir(parents=True, exist_ok=True)
                    metrics = YOLO(str(checkpoint)).val(
                        data=str(data_path), split="test", conf=confidence, iou=nms_iou,
                        plots=False, save_json=False, project=str(result_dir), name=corruption,
                        exist_ok=False, device=str(config["device"]), batch=1, imgsz=640, quantize=16,
                    )
                    rows.append({
                        "run": key, "corruption": corruption,
                        "confidence": confidence, "nms_iou": nms_iou,
                        "inference_batch": 1, "inference_imgsz": 640, "inference_quantize": 16,
                        "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
                        "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
                    })
                    print(f"Completed {key} on {corruption}", flush=True)
        with (output / "corruption_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Corruption results saved under {output}")
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, KeyError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
