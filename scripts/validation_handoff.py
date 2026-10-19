"""Generate final validation artifacts and per-class P/R/AP tables per run."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MATRIX_DIR = ROOT / "config" / "run_matrix"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=MATRIX_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    matrix_dir = args.matrix_dir if args.matrix_dir.is_absolute() else ROOT / args.matrix_dir
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    try:
        configs = sorted(matrix_dir.glob("*.yaml"))
        if len(configs) != 6:
            raise ValueError(f"Expected six configs under {matrix_dir}; found {len(configs)}")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite validation handoff: {output}")
        from ultralytics import YOLO
        import torch

        for config_path in configs:
            config = yaml.safe_load(config_path.read_text())
            for model_name in config["models"]:
                key = f"{config['experiment']}/{model_name}"
                checkpoint = (ROOT / "runs_compare" / config["experiment"] / model_name
                              / "weights" / "selected_by_val_map5095.pt")
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Run checkpoint selection first: {checkpoint}")
                run_output = output / config["experiment"] / model_name
                run_output.mkdir(parents=True)
                model = YOLO(str(checkpoint))
                metrics = model.val(
                    data=config["data"], split="val", plots=True, save_json=True,
                    project=str(run_output), name="validation", exist_ok=False,
                    device=str(config["device"]), batch=1, imgsz=640, quantize=16,
                )
                names = metrics.names
                classes = []
                for i, class_id in enumerate(metrics.ap_class_index):
                    precision, recall, ap50, ap5095 = metrics.class_result(i)
                    classes.append({
                        "class_id": int(class_id), "class_name": names[int(class_id)],
                        "precision": float(precision), "recall": float(recall),
                        "ap50": float(ap50), "ap50_95": float(ap5095),
                    })
                with (run_output / "per_class_validation_metrics.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(classes[0]) if classes else [])
                    writer.writeheader()
                    writer.writerows(classes)
                (run_output / "per_class_validation_metrics.json").write_text(json.dumps(classes, indent=2))
                (run_output / "validation_summary.json").write_text(json.dumps({
                    "run": key,
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "dataset": config["data"],
                    "inference_batch": 1,
                    "inference_imgsz": 640,
                    "inference_quantize": 16,
                    "metrics": {
                        "precision": float(metrics.box.mp),
                        "recall": float(metrics.box.mr),
                        "mAP50": float(metrics.box.map50),
                        "mAP50_95": float(metrics.box.map),
                    },
                    "classes": classes,
                    "plots_directory": "validation",
                }, indent=2))
                del model, metrics
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        print(f"Validation handoff saved under {output}")
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
