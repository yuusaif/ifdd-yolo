"""Validation threshold calibration and frozen-threshold test handoff.

`calibrate` reads validation images only. `test` requires the resulting locked
threshold JSON and writes test outputs to a new, user-named directory.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MATRIX_DIR = ROOT / "config" / "run_matrix"
CONFIDENCE_CANDIDATES = [i / 10 for i in range(1, 10)]
INFERENCE_BATCH = 1
INFERENCE_IMGSZ = 640
INFERENCE_QUANTIZE = 16
MIN_SWEEP_CONFIDENCE = min(CONFIDENCE_CANDIDATES)
PREDICTION_CONFIDENCE_FLOOR = MIN_SWEEP_CONFIDENCE - 0.001


def matrix_configs(matrix_dir: Path) -> list[dict]:
    paths = sorted(matrix_dir.glob("*.yaml"))
    if len(paths) != 6:
        raise ValueError(f"Expected six run-matrix configs under {matrix_dir}; found {len(paths)}")
    configs = [yaml.safe_load(path.read_text()) for path in paths]
    if sum(len(c["models"]) for c in configs) != 24:
        raise ValueError("Expected 24 model runs across the six configs")
    try:
        from scripts.audit_dataset import audit_dataset
    except ModuleNotFoundError:  # Support direct execution as `python scripts/...py`.
        from audit_dataset import audit_dataset

    audit_yaml = ROOT / "data" / "ifdd_930_v4_boxes" / "data.yaml"
    saved_audit = audit_yaml.parent / "dataset_audit.json"
    if not saved_audit.is_file():
        raise FileNotFoundError(f"Missing saved dataset audit: {saved_audit}")
    saved_fingerprint = json.loads(saved_audit.read_text())["dataset_sha256"]
    current_report = audit_dataset(audit_yaml)
    if current_report["issues"]:
        raise RuntimeError("Current dataset audit failed: " + "; ".join(current_report["issues"][:10]))
    if current_report["dataset_sha256"] != saved_fingerprint:
        raise ValueError("Dataset contents changed since the saved audit; recalibrate/review before evaluation")
    fingerprints = set()
    for config in configs:
        data_path = Path(config["data"])
        if not data_path.is_absolute():
            data_path = ROOT / data_path
        if data_path.resolve() != audit_yaml.resolve():
            raise ValueError(f"Run config points to a different dataset: {config['data']}")
        fingerprints.add(current_report["dataset_sha256"])
    if len(fingerprints) != 1:
        raise ValueError("Matrix configs do not share a single audited dataset fingerprint")
    return configs


def validate_prediction_floor(value: float) -> None:
    if not PREDICTION_CONFIDENCE_FLOOR <= value <= MIN_SWEEP_CONFIDENCE:
        raise ValueError(
            f"Minimum prediction confidence must be between {PREDICTION_CONFIDENCE_FLOOR} "
            f"and {MIN_SWEEP_CONFIDENCE} to cover the configured confidence sweep"
        )


def image_paths(data_yaml: Path, split: str) -> tuple[list[Path], dict]:
    if not data_yaml.is_absolute():
        data_yaml = ROOT / data_yaml
    data_yaml = data_yaml.resolve()
    data = yaml.safe_load(data_yaml.read_text())
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    value = data[split]
    source = Path(value)
    if not source.is_absolute():
        source = root / source
    if source.is_file():
        paths = [Path(line.strip()) for line in source.read_text().splitlines() if line.strip()]
        paths = [p if p.is_absolute() else source.parent / p for p in paths]
    else:
        paths = [p for p in source.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    paths = sorted(p.resolve() for p in paths)
    if not paths:
        raise FileNotFoundError(f"No {split} images found at {source}")
    return paths, data


def label_path(image: Path) -> Path:
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def read_ground_truth(image_paths_: list[Path]) -> dict[str, list[dict]]:
    import PIL.Image

    truth = {}
    for path in image_paths_:
        with PIL.Image.open(path) as image:
            width, height = image.size
        label = label_path(path)
        boxes = []
        for row in label.read_text().splitlines():
            fields = row.split()
            if len(fields) != 5:
                raise ValueError(f"Expected YOLO detection row in {label}: {row}")
            cls, xc, yc, w, h = map(float, fields)
            boxes.append({
                "class_id": int(cls),
                "xyxy": [(xc - w / 2) * width, (yc - h / 2) * height,
                         (xc + w / 2) * width, (yc + h / 2) * height],
            })
        truth[str(path)] = boxes
    return truth


def inference_settings(configs: list[dict], requested_device: str | None) -> tuple[str, int]:
    configured = {str(config["device"]) for config in configs}
    if requested_device is None:
        if len(configured) != 1:
            raise ValueError(f"Run matrix uses inconsistent devices: {sorted(configured)}")
        device = next(iter(configured))
    else:
        device = str(requested_device)
    if device.lower() == "cpu":
        return "cpu", 32
    return device, INFERENCE_QUANTIZE


def predict_all(checkpoint: Path, images: list[Path], conf: float, nms_iou: float, device: str, batch: int,
                quantize: int) -> dict[str, list[dict]]:
    from ultralytics import YOLO
    import torch

    model = YOLO(str(checkpoint))
    results = None
    result = None
    try:
        results = model.predict(
            source=[str(p) for p in images], conf=conf, iou=nms_iou, device=device,
            batch=batch, imgsz=INFERENCE_IMGSZ, quantize=quantize,
            save=False, verbose=False, stream=True,
        )
        output = {}
        for result in results:
            boxes = []
            if result.boxes is not None:
                xyxy = result.boxes.xyxy.cpu().tolist()
                classes = result.boxes.cls.cpu().tolist()
                scores = result.boxes.conf.cpu().tolist()
                boxes = [
                    {"class_id": int(cls), "confidence": float(score), "xyxy": [float(x) for x in box]}
                    for box, cls, score in zip(xyxy, classes, scores)
                ]
            output[str(Path(result.path).resolve())] = boxes
        if len(output) != len(images):
            raise RuntimeError(f"Prediction returned {len(output)} images for {len(images)} inputs")
        return output
    finally:
        result = None
        if results is not None:
            close = getattr(results, "close", None)
            if close is not None:
                close()
        results = None
        predictor = getattr(model, "predictor", None)
        if predictor is not None:
            predictor.results = None
        predictor = None
        model = None
        gc.collect()
        if str(device).lower() != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def score_predictions(truth: dict, predictions: dict, confidence: float, matching_iou: float) -> dict:
    tp = fp = fn = 0
    by_class = {}
    for image, targets in truth.items():
        preds = [p for p in predictions[image] if p["confidence"] >= confidence]
        possible = []
        for pi, pred in enumerate(preds):
            for ti, target in enumerate(targets):
                if pred["class_id"] == target["class_id"]:
                    overlap = iou(pred["xyxy"], target["xyxy"])
                    if overlap >= matching_iou:
                        possible.append((overlap, pi, ti))
        matched_p, matched_t = set(), set()
        for _, pi, ti in sorted(possible, reverse=True):
            if pi not in matched_p and ti not in matched_t:
                matched_p.add(pi)
                matched_t.add(ti)
        tp += len(matched_p)
        fp += len(preds) - len(matched_p)
        fn += len(targets) - len(matched_t)
        for cls in {x["class_id"] for x in targets} | {x["class_id"] for x in preds}:
            entry = by_class.setdefault(str(cls), {"tp": 0, "fp": 0, "fn": 0})
            cls_targets = sum(x["class_id"] == cls for x in targets)
            cls_preds = sum(x["class_id"] == cls for x in preds)
            cls_tp = sum(1 for pi in matched_p if preds[pi]["class_id"] == cls)
            entry["tp"] += cls_tp
            entry["fp"] += cls_preds - cls_tp
            entry["fn"] += cls_targets - cls_tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "by_class_id": by_class}


def save_failure_images(truth: dict, predictions: dict, confidence: float, matching_iou: float, out_dir: Path) -> None:
    from PIL import Image, ImageDraw

    fn_dir, fp_dir = out_dir / "FN", out_dir / "FP"
    fn_dir.mkdir(parents=True)
    fp_dir.mkdir(parents=True)
    for image_name, targets in truth.items():
        preds = [p for p in predictions[image_name] if p["confidence"] >= confidence]
        edges = []
        for pi, pred in enumerate(preds):
            for ti, target in enumerate(targets):
                if pred["class_id"] == target["class_id"]:
                    overlap = iou(pred["xyxy"], target["xyxy"])
                    if overlap >= matching_iou:
                        edges.append((overlap, pi, ti))
        used_preds, used_targets = set(), set()
        for _, pi, ti in sorted(edges, reverse=True):
            if pi not in used_preds and ti not in used_targets:
                used_preds.add(pi)
                used_targets.add(ti)
        misses = [t for i, t in enumerate(targets) if i not in used_targets]
        false_alarms = [p for i, p in enumerate(preds) if i not in used_preds]
        if not misses and not false_alarms:
            continue
        source = Path(image_name)
        stem = source.stem
        if misses:
            with Image.open(source) as im:
                canvas, draw = im.convert("RGB"), None
                draw = ImageDraw.Draw(canvas)
                for target in misses:
                    draw.rectangle(target["xyxy"], outline=(255, 40, 40), width=3)
                    draw.text((target["xyxy"][0], target["xyxy"][1]), f"FN class {target['class_id']}", fill=(255, 40, 40))
                canvas.save(fn_dir / f"{stem}.jpg", quality=95)
        if false_alarms:
            with Image.open(source) as im:
                canvas = im.convert("RGB")
                draw = ImageDraw.Draw(canvas)
                for pred in false_alarms:
                    draw.rectangle(pred["xyxy"], outline=(255, 180, 0), width=3)
                    draw.text((pred["xyxy"][0], pred["xyxy"][1]),
                              f"FP class {pred['class_id']} {pred['confidence']:.2f}", fill=(255, 180, 0))
                canvas.save(fp_dir / f"{stem}.jpg", quality=95)


def calibrate(args) -> None:
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output.exists():
        raise FileExistsError(f"Output directory exists; choose a new calibration name: {output}")
    validate_prediction_floor(args.min_prediction_confidence)
    configs = matrix_configs(args.matrix_dir)
    device, quantize = inference_settings(configs, args.device)
    audit_path = ROOT / "data" / "ifdd_930_v4_boxes" / "dataset_audit.json"
    audit = json.loads(audit_path.read_text())
    output.mkdir(parents=True)
    if not args.nms_candidates or any(not 0 < x < 1 for x in args.nms_candidates):
        raise ValueError("NMS candidates must be between 0 and 1")
    validation_sets = []
    for config in configs:
        images, data = image_paths(Path(config["data"]), "val")
        truth = read_ground_truth(images)
        for model_name in config["models"]:
            key = f"{config['experiment']}/{model_name}"
            checkpoint = (ROOT / "runs_compare" / config["experiment"] / model_name
                          / "weights" / "selected_by_val_map5095.pt")
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Run checkpoint selection first: {checkpoint}")
            print(f"Preparing validation calibration: {key}", flush=True)
            validation_sets.append({
                "key": key, "config": config, "model": model_name,
                "checkpoint": checkpoint, "images": images, "truth": truth,
            })

    # Select one shared confidence and NMS threshold over the validation
    # predictions for all 24 runs, so test comparisons use identical settings.
    all_rows = []
    saved_predictions = {}
    for nms_iou in args.nms_candidates:
        totals = {confidence: {"tp": 0, "fp": 0, "fn": 0} for confidence in CONFIDENCE_CANDIDATES}
        for item in validation_sets:
            print(
                f"Calibrating {item['key']} at NMS IoU={nms_iou} "
                f"(conf_floor={args.min_prediction_confidence}, batch={INFERENCE_BATCH}, "
                f"imgsz={INFERENCE_IMGSZ}, quantize={quantize}, device={device})",
                flush=True,
            )
            predictions = predict_all(
                item["checkpoint"], item["images"], args.min_prediction_confidence,
                nms_iou, device, INFERENCE_BATCH, quantize,
            )
            saved_predictions[(nms_iou, item["key"])] = predictions
            for confidence in CONFIDENCE_CANDIDATES:
                score = score_predictions(item["truth"], predictions, confidence, args.matching_iou)
                for metric in ("tp", "fp", "fn"):
                    totals[confidence][metric] += score[metric]
        for confidence, counts in totals.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            all_rows.append({
                "confidence": confidence, "nms_iou": nms_iou, **counts,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
            })
    chosen = sorted(all_rows, key=lambda r: (-r["recall"], r["fp"], -r["confidence"], -r["nms_iou"]))[0]
    threshold_doc = {
        "dataset_sha256": audit["dataset_sha256"],
        "selection_scope": "one shared operating point across all 24 runs",
        "policy": "maximize pooled validation recall, then minimize pooled false positives, then prefer higher confidence and higher NMS IoU",
        "matching_iou": args.matching_iou,
        "minimum_prediction_confidence": args.min_prediction_confidence,
        "inference_batch": INFERENCE_BATCH,
        "inference_imgsz": INFERENCE_IMGSZ,
        "inference_quantize": quantize,
        "inference_device": device,
        "validation_sweep": all_rows,
        "selected": chosen,
        "runs": {},
    }
    for item in validation_sets:
        key = item["key"]
        predictions = saved_predictions[(chosen["nms_iou"], key)]
        model_dir = output / item["config"]["experiment"] / item["model"]
        model_dir.mkdir(parents=True)
        (model_dir / "validation_predictions.json").write_text(json.dumps(predictions))
        (model_dir / "confidence_sweep.json").write_text(json.dumps([
            {"confidence": row["confidence"], "nms_iou": row["nms_iou"],
             **score_predictions(item["truth"], predictions, row["confidence"], args.matching_iou)}
            for row in all_rows if row["nms_iou"] == chosen["nms_iou"]
        ], indent=2))
        threshold_doc["runs"][key] = {"checkpoint": str(item["checkpoint"].relative_to(ROOT))}
    (output / "locked_thresholds.json").write_text(json.dumps(threshold_doc, indent=2))
    print(f"Shared confidence={chosen['confidence']}, NMS IoU={chosen['nms_iou']}; locked in {output / 'locked_thresholds.json'}")


def evaluate_test(args) -> None:
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite test output directory: {output}")
    configs = matrix_configs(args.matrix_dir)
    device, quantize = inference_settings(configs, args.device)
    locked_path = args.thresholds if args.thresholds.is_absolute() else ROOT / args.thresholds
    locked = json.loads(locked_path.read_text())
    validate_prediction_floor(args.min_prediction_confidence)
    audit = json.loads((ROOT / "data" / "ifdd_930_v4_boxes" / "dataset_audit.json").read_text())
    if locked.get("dataset_sha256") != audit.get("dataset_sha256"):
        raise ValueError("Locked thresholds do not match the current audited dataset fingerprint")
    if locked.get("inference_device") != device or locked.get("inference_quantize") != quantize:
        raise ValueError(
            "Test inference settings must match calibration: "
            f"locked device={locked.get('inference_device')}, quantize={locked.get('inference_quantize')}; "
            f"requested device={device}, quantize={quantize}"
        )
    output.mkdir(parents=True)
    source_groups = None
    if args.source_groups is not None:
        group_path = args.source_groups if args.source_groups.is_absolute() else ROOT / args.source_groups
        source_groups = json.loads(group_path.read_text())
        if not isinstance(source_groups, dict):
            raise ValueError("Source-group mapping must be a JSON object keyed by image filename")
    aggregate = []
    for config in configs:
        images, _data = image_paths(Path(config["data"]), "test")
        truth = read_ground_truth(images)
        for model_name in config["models"]:
            key = f"{config['experiment']}/{model_name}"
            if key not in locked["runs"]:
                raise ValueError(f"No validation calibration record for {key}")
            confidence = float(locked["selected"]["confidence"])
            nms_iou = float(locked["selected"]["nms_iou"])
            matching_iou = float(locked["matching_iou"])
            run = ROOT / "runs_compare" / config["experiment"] / model_name
            checkpoint = run / "weights" / "selected_by_val_map5095.pt"
            model_out = output / config["experiment"] / model_name
            model_out.mkdir(parents=True)
            from ultralytics import YOLO

            print(f"Final locked test evaluation: {key}", flush=True)
            model = YOLO(str(checkpoint))
            model.val(
                data=config["data"], split="test", conf=confidence, iou=nms_iou,
                save_json=True, plots=True, project=str(model_out), name="ultralytics_val",
                exist_ok=False, device=device, batch=INFERENCE_BATCH,
                imgsz=INFERENCE_IMGSZ, quantize=quantize,
            )
            del model
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Retain low-confidence boxes once, then compute the confidence sweep
            # offline; never retune from test results.
            predictions = predict_all(checkpoint, images, args.min_prediction_confidence, nms_iou,
                                      device, INFERENCE_BATCH, quantize)
            metrics = score_predictions(truth, predictions, confidence, matching_iou)
            confidence_sweep = [
                {"confidence": candidate,
                 **score_predictions(truth, predictions, candidate, matching_iou)}
                for candidate in CONFIDENCE_CANDIDATES
            ]
            (model_out / "test_predictions_per_image.json").write_text(json.dumps(predictions))
            save_failure_images(truth, predictions, confidence, matching_iou, model_out / "failure_images")
            (model_out / "test_confidence_sweep_descriptive.json").write_text(
                json.dumps(confidence_sweep, indent=2)
            )
            if source_groups is not None:
                missing = [Path(path).name for path in truth if Path(path).name not in source_groups]
                if missing:
                    raise ValueError(f"Source-group mapping is missing test images, e.g. {missing[:5]}")
                subgroup_metrics = {}
                for group_name in sorted(set(source_groups.values())):
                    selected = {path for path in truth if source_groups[Path(path).name] == group_name}
                    subset_truth = {path: truth[path] for path in selected}
                    subset_predictions = {path: predictions[path] for path in selected}
                    subgroup_metrics[group_name] = {
                        "images": len(selected),
                        **score_predictions(subset_truth, subset_predictions, confidence, matching_iou),
                    }
                (model_out / "test_source_group_metrics.json").write_text(json.dumps(subgroup_metrics, indent=2))
            record = {
                "experiment": config["experiment"], "model": model_name,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "confidence": confidence, "nms_iou": nms_iou,
                "matching_iou": matching_iou, "inference_batch": INFERENCE_BATCH,
                "inference_imgsz": INFERENCE_IMGSZ, "inference_quantize": quantize,
                "inference_device": device, **metrics,
            }
            (model_out / "fixed_threshold_test_metrics.json").write_text(json.dumps(record, indent=2))
            aggregate.append(record)
    (output / "test_metrics_all_runs.json").write_text(json.dumps(aggregate, indent=2))
    print(f"Test handoff saved to {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate", help="Select confidence using validation predictions only")
    cal.add_argument("--matrix-dir", type=Path, default=MATRIX_DIR)
    cal.add_argument("--output-dir", type=Path, required=True)
    cal.add_argument("--device", help="Inference device for all models (for example: cpu or 0); recorded in thresholds")
    cal.add_argument("--min-prediction-confidence", type=float, default=PREDICTION_CONFIDENCE_FLOOR,
                     help="Lowest retained score; use 0.099 to cover the 0.1–0.9 sweep without excess NMS candidates")
    cal.add_argument("--nms-candidates", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    cal.add_argument("--matching-iou", type=float, default=0.5)
    test = sub.add_parser("test", help="Run the final test handoff from locked validation thresholds")
    test.add_argument("--matrix-dir", type=Path, default=MATRIX_DIR)
    test.add_argument("--thresholds", type=Path, required=True)
    test.add_argument("--output-dir", type=Path, required=True)
    test.add_argument("--device", help="Must match the device used for calibration (for example: cpu or 0)")
    test.add_argument("--min-prediction-confidence", type=float, default=PREDICTION_CONFIDENCE_FLOOR,
                      help="Lowest retained score; use 0.099 to cover the 0.1–0.9 sweep without excess NMS candidates")
    test.add_argument("--source-groups", type=Path,
                      help="Optional JSON mapping image basename to source group, e.g. original_630/new_300")
    args = parser.parse_args()
    try:
        if args.command == "calibrate":
            calibrate(args)
        else:
            evaluate_test(args)
    except (FileNotFoundError, ValueError, RuntimeError, FileExistsError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
