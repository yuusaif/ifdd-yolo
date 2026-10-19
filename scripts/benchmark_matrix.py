"""Measure parameters, GFLOPs, and batch-1 validation-image latency on the run device."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MATRIX_DIR = ROOT / "config" / "run_matrix"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=MATRIX_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images-per-repeat", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    matrix_dir = args.matrix_dir if args.matrix_dir.is_absolute() else ROOT / args.matrix_dir
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    try:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite benchmark output: {output}")
        configs = sorted(matrix_dir.glob("*.yaml"))
        if len(configs) != 6:
            raise ValueError(f"Expected six matrix configs, found {len(configs)}")
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("Benchmark requires CUDA; run on the intended RTX 3050 host")
        output.mkdir(parents=True)
        rows = []
        for config_path in configs:
            config = yaml.safe_load(config_path.read_text())
            if int(config["train"]["seed"]) != 42:
                continue  # benchmark the eight scale/architecture combinations once
            data_yaml = Path(config["data"])
            if not data_yaml.is_absolute():
                data_yaml = ROOT / data_yaml
            data = yaml.safe_load(data_yaml.read_text())
            data_root = Path(data.get("path", data_yaml.parent))
            if not data_root.is_absolute():
                data_root = data_yaml.parent / data_root
            val_dir = Path(data["val"])
            if not val_dir.is_absolute():
                val_dir = data_root / val_dir
            images = sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if not images:
                raise FileNotFoundError(f"No benchmark images in {val_dir}")
            for model_name in config["models"]:
                checkpoint = (ROOT / "runs_compare" / config["experiment"] / model_name
                              / "weights" / "selected_by_val_map5095.pt")
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Missing selected checkpoint: {checkpoint}")
                model = YOLO(str(checkpoint))
                info = model.info(verbose=True, imgsz=640)
                if not info or len(info) < 4:
                    raise RuntimeError(f"Unable to read model parameter/FLOP info for {model_name}")
                _, params, _, gflops = info
                device = str(config["device"])
                for index in range(args.warmup):
                    model.predict(str(images[index % len(images)]), imgsz=640, batch=1,
                                  device=device, conf=0.25, iou=0.7, verbose=False)
                if device != "cpu":
                    torch.cuda.synchronize()
                repeat_latencies = []
                for _repeat in range(3):
                    elapsed = []
                    for index in range(args.images_per_repeat):
                        image = images[index % len(images)]
                        if device != "cpu":
                            torch.cuda.synchronize()
                        start = time.perf_counter()
                        model.predict(str(image), imgsz=640, batch=1, device=device,
                                      conf=0.25, iou=0.7, verbose=False)
                        if device != "cpu":
                            torch.cuda.synchronize()
                        elapsed.append((time.perf_counter() - start) * 1000)
                    repeat_latencies.append(statistics.mean(elapsed))
                mean_ms = statistics.mean(repeat_latencies)
                rows.append({
                    "experiment": config["experiment"], "model": model_name,
                    "seed": config["train"]["seed"], "parameters": int(params),
                    "gflops": float(gflops), "batch": 1,
                    "imgsz": 640, "benchmark_confidence": 0.25, "benchmark_nms_iou": 0.7,
                    "timing_scope": "end-to-end single-image predict call including image decode",
                    "pytorch": str(torch.__version__), "cuda": str(torch.version.cuda),
                    "gpu": torch.cuda.get_device_name(0),
                    "repeat_mean_latency_ms": repeat_latencies,
                    "latency_ms_mean": mean_ms,
                    "latency_ms_sd_between_repeats": statistics.stdev(repeat_latencies),
                    "fps_from_mean_latency": 1000 / mean_ms if mean_ms else 0,
                    "timed_images_per_repeat": args.images_per_repeat,
                    "warmup_images": args.warmup,
                })
                print(f"Benchmarked {config['experiment']}/{model_name}", flush=True)
        with (output / "benchmark_matrix.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Benchmark table saved under {output}")
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
