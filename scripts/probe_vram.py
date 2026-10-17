"""Measure model forward/backward VRAM without using the dataset or training weights."""

from __future__ import annotations

import argparse
import gc
from types import SimpleNamespace
import torch
from ultralytics import YOLO


DEFAULT_MODELS = ("yolov8m.pt", "yolo11m.pt", "yolo12m.pt", "yolo26m.pt")


def mib(value: int) -> float:
    return value / 1024**2


def probe(weights: str, batch: int, imgsz: int) -> dict:
    torch.cuda.empty_cache()
    gc.collect()
    model = YOLO(weights)
    network = model.model.to("cuda").train()
    network.requires_grad_(True)
    # Standalone checkpoints carry model.args as a dict; the Trainer normally
    # replaces it with an attribute namespace before calling model.loss().
    network.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    images = torch.zeros((batch, 3, imgsz, imgsz), device="cuda")
    targets = {
        "img": images,
        "batch_idx": torch.empty(0, device="cuda", dtype=torch.long),
        "cls": torch.empty((0, 1), device="cuda"),
        "bboxes": torch.empty((0, 4), device="cuda"),
    }
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # Use the model's actual detection-loss route to preserve the training graph.
            loss, _ = network.loss(targets)
        loss.sum().backward()
        result = {
            "status": "fits",
            "peak_allocated_mib": round(mib(torch.cuda.max_memory_allocated()), 1),
            "peak_reserved_mib": round(mib(torch.cuda.max_memory_reserved()), 1),
        }
    except torch.OutOfMemoryError:
        result = {"status": "oom"}
    finally:
        del targets, images, network, model
        torch.cuda.empty_cache()
        gc.collect()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable to this Python process")
    properties = torch.cuda.get_device_properties(0)
    print(f"GPU: {properties.name}; VRAM: {mib(properties.total_memory):.1f} MiB")
    print("This is a synthetic forward/backward probe; it does not train on images or write a run.")
    for weights in args.models:
        print(f"\n{weights}")
        for batch in args.batches:
            result = probe(weights, batch, args.imgsz)
            details = " ".join(f"{key}={value}" for key, value in result.items())
            print(f"  batch={batch}: {details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
