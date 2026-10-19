"""Create six immutable configs for the four-model, two-scale, three-seed study."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "config" / "experiment.yaml"
OUTPUT_DIR = ROOT / "config" / "run_matrix"
SCALES = {
    "nano": {
        "yolov8n": "yolov8n.pt",
        "yolo11n": "yolo11n.pt",
        "yolo12n": "yolo12n.pt",
        "yolo26n": "weights/yolo26n.pt",
    },
    "medium": {
        "yolov8m": "yolov8m.pt",
        "yolo11m": "yolo11m.pt",
        "yolo12m": "yolo12m.pt",
        "yolo26m": "yolo26m.pt",
    },
}
SEEDS = (42, 0, 1)


def build_configs() -> dict[Path, dict]:
    base = yaml.safe_load(BASE_CONFIG.read_text())
    if not isinstance(base, dict) or not isinstance(base.get("train"), dict):
        raise ValueError(f"Invalid base experiment config: {BASE_CONFIG}")
    if base.get("training_ready") is not True:
        raise ValueError("Base config is not training_ready")
    if base["train"].get("batch") != 4:
        raise ValueError("This study requires the shared batch size 4 for every scale and model")

    generated = {}
    for scale, models in SCALES.items():
        for seed in SEEDS:
            config = copy.deepcopy(base)
            config["experiment"] = f"ifdd_930_v4_{scale}_seed{seed}"
            config["scale"] = scale
            config["train"]["seed"] = seed
            config["models"] = models.copy()
            config["study_protocol"] = {
                "seed": seed,
                "scale": scale,
                "shared_batch": 4,
                "dataset_review": "data/ifdd_930_v4_boxes/dataset_review.json",
                "checkpoint_selection": "maximum validation mAP50-95; ties resolved by earlier epoch",
                "test_policy": "test only after all planned runs and validation decisions are frozen",
            }
            path = OUTPUT_DIR / f"{scale}_seed{seed}.yaml"
            generated[path] = config
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write configs; without this flag, print a preview")
    args = parser.parse_args()
    try:
        configs = build_configs()
        existing = [path for path in configs if path.exists()]
        if args.write and existing:
            raise FileExistsError(
                "Refusing to overwrite existing matrix configs: "
                + ", ".join(str(path.relative_to(ROOT)) for path in existing)
            )
        for path, config in configs.items():
            if args.write:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(config, sort_keys=False))
            print(f"{path.relative_to(ROOT)}: {config['experiment']} ({len(config['models'])} models)")
        if not args.write:
            print("Preview only. Pass --write to create these files.")
    except (FileNotFoundError, ValueError, FileExistsError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
