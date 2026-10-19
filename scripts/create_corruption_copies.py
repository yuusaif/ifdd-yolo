"""Create severity-controlled test copies without modifying the audited dataset."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ifdd_930_v4_boxes" / "data.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data_yaml = args.data if args.data.is_absolute() else ROOT / args.data
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    try:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite corruption copies: {output}")
        spec = yaml.safe_load(data_yaml.read_text())
        data_root = Path(spec.get("path", data_yaml.parent))
        if not data_root.is_absolute():
            data_root = (data_yaml.parent / data_root).resolve()
        test_dir = Path(spec["test"])
        if not test_dir.is_absolute():
            test_dir = data_root / test_dir
        images = sorted(p for p in test_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not images:
            raise FileNotFoundError(f"No test images in {test_dir}")
        output.mkdir(parents=True)
        rng = np.random.default_rng(args.seed)
        transforms = {}
        for severity, radius in enumerate((1.0, 2.0, 3.0), 1):
            transforms[f"gaussian_blur_s{severity}"] = lambda im, v=radius: im.filter(ImageFilter.GaussianBlur(v))
        for severity, factor in enumerate((0.8, 0.6, 0.4), 1):
            transforms[f"brightness_down_s{severity}"] = lambda im, v=factor: ImageEnhance.Brightness(im).enhance(v)
        for severity, factor in enumerate((1.2, 1.4, 1.6), 1):
            transforms[f"brightness_up_s{severity}"] = lambda im, v=factor: ImageEnhance.Brightness(im).enhance(v)
        for severity, sigma in enumerate((10, 25, 40), 1):
            def add_noise(im, amount=sigma):
                array = np.asarray(im.convert("RGB"), dtype=np.float32)
                noise = rng.normal(0, amount, array.shape)
                return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
            transforms[f"gaussian_noise_s{severity}"] = add_noise
        for severity, quality in enumerate((60, 35, 15), 1):
            transforms[f"jpeg_quality_s{severity}"] = (lambda im, q=quality: im.copy(), quality)

        for name, transform in transforms.items():
            severity_dir = output / name
            image_out, label_out = severity_dir / "images", severity_dir / "labels"
            image_out.mkdir(parents=True)
            label_out.mkdir(parents=True)
            jpeg_quality = None
            if isinstance(transform, tuple):
                transform, jpeg_quality = transform
            for image_path in images:
                with Image.open(image_path) as original:
                    converted = transform(original.convert("RGB"))
                    target = image_out / f"{image_path.stem}.jpg"
                    converted.save(target, quality=jpeg_quality or 95)
                label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
                shutil.copy2(label_path, label_out / label_path.name)
            corrupted_yaml = {
                "path": str(data_root),
                "train": spec["train"],
                "val": spec["val"],
                "test": str(image_out.resolve()),
                "nc": spec["nc"],
                "names": spec["names"],
            }
            (severity_dir / "data.yaml").write_text(yaml.safe_dump(corrupted_yaml, sort_keys=False))
        print(f"Created {len(transforms)} corruption copies for {len(images)} test images under {output}")
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
