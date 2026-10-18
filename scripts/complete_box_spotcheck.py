"""Validate a completed box review record and unlock the six run configs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True,
                        help="Completed JSON based on config/box_spotcheck_review.template.json")
    args = parser.parse_args()
    review_path = args.review if args.review.is_absolute() else ROOT / args.review
    review_target = ROOT / "data" / "ifdd_930_v4_boxes" / "box_spotcheck_review.json"
    audit_path = ROOT / "data" / "ifdd_930_v4_boxes" / "dataset_audit.json"
    config_paths = [ROOT / "config" / "experiment.yaml", *sorted((ROOT / "config" / "run_matrix").glob("*.yaml"))]
    try:
        if review_target.exists():
            raise FileExistsError(f"Review record already exists; preserve it and investigate: {review_target}")
        if len(config_paths) != 7:
            raise FileNotFoundError("Expected base config plus all six generated run configs")
        record = json.loads(review_path.read_text())
        audit = json.loads(audit_path.read_text())
        items = record.get("records", [])
        unique_images = {str(item.get("image", "")) for item in items}
        if (record.get("dataset_sha256") != audit.get("dataset_sha256")
                or int(record.get("samples_reviewed", 0)) < 30
                or len(items) < 30 or len(unique_images) < 30
                or int(record.get("angled_clip_samples", 0)) < 1
                or sum(item.get("angled_clip") is True for item in items[:30]) < 1
                or not all(item.get("conversion_acceptable") is True for item in items[:30])):
            raise ValueError(
                "Review must match the current dataset, document 30 distinct acceptable samples, "
                "and include angled clips. If any box is unacceptable, fix/re-audit the dataset first."
            )
        updated = []
        for path in config_paths:
            config = yaml.safe_load(path.read_text())
            check = config.setdefault("preflight", {}).setdefault("box_conversion_spotcheck", {})
            check["status"] = "complete"
            check["record"] = "data/ifdd_930_v4_boxes/box_spotcheck_review.json"
            updated.append((path, yaml.safe_dump(config, sort_keys=False)))
        for path, contents in updated:
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(contents)
            os.replace(temp, path)
        review_target.write_text(json.dumps(record, indent=2))
        print(f"Validated review and unlocked {len(config_paths) - 1} matrix configs")
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
