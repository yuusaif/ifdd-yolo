"""Verify the 24 validation-selected checkpoints and write a selection index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs_compare"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=ROOT / "config" / "run_matrix")
    parser.add_argument("--output", type=Path, default=RUNS_ROOT / "checkpoint_selection_summary.json")
    args = parser.parse_args()
    matrix_dir = args.matrix_dir if args.matrix_dir.is_absolute() else ROOT / args.matrix_dir
    output = args.output if args.output.is_absolute() else ROOT / args.output
    try:
        configs = sorted(matrix_dir.glob("*.yaml"))
        if len(configs) != 6:
            raise ValueError(f"Expected six generated matrix configs under {matrix_dir}, found {len(configs)}")
        records = []
        for config_path in configs:
            config = yaml.safe_load(config_path.read_text())
            for model_name in config["models"]:
                run = RUNS_ROOT / config["experiment"] / model_name
                results_path = run / "results.csv"
                checkpoint = run / "weights" / "selected_by_val_map5095.pt"
                selection_path = run / "checkpoint_selection.json"
                manifest_path = run / "run_manifest.json"
                if not all(p.is_file() for p in (results_path, checkpoint, selection_path, manifest_path)):
                    raise FileNotFoundError(f"Incomplete run/selection handoff under {run}")
                df = pd.read_csv(results_path)
                df.columns = df.columns.str.strip()
                metric = "metrics/mAP50-95(B)"
                if len(df) != int(config["train"]["epochs"]) or metric not in df or "epoch" not in df:
                    raise ValueError(f"Incomplete or invalid results.csv under {run}")
                best = df.loc[pd.to_numeric(df[metric], errors="raise").idxmax()]
                selection = json.loads(selection_path.read_text())
                if int(selection["epoch_csv_value_one_based"]) != int(best["epoch"]):
                    raise ValueError(f"Selection record epoch disagrees with results.csv under {run}")
                if abs(float(selection["metric_value"]) - float(best[metric])) > 1e-7:
                    raise ValueError(f"Selection record metric disagrees with results.csv under {run}")
                manifest = json.loads(manifest_path.read_text())
                if manifest["dataset_sha256"] != json.loads(
                    (ROOT / "data" / "ifdd_930_v4_boxes" / "dataset_audit.json").read_text()
                )["dataset_sha256"]:
                    raise ValueError(f"Dataset fingerprint mismatch under {run}")
                records.append({
                    "experiment": config["experiment"], "model": model_name,
                    "selected_epoch": int(best["epoch"]),
                    "selected_map50_95": float(best[metric]),
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                })
        if output.exists():
            raise FileExistsError(f"Selection index already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(records, indent=2))
        print(f"Verified {len(records)} selected checkpoints; index: {output}")
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
