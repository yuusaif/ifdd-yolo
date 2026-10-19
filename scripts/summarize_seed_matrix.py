"""Build per-run and mean ± SD validation tables for all 24 completed runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
MATRIX_DIR = ROOT / "config" / "run_matrix"
METRICS = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=MATRIX_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs_compare" / "matrix_summary")
    args = parser.parse_args()
    matrix_dir = args.matrix_dir if args.matrix_dir.is_absolute() else ROOT / args.matrix_dir
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    try:
        configs = sorted(matrix_dir.glob("*.yaml"))
        if len(configs) != 6:
            raise ValueError(f"Expected six configs; found {len(configs)} in {matrix_dir}")
        rows, fingerprints = [], set()
        for config_path in configs:
            config = yaml.safe_load(config_path.read_text())
            experiment = config["experiment"]
            for model in config["models"]:
                run = ROOT / "runs_compare" / experiment / model
                results_path = run / "results.csv"
                manifest_path = run / "run_manifest.json"
                if not results_path.is_file() or not manifest_path.is_file():
                    raise FileNotFoundError(f"Incomplete run handoff: {run}")
                manifest = json.loads(manifest_path.read_text())
                fingerprints.add(manifest["dataset_sha256"])
                df = pd.read_csv(results_path)
                df.columns = df.columns.str.strip()
                if len(df) != int(config["train"]["epochs"]):
                    raise ValueError(f"Incomplete epoch history for {run}: {len(df)} rows")
                metric = "metrics/mAP50-95(B)"
                absent = [key for key in ["epoch", *METRICS] if key not in df]
                if absent:
                    raise ValueError(f"Missing metric columns in {results_path}: {absent}")
                best = df.loc[pd.to_numeric(df[metric], errors="raise").idxmax()]
                family = model[:-1]
                scale = config["scale"]
                rows.append({
                    "family": family, "scale": scale, "model": model,
                    "seed": config["train"]["seed"],
                    "experiment": experiment,
                    "selected_epoch": int(best["epoch"]),
                    **{name: float(best[column]) for column, name in METRICS.items()},
                })
        if len(fingerprints) != 1:
            raise ValueError(f"Runs do not share one dataset fingerprint: {sorted(fingerprints)}")
        per_run = pd.DataFrame(rows).sort_values(["family", "scale", "seed"])
        means = per_run.groupby(["family", "scale"], as_index=False).agg(
            seeds=("seed", "nunique"),
            **{f"{metric}_mean": (metric, "mean") for metric in METRICS.values()},
            **{f"{metric}_sd": (metric, "std") for metric in METRICS.values()},
        )
        if (means["seeds"] != 3).any():
            raise ValueError("Every family/scale group must include all three seeds")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing matrix summary: {output}")
        output.mkdir(parents=True)
        per_run.to_csv(output / "per_run_validation.csv", index=False)
        means.to_csv(output / "mean_sd_validation.csv", index=False)
        (output / "dataset_fingerprint.txt").write_text(next(iter(fingerprints)) + "\n")
        print(f"Validation matrix summary saved under {output}")
    except (FileNotFoundError, FileExistsError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
