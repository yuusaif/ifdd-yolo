"""Train the four IFDD detectors, or rebuild comparisons from completed runs.

Training never starts during import. Pass --skip-train to analyze existing runs.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import yaml

from scripts.audit_dataset import audit_dataset

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "experiment.yaml"
PLOTS = {
    "metrics/mAP50(B)": ("Validation mAP@0.5", "cmp_map50.png"),
    "metrics/mAP50-95(B)": ("Validation mAP@0.5:0.95", "cmp_map50_95.png"),
    "metrics/precision(B)": ("Validation precision", "cmp_precision.png"),
    "metrics/recall(B)": ("Validation recall", "cmp_recall.png"),
    "train/box_loss": ("Train box loss (diagnostic)", "cmp_train_box_loss.png"),
    "val/box_loss": ("Validation box loss (diagnostic)", "cmp_val_box_loss.png"),
    "train/cls_loss": ("Train class loss (diagnostic)", "cmp_train_cls_loss.png"),
    "val/cls_loss": ("Validation class loss (diagnostic)", "cmp_val_cls_loss.png"),
}
METRICS = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
}


def load_config(path: Path, data_override: str | None, device_override: str | None) -> dict:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or not all(k in config for k in ("experiment", "data", "device", "models", "train")):
        raise ValueError("Experiment YAML needs experiment, data, device, models and train")
    if data_override:
        config["data"] = data_override
    if device_override is not None:
        config["device"] = device_override
    data_path = Path(config["data"])
    config["data"] = str((data_path if data_path.is_absolute() else ROOT / data_path).resolve())
    return config


def train_all(config: dict, run_dir: Path) -> None:
    if config.get("training_ready") is not True:
        raise RuntimeError("Training is locked: set training_ready: true only after dataset and settings are frozen")
    if config.get("augmentation_strategy") in (None, "undecided"):
        raise ValueError("Choose and record an augmentation_strategy before training")
    batch = config["train"].get("batch")
    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        raise ValueError("Choose a fixed positive integer train.batch before training")
    from ultralytics import YOLO, __version__ as ultralytics_version
    import torch

    report = audit_dataset(Path(config["data"]))
    if report["issues"]:
        raise RuntimeError("Dataset audit failed:\n" + "\n".join(report["issues"][:20]))
    for label in config["models"]:
        if (run_dir / label).exists():
            raise FileExistsError(f"Existing run directory: {run_dir / label}. Choose a new experiment name.")
    run_dir.mkdir(parents=True, exist_ok=True)
    for label, weights in config["models"].items():
        print(f"\n===== Training {label} ({weights}) =====", flush=True)
        model = YOLO(weights)
        args = dict(config["train"])
        args.update(data=config["data"], device=config["device"], project=str(run_dir), name=label, exist_ok=False)
        model.train(**args)
        manifest = {
            "model": label,
            "weights": weights,
            "dataset_sha256": report["dataset_sha256"],
            "dataset_audit": report,
            "configuration": config,
            "environment": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "ultralytics": ultralytics_version,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        }
        (run_dir / label / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def compare(
    config: dict,
    run_dir: Path,
    allow_partial: bool,
    output_dir: Path | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    missing = [label for label in config["models"] if not (run_dir / label / "results.csv").is_file()]
    if missing and not allow_partial:
        raise FileNotFoundError(f"Missing results.csv for {missing}. Use --allow-partial only for exploratory plots.")
    runs = {}
    hashes = set()
    verified = {}
    for label in config["models"]:
        csv = run_dir / label / "results.csv"
        if not csv.is_file():
            continue
        df = pd.read_csv(csv)
        df.columns = df.columns.str.strip()
        required = ["epoch", *METRICS]
        absent = [column for column in required if column not in df.columns]
        if absent or df.empty:
            raise ValueError(f"Invalid results.csv for {label}: missing {absent or 'rows'}")
        for column in required:
            df[column] = pd.to_numeric(df[column], errors="raise")
        if df[required].isna().any().any():
            raise ValueError(f"NaN in required columns for {label}")
        if not df["epoch"].is_unique or not df["epoch"].is_monotonic_increasing:
            raise ValueError(f"Duplicate or out-of-order epochs for {label}")
        if len(df) != int(config["train"]["epochs"]) and not allow_partial:
            raise ValueError(f"Incomplete {label} run: {len(df)} of {config['train']['epochs']} epochs")
        manifest_path = run_dir / label / "run_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            past = manifest["configuration"]
            if any(past[key] != config[key] for key in ("models", "train", "device")):
                raise ValueError(f"Run configuration differs from current config for {label}")
            hashes.add(manifest["dataset_sha256"])
            verified[label] = True
        else:
            print(f"[warning] {label} has no run_manifest.json; dataset identity is unverified")
            verified[label] = False
        runs[label] = df
    if len(hashes) > 1:
        raise ValueError("Run manifests have different dataset fingerprints")
    if not runs:
        raise FileNotFoundError("No results.csv files found")

    output_dir = output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([df.assign(model=label) for label, df in runs.items()], ignore_index=True)
    combined.to_csv(output_dir / "epochs_combined.csv", index=False)
    rows = []
    for label, df in runs.items():
        best = df.loc[df["metrics/mAP50-95(B)"].idxmax()]
        rows.append({
            "model": label,
            "best_epoch": int(best["epoch"]),
            "trained_epochs": len(df),
            "manifest_verified": verified[label],
            **{name: float(best[column]) for column, name in METRICS.items()},
        })
    table = pd.DataFrame(rows).round(4)
    table.to_csv(output_dir / "summary.csv", index=False)
    print("\n" + table.to_string(index=False))

    for column, (title, filename) in PLOTS.items():
        available = [(label, df) for label, df in runs.items() if column in df.columns]
        if not available:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label, df in available:
            ax.plot(df["epoch"], pd.to_numeric(df[column], errors="coerce"), label=label)
        ax.set(xlabel="Epoch", ylabel=column, title=title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)
    print(f"\nTables and plots: {output_dir}")


def _has_complete_history(run_dir: Path, labels: list[str]) -> bool:
    return run_dir.is_dir() and all((run_dir / label / "results.csv").is_file() for label in labels)


def _select_skip_train_run(config: dict, configured_run: Path) -> Path:
    """Select an existing history without changing the training target."""
    labels = list(config["models"])
    if _has_complete_history(configured_run, labels):
        return configured_run

    runs_root = ROOT / "runs_compare"
    candidates = [
        path for path in sorted(runs_root.iterdir())
        if path.is_dir() and path != configured_run and _has_complete_history(path, labels)
    ] if runs_root.is_dir() else []

    if len(candidates) == 1:
        selected = candidates[0]
        print(
            f"[info] Configured run {configured_run} is incomplete or missing; "
            f"using the only complete history: {selected}"
        )
        return selected
    if len(candidates) > 1:
        names = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Configured run {configured_run} is incomplete or missing. "
            f"Multiple complete histories exist: {names}. Set experiment to the intended run."
        )
    return configured_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", help="Override the dataset YAML in the config")
    parser.add_argument("--device", help="Override the device in the config, e.g. 0 or cpu")
    parser.add_argument("--skip-train", action="store_true", help="Only rebuild summary and plots")
    parser.add_argument("--check-data", action="store_true", help="Audit dataset and exit without training")
    parser.add_argument("--allow-partial", action="store_true", help="Analyze incomplete/missing runs for exploration")
    parser.add_argument("--output-dir", type=Path, help="Comparison output directory for --skip-train")
    opts = parser.parse_args()
    try:
        if opts.output_dir is not None and not opts.skip_train:
            raise ValueError("--output-dir requires --skip-train")
        config = load_config(opts.config, opts.data, opts.device)
        if opts.check_data:
            report = audit_dataset(Path(config["data"]))
            print(json.dumps(report, indent=2))
            return 1 if report["issues"] else 0
        configured_run_dir = ROOT / "runs_compare" / config["experiment"]
        run_dir = configured_run_dir
        output_dir = None
        if opts.skip_train:
            run_dir = _select_skip_train_run(config, run_dir)
            if opts.output_dir is not None:
                output_dir = opts.output_dir if opts.output_dir.is_absolute() else ROOT / opts.output_dir
            elif run_dir != configured_run_dir:
                output_dir = configured_run_dir
        if not opts.skip_train:
            train_all(config, run_dir)
        compare(config, run_dir, opts.allow_partial, output_dir)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

