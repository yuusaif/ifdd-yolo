"""Train the four IFDD detectors, or rebuild comparisons from completed runs.

Training never starts during import. Pass --skip-train to analyze existing runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_all(config: dict, run_dir: Path) -> None:
    if config.get("training_ready") is not True:
        raise RuntimeError("Training is locked: set training_ready: true only after dataset and settings are frozen")
    if config.get("augmentation_strategy") in (None, "undecided"):
        raise ValueError("Choose and record an augmentation_strategy before training")
    spotcheck = config.get("preflight", {}).get("box_conversion_spotcheck", {})
    record_path = ROOT / spotcheck.get("record", "data/ifdd_930_v4_boxes/box_spotcheck_review.json")
    if spotcheck.get("status") != "complete" or not record_path.is_file():
        raise RuntimeError(
            "Training is locked until the 30-image polygon-to-box spot-check is recorded "
            f"in {record_path}; review angled clips and mark preflight complete in each matrix config."
        )
    review_record = json.loads(record_path.read_text())
    if (int(review_record.get("samples_reviewed", 0)) < 30
            or len(review_record.get("records", [])) < 30
            or int(review_record.get("angled_clip_samples", 0)) < 1
            or review_record.get("dataset_sha256") != json.loads(
                (ROOT / "data" / "ifdd_930_v4_boxes" / "dataset_audit.json").read_text()
            ).get("dataset_sha256")):
        raise RuntimeError(
            "Box-conversion review record must cover at least 30 samples including an angled clip "
            "and match the current dataset fingerprint."
        )
    batch = config["train"].get("batch")
    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        raise ValueError("Choose a fixed positive integer train.batch before training")
    # Resolve and hash all checkpoints before starting any model. YOLO accepts
    # model names and may download missing weights implicitly; this workflow
    # requires all weights to be present locally and recorded in advance.
    checkpoint_paths = {}
    for label, weights in config["models"].items():
        weight_path = Path(weights)
        if not weight_path.is_absolute():
            weight_path = ROOT / weight_path
        weight_path = weight_path.resolve()
        if not weight_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint for {label} is missing: {weight_path}. "
                "Place the authorized checkpoint locally before training; no download was attempted."
            )
        digest = sha256_file(weight_path)
        checkpoint_paths[label] = {"path": str(weight_path), "sha256": digest}

    from ultralytics import YOLO, __version__ as ultralytics_version
    import torch

    report = audit_dataset(Path(config["data"]))
    if report["issues"]:
        raise RuntimeError("Dataset audit failed:\n" + "\n".join(report["issues"][:20]))
    if report["dataset_sha256"] != review_record["dataset_sha256"]:
        raise RuntimeError("Dataset changed after the box spot-check; re-audit and repeat the review before training")
    if run_dir.exists():
        raise FileExistsError(f"Existing experiment directory: {run_dir}. Choose a new experiment name.")
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "ultralytics": ultralytics_version,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (run_dir / "dataset_audit.json").write_text(json.dumps(report, indent=2))
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2))
    (run_dir / "study_manifest.json").write_text(json.dumps({
        "experiment": config["experiment"],
        "configuration": config,
        "dataset_sha256": report["dataset_sha256"],
        "checkpoint_files": checkpoint_paths,
        "environment": environment,
    }, indent=2))
    for label, weights in config["models"].items():
        print(f"\n===== Training {label} ({checkpoint_paths[label]['path']}) =====", flush=True)
        model = YOLO(checkpoint_paths[label]["path"])
        args = dict(config["train"])
        args.update(data=config["data"], device=config["device"], project=str(run_dir), name=label, exist_ok=False)

        selection = {"metric": "metrics/mAP50-95(B)", "value": None, "epoch": None}

        def save_best_map5095(trainer) -> None:
            metric_value = trainer.metrics.get("metrics/mAP50-95(B)")
            if metric_value is None:
                return
            metric_value = float(metric_value)
            if selection["value"] is None or metric_value > selection["value"]:
                source = Path(trainer.last)
                destination = Path(trainer.wdir) / "selected_by_val_map5095.pt"
                shutil.copy2(source, destination)
                selection.update(value=metric_value, epoch=int(trainer.epoch) + 1)
                record = {
                    "selection_metric": "metrics/mAP50-95(B)",
                    "direction": "maximize",
                    "tie_break": "earliest epoch",
                    "epoch_csv_value_one_based": selection["epoch"],
                    "metric_value": selection["value"],
                    "selected_checkpoint": destination.name,
                    "source_checkpoint": "last.pt at the selected epoch",
                }
                (destination.parent.parent / "checkpoint_selection.json").write_text(
                    json.dumps(record, indent=2)
                )

        model.add_callback("on_model_save", save_best_map5095)
        model.train(**args)
        if selection["value"] is None:
            raise RuntimeError(f"No validation-selected checkpoint was saved for {label}")
        selected_path = run_dir / label / "weights" / "selected_by_val_map5095.pt"
        manifest = {
            "model": label,
            "weights": checkpoint_paths[label],
            "dataset_sha256": report["dataset_sha256"],
            "dataset_audit": report,
            "configuration": config,
            "environment": environment,
            "checkpoint_selection": selection,
            "selected_checkpoint": {
                "path": str(selected_path),
                "sha256": sha256_file(selected_path),
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
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, ImportError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

