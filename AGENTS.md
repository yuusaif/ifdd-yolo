# Agent instructions

## Read first

Before changing the experiment, read:

1. `README.md`
2. `skills/ifdd-replication/SKILL.md`
3. `config/experiment.yaml` (or the explicitly requested config)

The manuscript is evidence, not an operational specification.

## Project contract

- Default experiment: `config/experiment.yaml` → `ifdd_930_test_m`.
- Default dataset: `data/ifdd_930_boxes/data.yaml`, the immutable Roboflow v3 export converted from polygon rows to axis-aligned YOLO boxes.
- Models: `yolov8m.pt`, `yolo11m.pt`, `yolo12m.pt`, `yolo26m.pt`.
- Shared training settings: 30 epochs, 640 px, batch 4, seed 42, deterministic, SGD, cosine LR, and no overlapping online augmentation. Never use a model-specific batch.
- Classes and order: `Clip`, `Rivet`, `Screw`.
- `config/paper_30e_m.yaml` is a separate paper-style proposal; it is not the default run and is not ready.

## Safety and reproducibility

- Do not train, download datasets/checkpoints, or run validation/evaluation unless the user explicitly requests that action.
- Before training, audit the exact dataset and review its fingerprint, split counts, labels, duplicates, class order, and image metadata. Save the audit and environment details with the experiment.
- Keep all four models on the same audited dataset, split, configuration, software, and hardware. Record every deviation.
- Never overwrite an existing directory under `runs_compare/`. Preserve failed or partial runs and use a new experiment name for a restart.
- Do not put API keys or other secrets in source, notebooks, configs, or manifests. Local data, runs, weights, caches, and environments are Git-ignored.

## Authorized training runbook

From the project root, on the intended GPU host:

```bash
source .venv/bin/activate
export YOLO_CONFIG_DIR="$PWD/.ultralytics"
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
python yolo_compare.py --check-data
python yolo_compare.py
```

The `--check-data` command audits without training; the final command trains the four models sequentially. The script refuses an unready config or existing model run directory, writes each run manifest, and stores outputs in `runs_compare/<experiment>/<model>/`.

After all four runs complete:

```bash
python yolo_compare.py --skip-train
```

This rebuilds `epochs_combined.csv`, `summary.csv`, and comparison plots from complete `results.csv` files. These are validation-history summaries, not independent test results.

## Handoff requirements

Report the config, dataset fingerprint/version, split, class order, checkpoint-selection rule, thresholds, software/hardware versions, actual `args.yaml`, and any deviation. Do not call validation metrics test metrics. For false-negative claims, save per-image predictions and use fixed confidence/IoU/NMS settings with one-to-one matching.
