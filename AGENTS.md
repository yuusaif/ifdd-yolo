# Agent instructions

## Read first

Before changing the experiment, read:

1. `README.md`
2. `skills/ifdd-replication/SKILL.md`
3. `config/experiment.yaml` (or the explicitly requested config)

The manuscript is evidence, not an operational specification.

## Project contract

- Default experiment: `config/experiment.yaml` → `ifdd_930_v4_test_m`.
- Default dataset: `data/ifdd_930_v4_boxes/data.yaml`, the Roboflow v4 export converted from polygon rows to axis-aligned YOLO boxes.
- Study matrix: nano and medium variants of YOLOv8, YOLO11, YOLO12, and YOLO26 at seeds 42, 0, and 1 (24 runs total).
- Shared training settings: 30 epochs, 640 px, batch 4, seeds 42/0/1, deterministic, SGD, cosine LR, and no overlapping online augmentation. Never use a model-specific batch.
- Classes and order: `Clip`, `Rivet`, `Screw`.
- The retained v4 split and known deviations are documented in `data/ifdd_930_v4_boxes/dataset_review.json`; preserve that record in run handoffs.
- `config/paper_30e_m.yaml` is a separate paper-style proposal; it is not the default run and is not ready.

## Safety and reproducibility

- Do not train, download datasets/checkpoints, or run validation/evaluation unless the user explicitly requests that action.
- Before training, audit the exact dataset and review its fingerprint, split counts, labels, duplicates, class order, and image metadata. Save the audit and environment details with the experiment.
- Keep all models on the same audited dataset, split, shared hyperparameters, software, and hardware. Record every deviation.
- Never overwrite an existing directory under `runs_compare/`. Preserve failed or partial runs and use a new experiment name for a restart.
- Do not put API keys or other secrets in source, notebooks, configs, or manifests. Local data, runs, weights, caches, and environments are Git-ignored.

## Authorized training runbook

From the project root, on the intended GPU host:

```bash
source .venv/bin/activate
export YOLO_CONFIG_DIR="$PWD/.ultralytics"
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
python yolo_compare.py --config config/run_matrix/nano_seed42.yaml --check-data
```

The six configs are already present. `scripts/build_run_matrix.py --write` generates them in a fresh checkout and refuses to overwrite existing files. Training must be explicitly requested. Before starting, review the retained split and its limitations, document at least 30 box checks including angled clips, and run `scripts/complete_box_spotcheck.py` to validate the record and unlock configs. Make all eight local checkpoints available. The trainer refuses missing checkpoints rather than allowing implicit downloads. Run each matrix config sequentially, with the same batch and all other shared settings:

```bash
python yolo_compare.py --config config/run_matrix/nano_seed42.yaml
python yolo_compare.py --config config/run_matrix/medium_seed42.yaml
python yolo_compare.py --config config/run_matrix/nano_seed0.yaml
python yolo_compare.py --config config/run_matrix/medium_seed0.yaml
python yolo_compare.py --config config/run_matrix/nano_seed1.yaml
python yolo_compare.py --config config/run_matrix/medium_seed1.yaml
```

Each experiment uses a unique directory. Never overwrite an existing path under `runs_compare/`; preserve partial runs and choose a new experiment name for a restart. Each run saves dataset audit/fingerprint, environment, checkpoint hashes, actual `args.yaml`, results, plots, and a manifest.

Once all 24 runs complete, select the maximum validation mAP@0.5:0.95 checkpoint (earliest epoch wins ties), then calibrate shared confidence/NMS settings on validation. Only then run the final test evaluation:

```bash
python scripts/select_checkpoints.py
python scripts/summarize_seed_matrix.py --output-dir runs_compare/matrix_summary_v1
python scripts/validation_handoff.py --output-dir runs_compare/validation_handoff/ifdd_v4_three_seed
python scripts/evaluate_locked_test.py calibrate --output-dir runs_compare/validation_calibration/ifdd_v4_three_seed
python scripts/evaluate_locked_test.py test --thresholds runs_compare/validation_calibration/ifdd_v4_three_seed/locked_thresholds.json --output-dir runs_compare/test_evaluation/ifdd_v4_three_seed
```

Test outputs are frozen-threshold metrics and per-image predictions. Do not use test results to revise thresholds or checkpoint selection. `yolo_compare.py --skip-train` still rebuilds validation-history summaries for one configured four-model group; it does not summarize test results or all six configs automatically.

## Handoff requirements

Report the config, dataset fingerprint/version, split, class order, checkpoint-selection rule, thresholds, software/hardware versions, actual `args.yaml`, and any deviation. Do not call validation metrics test metrics. For false-negative claims, save per-image predictions and use fixed confidence/IoU/NMS settings with one-to-one matching.
