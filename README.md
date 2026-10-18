# IFDD four-model comparison

This repository trains and compares four medium YOLO detectors on industrial fastener images:

`yolov8m.pt` · `yolo11m.pt` · `yolo12m.pt` · `yolo26m.pt`


## Current default experiment

| Item | Value |
| --- | --- |
| Config | `config/experiment.yaml` |
| Experiment name | `ifdd_930_v4_test_m` |
| Dataset | `data/ifdd_930_v4_boxes/data.yaml` |
| Dataset origin | Roboflow manufacturing-industrial-robot, version 4 |
| Dataset format | YOLO detection; polygon rows converted to enclosing axis-aligned boxes |
| Classes | `Clip`, `Rivet`, `Screw` |
| Training | 30 epochs, 640 px, shared batch 4, seed 42, deterministic SGD, cosine LR |
| Device | CUDA device 0; use the same GPU and software stack for all models |

The config currently has `training_ready: true`. The version 4 export was converted and passes the structural dataset audit. Its retained split has documented limitations: four training source groups have two variants instead of three, class representation differs across splits, and perceptual-hash screening found cross-split near-duplicate candidates. See `data/ifdd_930_v4_boxes/dataset_review.json` before interpreting results. The separate `config/paper_30e_m.yaml` proposal uses different settings and remains `training_ready: false`.

## Three-seed, two-scale study workflow

The prepared study contains 24 runs: four nano and four medium models at seeds 42, 0, and 1. Six generated configs live in `config/run_matrix/`; each config runs one scale/seed group of four models. All use the retained v4 split, 30 epochs, 640 px, shared batch 4, deterministic SGD, cosine LR, and offline Roboflow augmentation. The retained split is a documented deviation from the original request for a newly seeded, class-stratified split.

Training, checkpoint downloads, validation, and test evaluation are separate actions. Do not start them until authorized. Before training:

1. Review `data/ifdd_930_v4_boxes/dataset_review.json`. This checkout already has a completed 30-image polygon-to-box spot check, including 14 angled clips, in `data/ifdd_930_v4_boxes/box_spotcheck_review.json`; verify its fingerprint still matches the dataset audit before training. If a conversion is wrong, fix the derived dataset and re-audit before training.
2. All eight pretrained checkpoint files are already in their configured local paths. The three supplied nano files and their SHA-256 values are listed in the local `weights/checkpoint_sources.json`; `yolo_compare.py` rejects missing weights before training and never asks Ultralytics to fetch them implicitly. Run manifests also record checkpoint hashes.
3. Activate the environment and verify the intended GPU and data:

```bash
source .venv/bin/activate
export YOLO_CONFIG_DIR="$PWD/.ultralytics"
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
python yolo_compare.py --config config/run_matrix/nano_seed42.yaml --check-data
```

The trainer refuses to start until the review record matches the current dataset fingerprint and the six matrix configs are marked complete. Both checks pass in this checkout. `scripts/complete_box_spotcheck.py` is only needed if a new dataset conversion receives a fresh visual review; `scripts/build_run_matrix.py --write` can create the six configs in a fresh checkout and refuses to overwrite existing files. Run the following commands one at a time, checking each group’s outputs before continuing:

```bash
python yolo_compare.py --config config/run_matrix/nano_seed42.yaml
python yolo_compare.py --config config/run_matrix/medium_seed42.yaml
python yolo_compare.py --config config/run_matrix/nano_seed0.yaml
python yolo_compare.py --config config/run_matrix/medium_seed0.yaml
python yolo_compare.py --config config/run_matrix/nano_seed1.yaml
python yolo_compare.py --config config/run_matrix/medium_seed1.yaml
```

Each run writes to its own `runs_compare/<experiment>/` directory. The trainer refuses any existing experiment directory. It saves the exact dataset audit, software/hardware details, checkpoint hashes, study config, `args.yaml`, validation plots, and run manifests. Preserve failed or partial directories and use a new experiment name for any restart.

After all 24 model runs complete, select checkpoints by maximum validation mAP@0.5:0.95 (earliest epoch breaks ties):

```bash
python scripts/select_checkpoints.py
```

The training callback preserves only the best validation mAP@0.5:0.95 checkpoint as `weights/selected_by_val_map5095.pt` (earliest epoch wins ties), so it does not save 30 large epoch files per run. It also writes `checkpoint_selection.json`. `select_checkpoints.py` verifies the records against each `results.csv` and writes a study index. Ultralytics' own `best.pt` remains available; the named selected file is used by the locked evaluation workflow.

Build the three-seed mean ± SD table and final validation handoffs before test:

```bash
python scripts/summarize_seed_matrix.py --output-dir runs_compare/matrix_summary_v1
python scripts/validation_handoff.py --output-dir runs_compare/validation_handoff/ifdd_v4_three_seed
```

The validation handoff stores per-class precision/recall/AP tables and Ultralytics validation plots, including both confusion matrices and `BoxF1_curve.png`, for each run.

Calibrate one shared confidence and NMS threshold using validation predictions from all 24 runs. Matching IoU is fixed at 0.5. The script chooses maximum pooled validation recall, then minimum false positives, then higher confidence and NMS IoU. Use a new output name if a previous attempt created partial files:

```bash
python scripts/evaluate_locked_test.py calibrate \
  --output-dir runs_compare/validation_calibration/ifdd_v4_three_seed
```

Review `locked_thresholds.json`, confirm the selected shared thresholds, and do not tune them on test results. Only after every training run and validation decision is frozen, run the final test handoff once:

```bash
python scripts/evaluate_locked_test.py test \
  --thresholds runs_compare/validation_calibration/ifdd_v4_three_seed/locked_thresholds.json \
  --output-dir runs_compare/test_evaluation/ifdd_v4_three_seed
```

This runs `model.val(split="test", save_json=True)` and saves per-image predictions plus fixed-threshold one-to-one TP/FP/FN summaries. The JSON prediction files are the handoff for class confusions, McNemar, kappa, and image-level bootstrap analyses. Validation curves and test metrics must be reported separately. The confidence sweep on validation is saved during calibration; any test threshold sweep is descriptive only and cannot be used to select an operating point.

The 630-original vs. 300-new image comparison still needs a verified source-group mapping. Recover provenance from the earlier export/source records and create a JSON object keyed by test-image basename, with values `original_630` or `new_300`, for example `{"Clip-12_jpg.rf.a1b2.jpg":"new_300"}`. Pass it as `--source-groups path/to/source_groups.json` to the test command. The converted export alone does not preserve that provenance.

After the primary test result is frozen, the prepared benchmark and corruption commands are:

```bash
python scripts/benchmark_matrix.py --output-dir runs_compare/benchmarks/ifdd_v4_seed42
python scripts/create_corruption_copies.py --output-dir data/ifdd_930_v4_corruptions
python scripts/evaluate_corruptions.py \
  --corruption-dir data/ifdd_930_v4_corruptions \
  --thresholds runs_compare/validation_calibration/ifdd_v4_three_seed/locked_thresholds.json \
  --output-dir runs_compare/corruption_eval/ifdd_v4_three_seed
```

The benchmark reports the eight architecture/scale combinations once using seed-42 checkpoints, batch 1, five warmups, and three timed repeats of 25 validation images. The corruption builder creates blur, brightness up/down, noise, and JPEG copies at three severities, with labels unchanged; evaluation reuses the frozen confidence/NMS settings. The final test evaluator also saves FN and FP overlays for every run. CAM heatmaps still require selecting 4–6 representative images and a separately installed/recorded CAM implementation; the optional data-size ablation needs source-group-aware sampling, and the optimizer sensitivity run needs a separately reviewed YOLO26 config. Never use corrupted copies or ablations to tune primary test thresholds.

## Environment setup

The prepared environment uses Python 3.10, PyTorch `2.13.0+cu130`, torchvision `0.28.0+cu130`, and Ultralytics `8.4.177` (see `environment.lock.txt`). On a new compatible CUDA host:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
python -m pip freeze > environment.lock.txt
```

All eight checkpoint files are present at the paths in `config/run_matrix/*.yaml`. The four medium checkpoints are in the project root; `yolov8n.pt`, `yolo11n.pt`, `yolo12n.pt`, and `weights/yolo26n.pt` are the nano checkpoints. The first three were downloaded from the official Ultralytics assets release `v8.3.0`; their URLs and SHA-256 hashes are recorded locally in the Git-ignored `weights/checkpoint_sources.json`. The trainer rejects missing files and records checkpoint hashes in each run. The local RTX 3050 has 6 GiB VRAM; batch 4 is the shared tested setting. Re-run `scripts/probe_vram.py` if the GPU, PyTorch, image size, or model scale changes:

```bash
YOLO_CONFIG_DIR="$PWD/.ultralytics" .venv/bin/python scripts/probe_vram.py
```

## Dataset contract

Use one immutable export and one image-level split for all models. The current default uses Roboflow version 4: 930 source images, 651/140/139 train/validation/test source images, auto-orient, stretch to 640×640, horizontal and vertical flips, 90-degree rotation, blur up to 2.5 px, and 3× train augmentation. Keep Roboflow augmentation offline; the default config disables overlapping Ultralytics augmentation. Four train groups have only two exported variants, so the export contains 2,228 rather than the expected 2,232 files.

Before training, verify:

- `data.yaml` paths resolve and include train/validation/test as intended.
- Class order is exactly `Clip, Rivet, Screw`.
- Image/label counts, empty or missing labels, class IDs, normalized boxes, formats, sizes, color modes, and exact duplicate images are acceptable.
- The dataset SHA-256 fingerprint and Roboflow version are recorded.
- The export is inspected for box alignment and near-duplicate frames; byte hashes do not detect visual near duplicates. Known v4 perceptual-hash candidates and the user's decision to retain the split are recorded in `data/ifdd_930_v4_boxes/dataset_review.json`.

Audit commands:

```bash
python scripts/audit_dataset.py data/ifdd_930_v4_boxes/data.yaml
python yolo_compare.py --check-data
```

The v4 source export was downloaded under `~/Downloads/Manufacturing Industrial Robot.v4i.yolov8`; its converted local derivative is `data/ifdd_930_v4_boxes/`. The conversion script refuses to overwrite a destination:

```bash
python scripts/convert_yolo_polygons_to_boxes.py \
  "$HOME/Downloads/Manufacturing Industrial Robot.v4i.yolov8" data/ifdd_930_v4_boxes_new
```

Use a new destination for every derived dataset. `scripts/exclude_empty_label_images.py` similarly creates a new copy and leaves its source unchanged.

## Compare completed histories

After four complete 30-epoch runs:

```bash
python yolo_compare.py --skip-train
```

This reads the configured experiment directory. If it is missing and exactly one complete four-model history exists under `runs_compare/`, that history is used as input and the new summary files are written to the configured experiment directory. Use `--output-dir /tmp/ifdd-summary` when the repository is read-only. Summary values come from the epoch with the highest validation mAP@0.5:0.95. They are not test scores, fixed-threshold false-negative counts, or proof of zero missed detections.

For an independent test evaluation, freeze checkpoint selection and all thresholds on validation first. Use the same test split, confidence threshold, IoU matching threshold, NMS, inference path, and hardware for every model. Save per-image predictions and one-to-one matches, then report TP, FP, FN, class confusions, per-class metrics, and uncertainty intervals. The paper-style protocol has no independent test split and is validation-only.

## Reproducibility rules

- Keep one shared dataset, split, training config, Ultralytics version, and hardware for all four models.
- Record software/hardware versions, checkpoint hashes, dataset fingerprint, actual `args.yaml`, and deviations.
- Do not silently reduce batch size or change device for one model.
- Treat YOLO26’s architecture, pretrained-weight history, losses, and NMS behavior as model-specific limitations; identical fine-tuning settings do not make models identical.
- Report validation and test results separately, with the split and checkpoint-selection rule beside every result.

## Repository map

```text
config/experiment.yaml             default v4 930-source-image experiment
config/paper_30e_m.yaml             separate paper-style proposal
scripts/audit_dataset.py            read-only dataset audit and fingerprint
scripts/probe_vram.py               synthetic memory probe; does not train
scripts/build_run_matrix.py         create six scale/seed configs without overwriting
scripts/complete_box_spotcheck.py   validate review record and unlock matrix configs
scripts/select_checkpoints.py       select mAP50-95 checkpoint from saved epochs
scripts/summarize_seed_matrix.py    report per-seed and mean ± SD validation metrics
scripts/validation_handoff.py       produce per-class validation tables and plots
scripts/evaluate_locked_test.py     validation calibration and frozen test handoff
scripts/benchmark_matrix.py         batch-1 size/latency benchmark on seed-42 runs
scripts/create_corruption_copies.py make labeled corruption copies without source edits
scripts/evaluate_corruptions.py     test fixed checkpoints on corruption copies
scripts/convert_yolo_polygons_to_boxes.py
scripts/exclude_empty_label_images.py
yolo_compare.py                     train or compare completed histories
data/                               local datasets; ignored
runs_compare/                       local checkpoints and outputs; ignored
```
