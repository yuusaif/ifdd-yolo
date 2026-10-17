# IFDD four-model comparison

This repository trains and compares four medium YOLO detectors on industrial fastener images:

`yolov8m.pt` · `yolo11m.pt` · `yolo12m.pt` · `yolo26m.pt`


## Current default experiment

| Item | Value |
| --- | --- |
| Config | `config/experiment.yaml` |
| Experiment name | `ifdd_930_test_m` |
| Dataset | `data/ifdd_930_boxes/data.yaml` |
| Dataset origin | Roboflow manufacturing-industrial-robot, version 3 |
| Dataset format | YOLO detection; polygon export converted to enclosing axis-aligned boxes |
| Classes | `Clip`, `Rivet`, `Screw` |
| Training | 30 epochs, 640 px, shared batch 4, seed 42, deterministic SGD, cosine LR |
| Device | CUDA device 0; use the same GPU and software stack for all models |

The config currently has `training_ready: true`. Re-run and review the dataset audit before any new training. The separate `config/paper_30e_m.yaml` proposal uses a different dataset/settings and remains `training_ready: false`.

## One-command training workflow

Training, downloads, and validation require explicit user authorization. Once authorized, run from the project root:

```bash
source .venv/bin/activate
export YOLO_CONFIG_DIR="$PWD/.ultralytics"
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
python yolo_compare.py --check-data
python yolo_compare.py
```

`--check-data` is read-only. The final command audits again, trains all four models sequentially, rejects an existing model run directory, and writes a `run_manifest.json` for each completed run.

Outputs are written to:

```text
runs_compare/ifdd_930_test_m/<model>/
```

If any model fails, preserve its directory and diagnose it. Do not delete or overwrite it; restart with a new `experiment` name. Check each model’s `args.yaml` for the actual batch, device, and other runtime settings.

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

The four checkpoint files must be available in the project root before training. If they are missing, obtain the official pretrained files only after explicit authorization and record their SHA-256 hashes. The local RTX 3050 has 6 GiB VRAM; batch 4 is the shared tested setting. Re-run `scripts/probe_vram.py` if the GPU, PyTorch, image size, or model scale changes:

```bash
YOLO_CONFIG_DIR="$PWD/.ultralytics" .venv/bin/python scripts/probe_vram.py
```

## Dataset contract

Use one immutable export and one image-level split for all models. For the primary study, preserve the Roboflow v3 preprocessing/version: auto-orient, stretch to 640×640, horizontal and vertical flips, 90-degree rotation, blur up to 2.5 px, and the 3× training augmentation. Keep Roboflow augmentation offline; the default config disables overlapping Ultralytics augmentation.

Before training, verify:

- `data.yaml` paths resolve and include train/validation/test as intended.
- Class order is exactly `Clip, Rivet, Screw`.
- Image/label counts, empty or missing labels, class IDs, normalized boxes, formats, sizes, color modes, and exact duplicate images are acceptable.
- The dataset SHA-256 fingerprint and Roboflow version are recorded.
- The export is inspected for box alignment and near-duplicate frames; byte hashes do not detect visual near duplicates.

Audit commands:

```bash
python scripts/audit_dataset.py data/ifdd_930_boxes/data.yaml
python yolo_compare.py --check-data
```

The raw export is under `data/ifdd_930/`. The conversion script refuses to overwrite a destination:

```bash
python scripts/convert_yolo_polygons_to_boxes.py \
  data/ifdd_930 data/ifdd_930_boxes_new
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
config/experiment.yaml             default 930-image experiment
config/paper_30e_m.yaml             separate paper-style proposal
scripts/audit_dataset.py            read-only dataset audit and fingerprint
scripts/probe_vram.py               synthetic memory probe; does not train
scripts/convert_yolo_polygons_to_boxes.py
scripts/exclude_empty_label_images.py
yolo_compare.py                     train or compare completed histories
data/                               local datasets; ignored
runs_compare/                       local checkpoints and outputs; ignored
```
