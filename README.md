<div align="center">

# CW-DETR

**A DINOv3 multi-task perception model for ADAS — built on RF-DETR.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pirazor/CW-DETR/blob/main/notebooks/CW_DETR_Colab.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-green.svg)](#license)

*One camera frame in → detection, tracking, lane/area segmentation, sign classification, and trajectory prediction out — in a single forward pass.*

</div>

---

## What is this?

CW-DETR is an **end-to-end perception stack for advanced driver assistance**. It takes Roboflow's
[**RF-DETR**](https://github.com/roboflow/rf-detr) — the current state of the art for real-time
detection transformers — and makes two structural upgrades:

1. **DINOv2 → DINOv3 backbone.** Patch-16 (fewer tokens → faster), rotary position embeddings
   (native multi-resolution), **Gram-anchored dense features** (much cleaner masks and small
   objects), and a fully-convolutional **ConvNeXt** edge variant distilled from a 7B-parameter
   teacher (INT8/TensorRT-friendly). This single change improves accuracy *and* latency.

2. **One detector → one shared trunk + five task heads.** The expensive backbone and deformable
   decoder run **once per frame**; five lightweight heads share the representation. Five tasks for
   roughly the cost of one detector — which is what makes a full perception stack run on a Jetson
   Orin Nano.

> **Status:** repaired, runnable baseline scaffold — **not yet trained**. The offline suite
> shape-verifies the shared trunk, heads, losses, config loader, and tracker state machinery.
> End-to-end temporal clip training, RoPE-correct ViT windowing, and on-device TensorRT profiling
> remain roadmap work. Learned tracking and trajectory heads are disabled in the baseline configs
> until their temporal supervision lands. All
> accuracy / latency figures in the docs are engineering targets derived from component baselines,
> not measured CW-DETR results.

## Capabilities

| Head | Task | Approach |
|---|---|---|
| Detection | 2D objects, 13-class ADAS taxonomy | DETR class + box, iterative refinement |
| Tracking | Multi-object tracking + IDs | Query-based (MOTR-style) **+** ByteTrack fallback |
| Segmentation | Drivable area + lane lines | Semantic-FPN over DINOv3 dense features |
| Sign classification | Fine-grained sign type | ROI cascade off the detector (GTSRB / Mapillary) |
| Trajectory | Multimodal path prediction | Per-track motion decoder, winner-takes-all |

## Architecture

```
                 ┌───────────── shared trunk (runs once per frame) ─────────────┐
   image ─► DINOv3 backbone ─► multi-scale projector ─► deformable-DETR decoder ─┘
            (ConvNeXt-T / ViT-B)   (C2f / PAN neck)       (two-stage, box-refine)
                          │ dense maps                     │ query embeddings
        ┌─────────────────┴───────┬──────────────┬─────────┴───────┬──────────────┐
        ▼                         ▼              ▼                 ▼              ▼
   segmentation              detection     track queries      sign class.    trajectory
   lane + drivable           cls + box     (MOTR-style)       (ROI cascade)   (multimodal)
```

Full rationale, training plan, Jetson optimization, and roadmap:
**[`docs/CW-DETR_Architecture_and_Strategy.md`](docs/CW-DETR_Architecture_and_Strategy.md)**.

Pre-training correctness and readiness gates:
**[`docs/CW-DETR_Training_Readiness_Plan.md`](docs/CW-DETR_Training_Readiness_Plan.md)**.

Reproducible verification evidence and pending hardware/data gates:
**[`docs/CW-DETR_Training_Readiness_Report.md`](docs/CW-DETR_Training_Readiness_Report.md)**.

Measured improvement plan and acceptance gates:
**[`docs/CW-DETR_Jetson_Improvement_Plan.md`](docs/CW-DETR_Jetson_Improvement_Plan.md)**.

## Deployment tiers

| Tier | Backbone | Precision | Target board | Input | Design target |
|---|---|---|---|---|---|
| **CW-DETR-N** | DINOv3 ConvNeXt-Tiny | INT8 | Jetson Orin Nano Super (8 GB, 67 TOPS) | 384×640 | ~35 FPS |
| **CW-DETR-B** | DINOv3 ViT-B/16 | FP16 | Jetson Orin NX 16 GB (100–157 TOPS) | 512×896 | ~22 FPS |

## Quickstart

### ▶ Colab (easiest — no local setup, GPU optional)
Open [`notebooks/CW_DETR_Colab.ipynb`](notebooks/CW_DETR_Colab.ipynb) in Colab (badge above). It
installs deps, runs the shape sanity tests (no pretrained weights needed), and optionally builds the
real DINOv3 model after pretrained-weight access is configured.

### 💻 Local
```bash
bash setup_clone.sh          # clones rf-detr + dinov3, installs deps, inspects the DINOv2 seam
# Choose one pretrained-weight route:
huggingface-cli login        # default Transformers backend
# or set source: meta_hub in the YAML and provide an approved checkpoint path / URL:
# export DINOV3_BACKBONE_WEIGHTS=/path/to/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
# export DINOV3_TEACHER_WEIGHTS=/path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
# Set pretrained: false with source: meta_hub only for an ungated random initialization.

# sanity-check shapes end-to-end (dummy backbone, no download, runs on CPU)
python -m tests.test_forward --config configs/cwdetr_nano_orin.yaml

# train across datasets (provide the roots you have)
python -m cwdetr.engine.train --config configs/cwdetr_nano_orin.yaml \
    --bdd-root /data/bdd100k --nuscenes-root /data/nuscenes \
    --gtsrb-root /data/GTSRB --epochs 50

# validate a checkpoint (COCO mAP, drivable/lane metrics, teacher-forced sign top-1)
python -m cwdetr.engine.evaluate --config configs/cwdetr_nano_orin.yaml \
    --ckpt checkpoints/best_detection_map.pth --bdd-root /data/bdd100k \
    --gtsrb-root /data/GTSRB

# export + build Jetson engines (run TensorRT builds and profiling on-device)
python -m cwdetr.export.export_onnx    --config configs/cwdetr_nano_orin.yaml \
    --ckpt checkpoints/best_detection_map.pth --sign-features --out cwdetr.onnx
python -m cwdetr.export.export_sign_sidecar --config configs/cwdetr_nano_orin.yaml \
    --ckpt checkpoints/best_detection_map.pth --out cwdetr_sign.onnx
python -m cwdetr.export.build_tensorrt --onnx cwdetr.onnx --precision int8 \
    --calib-dir /data/calib --out cwdetr_nano_int8.plan
python -m cwdetr.export.build_tensorrt --onnx cwdetr_sign.onnx --precision fp16 \
    --out cwdetr_sign_fp16.plan
python -m cwdetr.export.jetson_infer   --engine cwdetr_nano_int8.plan \
    --sign-engine cwdetr_sign_fp16.plan --track
python -m cwdetr.export.profile_jetson --core-engine cwdetr_nano_int8.plan \
    --sign-engine cwdetr_sign_fp16.plan --precision int8 --out profiles/nano_int8.json
```

### YOLO-format BDD100K detection-only training

For a YOLO export with sibling `images/` and `labels/` directories, use the
nine-class detection-only config and pass its `data.yaml` directly:

```bash
python -m cwdetr.engine.train \
    --config configs/cwdetr_nano_yolo_bdd_detection.yaml \
    --yolo-data /data/bdd100k_merged/data.yaml --epochs 50
python -m cwdetr.engine.evaluate \
    --config configs/cwdetr_nano_yolo_bdd_detection.yaml \
    --yolo-data /data/bdd100k_merged/data.yaml \
    --ckpt checkpoints/best_detection_map.pth
```

The first run caches per-split image manifests and parsed labels beside
`data.yaml`, avoiding repeated recursive scans and per-sample label-file reads
on mounted Google Drive. Existing Ultralytics `labels.cache` files are imported
when available. Pass `--refresh-yolo-index` after adding, removing, or editing
images or labels. The adapter retries transient mounted-drive read failures.

Colab workflow:
**[`notebooks/CW_DETR_YOLO_Detection_Training_Colab.ipynb`](notebooks/CW_DETR_YOLO_Detection_Training_Colab.ipynb)**.

## Repository layout

```
CW-DETR/
├── configs/                 cwdetr_nano_orin.yaml (INT8) · cwdetr_base_nx.yaml (FP16)
├── cwdetr/
│   ├── config.py            typed YAML config loader
│   ├── models/
│   │   ├── backbone/        DINOv3 wrapper + windowed attention + ViTDet simple-FPN
│   │   ├── projector.py     PAN / C2f multi-scale neck
│   │   ├── decoder/         deformable attention (grid_sample, ONNX-friendly) + decoder
│   │   ├── heads/           detection · track-query · segmentation · sign · trajectory
│   │   ├── cwdetr.py        top-level model (shared trunk → 5 heads)
│   │   ├── matcher.py       Hungarian matcher
│   │   └── criterion.py     multi-task loss + uncertainty weighting + distillation
│   ├── tracking/            ByteTrack associator (deterministic deploy path)
│   ├── data/                BDD100K · nuScenes · sign datasets · mixed sampler · transforms
│   ├── engine/train.py      multi-task trainer (AMP, param groups, distillation)
│   └── export/              ONNX export · TensorRT build (INT8/FP16) · Jetson runtime
├── notebooks/               Colab notebook
├── tests/test_forward.py    offline shape/forward sanity tests (mocked backbone)
├── docs/                    architecture & strategy whitepaper
└── setup_clone.sh           clone rf-detr + dinov3, install, investigate the DINOv2 seam
```

## Design targets vs. baselines

| Task | Public baseline | CW-DETR target |
|---|---|---|
| Detection (BDD100K) | YOLOP 76.5 mAP50 | RF-DETR-class AP via DINOv3 features |
| Drivable area | YOLOP 91.5 mIoU / YOLOPv2 ~93 | ≥ 93 mIoU (Gram-anchored dense features) |
| Lane lines | YOLOP 26.2 IoU | sharper masks from clean dense features |
| Tracking | ByteTrack | competitive AMOTA via track queries |
| Signs | GTSRB >99% on crops | high top-1 via ROI cascade |

*(Targets are projections from component baselines, not measured CW-DETR results.)*

## License

The CW-DETR code in this repository is released under **Apache-2.0**. It builds on RF-DETR
(Nano–Large: Apache-2.0) and the **DINOv3** weights, which are distributed under **Meta's DINOv3
license**. Meta's source repository is public, while pretrained checkpoint access requires
accepting Meta's terms through either the official repository instructions or Hugging Face.
Review the DINOv3 license before shipping a product; the backbone interface is swappable if a
fully-permissive backbone is required.

## Acknowledgements

RF-DETR (Roboflow), DINOv3 (Meta AI), Deformable DETR, DINO, LW-DETR, ViTDet, MOTR/MOTRv2,
ByteTrack, YOLOP/YOLOPv2, nuScenes, BDD100K, GTSRB, Mapillary Traffic
