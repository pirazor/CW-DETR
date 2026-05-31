# CW-DETR: A DINOv3 Multi-Task Perception Model for ADAS

**ConnectedWise — Architecture & Strategy Whitepaper**
*Status: design + repaired baseline scaffold (untrained). Version 0.2.*

---

## 1. Executive summary

CW-DETR is an end-to-end perception model for advanced driver assistance. It takes a single
camera frame and, in one forward pass, produces object detection, multi-object tracking,
drivable-area and lane-line segmentation, fine-grained traffic-sign classification, and
multimodal trajectory prediction. It is built by taking Roboflow's **RF-DETR** — the current
state of the art for real-time detection transformers — and making two structural changes:

1. **Replace the DINOv2 backbone with DINOv3.** DINOv2 (patch-14, interpolated positional
   embeddings) becomes DINOv3 (patch-16, rotary position embeddings, register tokens,
   Gram-anchored dense features, and a fully-convolutional ConvNeXt option distilled from a
   7-billion-parameter teacher). This single change improves accuracy *and* latency, and it
   is the foundation everything else builds on.

2. **Turn the single-task detector into a shared-trunk, five-head multi-task model.** One
   backbone and one deformable-attention decoder run once per frame; five lightweight heads
   read the shared representation. The expensive part of the network is amortized across all
   tasks, which is what makes a complete perception stack feasible on a Jetson Orin Nano.

The model ships in two tiers: **CW-DETR-N** (DINOv3 ConvNeXt-Tiny, INT8) targeting the Jetson
Orin Nano Super, and **CW-DETR-B** (DINOv3 ViT-B/16, FP16) targeting the Jetson Orin NX 16GB.

This document explains the rationale, the architecture, the training strategy across three
datasets, the Jetson deployment path, and a phased roadmap. **All accuracy and latency
figures for CW-DETR in this document are engineering targets derived from the published
numbers of its components, not measured results**. The repository now carries an executable
offline baseline and front-camera nuScenes sample assembly, but temporal clip training and
on-device TensorRT profiling are still roadmap work.

---

## 2. Where we start: RF-DETR and DINOv2

RF-DETR (Roboflow, accepted to ICLR 2026) is a real-time detection/segmentation transformer.
Architecturally it combines three ideas: a **DINOv2-with-registers ViT backbone** run with an
RF-DETR windowed/global attention pattern; a **single-scale-to-multi-scale projector** in the
LW-DETR lineage (CSP/"C2f" fusion blocks); and a **Deformable-DETR decoder** with multi-scale
deformable cross-attention. A more recent paper frames the family in terms of weight-sharing
neural architecture search over that design space. The published detection variants:

| Variant | Resolution | Params | COCO AP | T4 latency (TRT FP16) | License |
|---|---|---|---|---|---|
| RF-DETR-Nano | 384² | 30.5M | 48.4 | 2.3 ms | Apache-2.0 |
| RF-DETR-Small | 512² | 32.1M | 53.0 | 3.5 ms | Apache-2.0 |
| RF-DETR-Medium | 576² | 33.7M | 54.7 | 4.4 ms | Apache-2.0 |
| RF-DETR-Large | 704² | 33.9M | 56.5 | 6.8 ms | Apache-2.0 |
| RF-DETR-2XL | 880² | 126.9M | 60.1 | 17.2 ms | PML-1.0 |

RF-DETR-2XL was the first real-time model to exceed 60 AP on COCO. The Nano–Large variants are
Apache-2.0, so CW-DETR builds on those to keep a permissive license on the detector side.

The relevant limitations we target: (a) the DINOv2 backbone is the latency bottleneck and uses
patch-14 with positional embeddings that must be interpolated whenever input resolution
changes; (b) RF-DETR is single-task; (c) it has no temporal/tracking or motion-forecasting
component. CW-DETR addresses all three.

---

## 3. Why DINOv3 is the right backbone swap

DINOv3 (Meta, August 2025) is a self-supervised vision foundation model trained on 1.7B
images, with a 7B-parameter ViT flagship and a family of distilled students. The lineup
(all ViTs are **patch-16, RoPE, 4 register tokens**):

| Family | Variant | Params | Embed dim | Notes |
|---|---|---|---|---|
| ViT | S / S+ | 21M / 29M | 384 | smallest; edge ViT tier |
| ViT | B | 86M | 768 | **CW-DETR-B backbone** |
| ViT | L / H+ | 300M / 840M | 1024 / 1280 | distillation teachers |
| ViT | 7B | 6.7B | 4096 | flagship teacher |
| ConvNeXt | T / S / B / L | 29 / 50 / 89 / 198M | — | distilled from ViT-7B |

Five properties make this the highest-leverage change we can make:

**Patch-16 instead of patch-14 → fewer tokens → lower latency.** ViT self-attention cost grows
with the square of the token count. At a fixed input size, a patch-16 grid has about 23% fewer
tokens than patch-14 (a 14²/16² ratio), so the attention and decoder cross-attention both get
cheaper for free, before any other optimization.

**Rotary position embeddings → native resolution flexibility.** DINOv2 used learned positional
embeddings that RF-DETR had to interpolate for every new input resolution, which both costs a
little accuracy and complicates the windowed-attention scheme. DINOv3's RoPE encodes position
relatively inside attention, so the same weights run at any resolution and any aspect ratio.
This matters for ADAS, where the useful field of view is wide (we train at 16:9-ish ratios such
as 384×640 and 512×896) and where multi-resolution training is a cheap accuracy lever.

**Gram anchoring → clean dense features.** DINOv3 introduces a Gram-anchoring loss that keeps
patch-to-patch feature similarity stable through long training, eliminating the high-norm
"artifact" tokens that degrade dense prediction. The result is unusually clean per-patch
features. Our segmentation head and our small-object detection both consume dense features
directly, so this is a direct, compounding accuracy gain — arguably the single biggest reason
to move to DINOv3 for a perception model rather than a pure classifier.

**ConvNeXt variants → INT8/TensorRT-friendly edge tier.** A ViT with windowed deformable
attention is not the easiest graph to quantize. DINOv3 also ships ConvNeXt backbones distilled
from the 7B ViT, which are fully convolutional, quantize to INT8 cleanly, and have a hierarchical
{stride 4, 8, 16, 32} pyramid that maps naturally onto a detection neck. CW-DETR-N uses
ConvNeXt-Tiny precisely because it is the realistic path to real-time INT8 on the Orin Nano.

**A built-in distillation teacher.** Because DINOv3's small models were themselves produced by
distillation from ViT-7B, we can keep distilling on *automotive* data: freeze a DINOv3 ViT-B (or
larger) teacher and add a feature-distillation loss on the student backbone. This transfers the
teacher's representational quality into the tiny edge backbone without any extra labels. CW-DETR
implements this as a cosine feature-distillation loss (`criterion.loss_distill`) gated by the
`gram_anchor_distill` config flag.

**License note.** RF-DETR Nano–Large are Apache-2.0; DINOv3 code and weights are released under
Meta's DINOv3 license. Meta's source repository is public, but pretrained checkpoint access still
requires accepting Meta's terms. CW-DETR supports both the official Meta repository backbone
factories with an approved checkpoint URL or local file and the Hugging Face Transformers path. Confirm the DINOv3
license terms apply to your product before shipping; the architecture itself is backbone-agnostic
and can fall back to a permissively-licensed ConvNeXt if required.

---

## 4. CW-DETR architecture

```
                 ┌──────────────────────────── shared trunk (runs once) ───────────────────────────┐
   image ─► DINOv3 backbone ─► multi-scale projector ─► deformable-DETR decoder ─► query embeddings
            (ConvNeXt-T / ViT-B)   (C2f PAN neck)        (two-stage, box-refine)        │
                          │ dense feature maps                                          │
                          ▼                                                              ▼
                 ┌────────────────┐   ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐
                 │ segmentation   │   │  detection   │  │ track queries │  │  sign class.  │  │  trajectory  │
                 │ lane+drivable  │   │  cls + box   │  │  (MOTR-style) │  │ (ROI cascade) │  │ (multimodal) │
                 └────────────────┘   └──────────────┘  └───────────────┘  └──────────────┘  └──────────────┘
```

The design principle is **one expensive trunk, many cheap heads**. The backbone and decoder
dominate FLOPs and latency; every head is a thin module reading either the shared decoder query
embeddings or the shared dense feature maps. Adding a task therefore adds milliwatts, not a
second network. This is the difference between "five models" and "one perception model."

**Backbone (`models/backbone/dinov3_backbone.py`).** Supports two interchangeable loaders:
`source: huggingface` uses `transformers.AutoBackbone` with an `AutoModel` fallback, while
`source: meta_hub` uses the cloned official `facebookresearch/dinov3` repo plus an approved
checkpoint URL or local file. Setting `pretrained: false` permits an ungated architecture-only
Meta-repository load for experiments, but it gives up the pretrained representation. For the ConvNeXt
tier the wrapper returns the native {8,16,32}-stride pyramid. For the ViT tier it takes the
stride-16 patch grid (dropping the CLS + 4 register tokens, per the confirmed
`[CLS|registers|patches]` token layout) and lifts it to a three-level pyramid with a ViTDet
"simple feature pyramid." ViT windowed attention is intentionally disabled in baseline configs:
the current scaffold is not RoPE-correct. Reintroduce local windows only as a measured follow-up
with per-window rotary positions.

**Projector (`models/projector.py`).** A light PAN-style neck with C2f fusion blocks unifies the
backbone maps to a common width (256 for N, 384 for B) and emits exactly the number of feature
levels the decoder samples (3). This is the RF-DETR/LW-DETR projector role, made backbone-agnostic.

**Decoder (`models/decoder/`).** A Deformable-DETR decoder cross-attends directly into the
flattened projector memory — the backbone+projector *are* the encoder, so there is no separate
deformable encoder (this is the RF-DETR single-scale choice, which is faster). It uses two-stage
query initialization (class-ranked proposals from the memory), iterative box refinement with
DINO `look_forward_twice`, and a `self_attn_mask` hook used by track queries and DN-DETR
denoising. Crucially the multi-scale deformable attention is the **pure-PyTorch `grid_sample`
implementation**, which is numerically identical to the CUDA op but exports to ONNX opset-16+ and
is available in recent TensorRT releases. The exact JetPack/TensorRT parser version must be
validated on the target board before claiming a plugin-free deployment.

### 4.1 Heads

**Detection** (`heads/detection_head.py`) is a standard DETR class + box head, with the per-layer
box MLP *shared* with the decoder so refinement and supervision use the same boxes. The unified
13-class taxonomy (`data/taxonomy.py`) spans BDD100K and nuScenes; `traffic_sign` is one coarse
class deliberately, because fine sign typing is delegated to a dedicated head.

**Tracking** (`heads/track_query_head.py`) is query-based (MOTR/MOTRv2 style). Each frame, the
surviving objects' decoder queries are propagated as "track queries" prepended to the fresh object
queries; a track query keeps locking onto the same object across frames, so identity is implicit
and no IoU/Re-ID matching is needed in the end-to-end path. A Query Interaction Module refines
surviving embeddings between frames. A deterministic **ByteTrack** associator
(`tracking/bytetrack.py`) is provided as the alternative deploy path when a transparent,
certifiable tracker is preferred over the learned one.

**Segmentation** (`heads/segmentation_head.py`) is a semantic-FPN over the projector maps with two
sibling predictors: drivable area ({background, direct, alternative}, BDD100K) and lane lines
(binary or typed). This head benefits most from DINOv3's Gram-anchored dense features.

**Traffic-sign sub-classification** (`heads/sign_classification_head.py`) is a *cascade*: the
detector finds *where* signs are, and this head ROI-aligns crops from a shared feature map to
decide *which* sign (GTSRB-43 on the Nano tier, Mapillary ~400 on the Base tier). Decoupling
"where" from "what" keeps the detector's class space tiny and stable while still delivering
fine-grained signs, and it reuses backbone features so the extra cost is a small ROI head, not a
second classifier network. The ROI head uses a channel bottleneck, depthwise-separable residual
blocks, and pooled spatial features so it remains lightweight on the Nano tier.

**Trajectory** (`heads/trajectory_head.py`) conditions on a tracked object's decoder embedding and
its short motion history (from the tracker) to predict `num_modes` future paths plus a probability
over modes, trained winner-takes-all to avoid mode collapse (Wayformer/MTR-lite). Image-plane on
the Nano tier; BEV/ego-frame on the Base tier with nuScenes.

---

## 5. Multi-task training strategy

**Three datasets, one head each where labels exist.** BDD100K supplies detection + drivable-area
+ lane lines (100k driving images with all three). nuScenes supplies tracking identities + future
trajectories (and detection). GTSRB / Mapillary Traffic Sign supply fine sign labels. No single
dataset has every label, so we draw **dataset-homogeneous batches** (`MixedBatchSampler`): each
batch is pure BDD100K, or pure nuScenes, or pure signs. The criterion simply skips tasks whose
labels are absent in the current batch.

**Uncertainty-weighted losses.** Hand-tuning five loss coefficients across three datasets is
brittle. We use homoscedastic uncertainty weighting (Kendall et al., 2018): each task carries a
learnable log-variance and contributes `0.5·exp(−s)·L + 0.5·s`. The network learns to balance
tasks and to down-weight a task on batches where it is uncertain, which interacts well with the
mixed-batch scheme (`criterion.UncertaintyWeighter`).

**Feature distillation.** With `gram_anchor_distill` enabled, a frozen DINOv3 ViT-B teacher
provides dense features and the student backbone is pulled toward them with a cosine loss at
stride 16. This is how the tiny ConvNeXt-Tiny edge backbone inherits the 7B teacher's quality on
driving imagery.

**Convergence aids.** Two-stage query init and iterative box refinement (already in the decoder)
plus optional DN-DETR-style denoising groups (the decoder exposes the `self_attn_mask` hook for
this) give DETR-fast convergence. The backbone trains at 0.1× the head learning rate
(`build_param_groups`) because it starts from strong pretrained weights.

**Suggested schedule.** Phase 1: freeze backbone, train projector + decoder + detection head on
BDD100K + nuScenes detection until detection AP plateaus. Phase 2: unfreeze backbone at 0.1× LR,
add segmentation and sign heads. Phase 3: enable track queries (clip-based training, 2–5 frame
clips) and the trajectory head on nuScenes. Phase 4: turn on distillation and do multi-resolution
fine-tuning. This staged approach keeps early training stable and isolates regressions per task.

The current trainer covers frame-batch detection, segmentation, signs, and distillation. The
nuScenes loader emits track IDs and masked future trajectories, but Phase 3 still requires a
clip sampler and a temporal training loop that propagates track queries across adjacent frames.

---

## 6. Jetson deployment and the latency budget

**Two engines, two precisions.** CW-DETR-N exports to an INT8 TensorRT engine for the Orin Nano
Super (67 INT8 TOPS, 8GB LPDDR5, 102 GB/s). CW-DETR-B exports to an FP16 engine for the Orin NX
16GB (100→157 TOPS in Super Mode). The export path is `export/export_onnx.py` →
`export/build_tensorrt.py`; the runtime is `export/jetson_infer.py`.

**Why this quantizes well.** The ConvNeXt-Tiny backbone is fully convolutional (INT8-friendly).
The deformable attention is `grid_sample`-based, so it parses with the stock ONNX→TensorRT path.
INT8 uses entropy calibration over ~500 driving frames; precision-sensitive layers (the deformable
sampling and the final detection/segmentation logits) are kept in FP16 via the FP16-fallback flag.

**Projected latency budget — CW-DETR-N @ 384×640, Orin Nano Super, INT8, batch 1.** These are
engineering estimates, to be replaced by measured numbers after training and `trtexec` profiling:

| Component | Est. time | Reasoning |
|---|---|---|
| DINOv3 ConvNeXt-Tiny backbone | ~11 ms | ~29M-param conv net, INT8, 384×640 |
| Projector (PAN/C2f) | ~3 ms | a few conv blocks at 256-d |
| Deformable decoder (3 layers, 300 q) | ~9 ms | deformable sampling dominates |
| Detection + segmentation heads | ~4 ms | seg upsample is the larger piece |
| Sign ROI head (when signs present) | ~1 ms | per-ROI, sparse |
| **Total** | **~28 ms (~35 FPS)** | trajectory/track run on host, off the critical path |

For comparison, YOLOP runs its three tasks at 23 FPS on the older Jetson TX2; CW-DETR-N targets a
faster, more accurate, and *broader* (five-task) result on the Orin Nano. CW-DETR-B on the Orin NX
trades latency for accuracy (target ~22 FPS at 512×896, FP16). The trajectory and tracking updates
are cheap host-side operations that run between frames and do not sit on the detection critical
path.

---

## 7. Expected performance vs. baselines

Framed as **design targets** against the public baselines CW-DETR is meant to surpass:

| Task | Baseline (public) | CW-DETR design target |
|---|---|---|
| Detection (BDD100K) | YOLOP 76.5 mAP50; YOLOPv2 higher | ≥ RF-DETR-class AP, well above YOLOP, via DINOv3 features |
| Drivable area (BDD100K) | YOLOP 91.5 mIoU; YOLOPv2 ~93 | ≥ 93 mIoU (Gram-anchored dense features) |
| Lane lines (BDD100K) | YOLOP 26.2 IoU; YOLOPv2 ~87% acc | sharp lane masks from clean dense features |
| Tracking (nuScenes/MOT) | ByteTrack strong baseline | competitive AMOTA via track queries; ByteTrack fallback |
| Sign classification | GTSRB >99% on crops | high top-1 via ROI cascade |
| Trajectory | constant-velocity / Wayformer | multimodal minADE improvement over CV baseline |

The accuracy thesis is concentrated in the backbone: RF-DETR already hits 48–56 AP on COCO with a
DINOv2 backbone in the 30M-parameter class, and DINOv3 improves both global and (especially) dense
features over DINOv2, so a DINOv3-backed detector + segmentation stack should move up on every
dense task while getting *faster* from patch-16. The remaining gains come from multi-task
co-training (tasks regularize each other) and distillation.

---

## 8. Phased roadmap

**Phase 0 — reproduce (2–3 wks).** Clone RF-DETR + DINOv3 (`setup_clone.sh`), reproduce RF-DETR-N
on COCO, stand up BDD100K/nuScenes/GTSRB loaders, pass `tests/test_forward.py`.

**Phase 1 — DINOv3 detector (3–4 wks).** Swap in the DINOv3 backbone; train detection on
BDD100K+nuScenes; confirm the patch-16 latency win and an AP improvement over the DINOv2 baseline.

**Phase 2 — segmentation + signs (3–4 wks).** Add segmentation and sign heads; co-train; verify the
dense-feature accuracy thesis on drivable/lane.

**Phase 3 — temporal (4–6 wks).** Clip-based training of track queries; trajectory head on
nuScenes; integrate ByteTrack fallback; MOTA/IDF1 + minADE evaluation.

**Phase 4 — edge (3–4 wks).** INT8 calibration, TensorRT engine builds, on-device profiling on Orin
Nano/NX, distillation fine-tune, NAS sweep over decoder depth/width/queries per latency target.

**Phase 5 — hardening.** Multi-resolution and night/adverse-weather robustness, failure-case mining,
safety-case documentation for the tracker.

---

## 9. Risks and mitigations

The hardest accuracy risk is windowed attention on the ViT tier interacting with RoPE; mitigation
is to prefer the ConvNeXt tier on the edge (no attention to window) and, for the ViT tier, to wire
per-window RoPE using the official `facebookresearch/dinov3` blocks rather than monkeypatching HF.
The hardest systems risk is INT8 accuracy loss on the deformable sampling; mitigation is the
FP16-fallback for sensitive layers, already wired. The hardest data risk is that no single dataset
carries all labels; mitigation is the mixed-batch sampler + uncertainty weighting, plus
pseudo-labeling (run the trained detector to mine sign crops in BDD100K, etc.). The licensing risk
(DINOv3 terms) is mitigated by the backbone-agnostic interface.

---

## 10. Repository map

```
CW-DETR/
  configs/                cwdetr_nano_orin.yaml (INT8) · cwdetr_base_nx.yaml (FP16)
  cwdetr/
    config.py             typed config loader
    models/
      backbone/           DINOv3 wrapper + windowed attention + ViTDet simple-FPN
      projector.py        PAN/C2f multi-scale neck
      decoder/            deformable attention (grid_sample) + decoder + two-stage glue
      heads/              detection · track-query · segmentation · sign · trajectory
      cwdetr.py           top-level model (shared trunk -> 5 heads)
      matcher.py          Hungarian matcher
      criterion.py        multi-task loss + uncertainty weighting + distillation
    tracking/bytetrack.py deterministic associator (deploy fallback)
    data/                 BDD100K · nuScenes · sign datasets · mixed sampler · transforms
    engine/train.py       multi-task trainer (AMP, param groups, distillation)
    export/               ONNX export · TensorRT build (INT8/FP16) · Jetson runtime
  tests/test_forward.py   offline shape/forward sanity tests (dummy backbone)
  setup_clone.sh          clone rf-detr + dinov3, install, investigate the DINOv2 seam
```

Run the sanity tests with `python -m tests.test_forward --config configs/cwdetr_nano_orin.yaml`.
Train with `python -m cwdetr.engine.train --config configs/cwdetr_nano_orin.yaml --bdd-root ...`.
Use `docs/CW-DETR_Jetson_Improvement_Plan.md` for the measured optimization sequence and gates.

---

## 11. References

RF-DETR (Roboflow; ICLR 2026) and *RF-DETR: Neural Architecture Search for Real-Time Detection
Transformers* (arXiv:2511.09554). DINOv3 (Meta, 2025; arXiv:2508.10104) and the
`facebook/dinov3-*` model card family. Deformable DETR (Zhu et al., 2021). DINO (Zhang et al.,
2022). LW-DETR. ViTDet simple feature pyramid (Li et al., 2022). MOTR / MOTRv2 (query-based
tracking). ByteTrack (Zhang et al., 2022). YOLOP (MIR 2022) and YOLOPv2 (arXiv:2208.11434).
Multi-task uncertainty weighting (Kendall, Gal & Cipolla, 2018). Wayformer / MTR (motion
forecasting). NVIDIA Jetson Orin Nano Super and Orin NX module datasheets.

---

*Targets in this document are projections from component baselines, not measured CW-DETR results.*
