# CW-DETR Training-Readiness & Correctness Plan

**Purpose.** Make the architecture *correct and measurable* before any serious training run.
This plan is the prerequisite to [`CW-DETR_Jetson_Improvement_Plan.md`](CW-DETR_Jetson_Improvement_Plan.md):
that doc's experiment order assumes the model trains all intended signals and that metrics exist.
Today neither is true. Work the tasks below in order; each is scoped for a coding agent and ends
with an explicit acceptance check.

> Status of the codebase: correct forward graph for **detection + segmentation + sign**; tracking
> and trajectory heads are present but receive **no gradient** from the shipped trainer; there is
> **no evaluation/metrics code**; the **two-stage proposal head is unsupervised**; DN-DETR
> denoising is configured but unimplemented.

## Guiding rules for the implementer
1. **One task = one PR.** Keep the existing `tests/test_forward.py` green at every step.
2. **Every fix ships with a test.** Prefer a unit test that would fail on the current code.
3. **Do not "fix" things that already work** (see *Do-not-touch* list at the end).
4. **No measured claim without the eval harness** (Workstream B). Until B lands, treat all runs as
   smoke tests only.
5. Keep changes config-gated where behavior changes, so ablations are possible.

---

## Workstream A — Correctness blockers (must land before training)

### A1 — Supervise the two-stage proposal head  ★ highest priority
**Problem.** In [`deformable_decoder.py`](../cwdetr/models/decoder/deformable_decoder.py) the
two-stage path detaches both the seed content (`tgt`) and reference (`ref`), and `enc_objectness`
feeds only a non-differentiable `topk`. There is **no encoder auxiliary loss** in
[`criterion.py`](../cwdetr/models/criterion.py). Result: `enc_output`, `enc_bbox`,
`enc_objectness` never get gradients → query selection is ranked by a *random* head. The model
pays two-stage compute for none of the benefit and converges like a vanilla DETR.

**Change.**
- In `DeformableTransformer.__init__`, replace `enc_objectness` (1 logit) with
  `self.enc_class_embed = nn.Linear(d, num_classes)`. Pass `num_classes` in from the model:
  `DeformableTransformer(m.decoder, num_classes=m.heads.detection.num_classes)` in
  [`cwdetr.py`](../cwdetr/models/cwdetr.py).
- In the two-stage block, compute class logits over **all** proposals, rank top-k by
  `max(-1)` class logit (replacing objectness), then gather the **non-detached** selected
  predictions for the auxiliary loss while keeping `tgt`/`ref` detached as the decoder seed:
  ```python
  enc_logits = self.enc_class_embed(output_memory)        # [B, L, num_classes]
  coord      = (self.enc_bbox(output_memory) + output_proposals)   # [B, L, 4]
  score      = enc_logits.max(-1).values.masked_fill(~valid[...,0], float("-inf"))
  topk_idx   = score.topk(topk, dim=1)[1]
  enc_logits_sel = gather(enc_logits, topk_idx)           # keep grad
  enc_coord_sel  = gather(coord,      topk_idx).sigmoid() # keep grad
  ref = enc_coord_sel.detach()                            # decoder seed (detached, standard)
  tgt = gather(output_memory, topk_idx).detach()
  enc_out = {"pred_logits": enc_logits_sel, "pred_boxes": enc_coord_sel}
  ```
- In `criterion.forward`, if `outputs.get("enc_outputs")` carries `pred_logits`, run the existing
  `loss_detection` on it and **add** it into the detection loss (weight ~1.0):
  `losses["detection"] = det["detection"] + enc["detection"]`.

**Acceptance.** New test: one forward+backward on the dummy-backbone model leaves
`transformer.enc_class_embed.weight.grad` and `transformer.enc_bbox` grads **non-None and
non-zero**. `tests/test_forward.py` still passes.

### A2 — Build the evaluation harness  ★ highest priority (you are blind without it)
**Problem.** There is no validation loop and no metrics anywhere in the repo. You cannot tell
training from divergence.

**Change.** Add `cwdetr/engine/evaluate.py` with a `@torch.no_grad` eval loop reporting:
- Detection: COCO mAP / mAP@50 via `pycocotools` (already a dependency) on a val split.
- Segmentation: drivable mIoU, lane IoU/F1.
- Sign: top-1 on the matched/teacher-forced ROIs.
Wire a `--eval` path and an end-of-epoch eval call in [`train.py`](../cwdetr/engine/train.py).
Save **best-by-detection-mAP** checkpoint separately from the per-epoch dumps.

**Acceptance.** `python -m cwdetr.engine.evaluate --config ... --bdd-root ...` prints a metrics
table on a tiny subset without error; a unit test runs eval on synthetic outputs and checks the
metric dict keys/shapes.

### A3 — Verify the real backbone loads with correct strides
**Problem.** The HF `AutoBackbone` DINOv3 path ([`dinov3_backbone.py`](../cwdetr/models/backbone/dinov3_backbone.py))
reconciles channels but **not strides**; a wrong stage→stride mapping yields plausibly-shaped but
semantically wrong features with no crash. DINOv3 HF support is also version-sensitive and gated.

**Change.** After the first real forward in `DINOv3Backbone`, assert each returned map's spatial
size equals `H/stride, W/stride` for the configured `out_strides`; raise a clear error otherwise.
Pin `transformers`/`timm` versions in `requirements.txt`. Add a short README note on weight access.

**Acceptance.** A guard test (skipped if weights unavailable) confirms the assertion fires on a
deliberately wrong stride config and passes on the correct one.

### A4 — Stop sign-crop datasets from polluting the detector
**Problem.** GTSRB samples ([`traffic_signs.py`](../cwdetr/data/traffic_signs.py)) inject a
full-frame `traffic_sign` detection box for an upscaled 32px crop. This trains the *detector* on a
distorted distribution.

**Change.** For crop-style sign datasets, emit an **empty** detection target (keep only
`sign_boxes`/`sign_labels` for the sign head), or add a per-sample `train_detection: False` flag the
criterion respects. Keep Mapillary (real in-context boxes) contributing to detection.

**Acceptance.** Test: a GTSRB-style sample produces zero detection-loss contribution but a non-zero
sign-loss contribution.

---

## Workstream B — Trainability & generalization (land with or right after A)

### B1 — Real augmentation
**Problem.** Augmentation is only Resize + HFlip ([`transforms.py`](../cwdetr/data/transforms.py)).
The heads/decoder train from scratch and will overfit / underperform.
**Change.** Add (config-gated) scale jitter / large-scale-jitter, photometric (brightness/contrast/
hue), and optionally random crop that correctly updates boxes + masks + sign boxes. Consider an
albumentations path as the docstring already suggests.
**Acceptance.** Transform unit tests verify boxes/masks/sign-boxes stay consistent under each new op.

### B2 — Optimizer schedule & EMA
**Change.** Add LR warmup (linear, ~500–1000 steps) before the existing cosine schedule; add model
EMA for eval; expose `--warmup-steps`, `--ema-decay`. Keep the backbone 0.1× LR group.
**Acceptance.** Smoke run shows warmup in the logged LR; EMA weights load and eval cleanly.

### B3 — Logging & reproducibility
**Change.** Use the already-declared `tensorboard` dep: log per-task losses, LR, and eval metrics.
Seed `random`/`numpy`/`torch`; make `MixedBatchSampler` epoch-seeded
([`multitask_dataset.py`](../cwdetr/data/multitask_dataset.py)) so runs are reproducible/DDP-safe.
**Acceptance.** Two seeded runs of N steps produce identical loss curves; TB scalars appear.

---

## Workstream C — Convergence accelerators (after A; before long runs)

### C1 — Implement (or remove) DN-DETR denoising
**Problem.** `box_noise_scale` is configured and docstrings cite denoising + the `self_attn_mask`
hook, but it is **unimplemented**: `self_attn_mask` is always `None`, no DN groups, no DN loss.
**Change.** Either (a) implement DN-DETR: build noised GT query groups, the block-diagonal
`self_attn_mask` that isolates them, and a DN reconstruction loss; or (b) delete the dead config +
docstring claims and budget a longer schedule. **(a) is recommended** — with A1 it restores the
"fast convergence" the design depends on.
**Acceptance.** With DN on, detection mAP reaches a fixed threshold in materially fewer epochs than
with DN off (record the ablation).

---

## Workstream D — Temporal enablement (defer until detection+seg+sign baseline converges)

The nuScenes loader **already provides** `track_ids`, `future`, `future_mask` — so this is wiring,
not data work.

### D1 — Clip sampler + temporal training loop
Add a 2–5 frame clip sampler; in the loop, run frame *t*, build `TrackInstances` via
`step_track`, and feed them (plus `traj_history` assembled from matched past boxes) into frame
*t+1*'s forward. Currently [`train.py`](../cwdetr/engine/train.py) calls
`model(images, sign_rois=...)` only — tracking/trajectory get no gradient.

### D2 — Track-identity loss
Add a contrastive/ID loss on track-query embeddings (none exists in the criterion today). Supervise
identity preservation with `track_ids`.

### D3 — Activate the trajectory loss
Pass `traj_history` so `outputs["trajectory"]` is produced; the criterion's winner-takes-all
trajectory loss then becomes active. Validate minADE/minFDE against a constant-velocity baseline.

**Acceptance.** Tracking beats the ByteTrack baseline on a held-out clip set (IDF1/MOTA); trajectory
beats constant-velocity (minADE).

---

## Workstream E — Deployment de-risk (do the smoke test NOW; full work later)

### E1 — Export smoke test before investing in training
**Why now:** the deformable attention is `grid_sample`-based; `GridSample` does **not** run in INT8
and TensorRT support is version-sensitive. If the graph won't parse/run on your Orin, that changes
the architecture, so learn it before training.
**Change.** Export the **untrained** nano model and run `trtexec` on the target Orin for FP16 and
INT8; record whether `GridSample` parses, the layer profile, and FPS. Feed results back into the
Jetson plan's speed workstream.
**Acceptance.** A recorded `trtexec` profile (or a documented failure mode + mitigation) for the
untrained graph on-device.

---

## Recommended execution order

```
A1  two-stage aux loss        ─┐ correctness
A3  backbone stride assert     │  (press-train blockers)
A4  sign/detector decoupling   │
A2  eval harness              ─┘  ← you cannot measure anything without this
B1  augmentation
B2  schedule + EMA
B3  logging + repro
E1  export smoke test (parallel, cheap, high-information)
C1  DN denoising
──────────────  baseline: detection + seg + sign, measured ──────────────
D1–D3  temporal (tracking + trajectory)
→ then proceed to CW-DETR_Jetson_Improvement_Plan.md experiment order
```

**Minimum set to press "train" on a meaningful detection+seg+sign baseline:** A1, A2, A3, A4
(+ B1/B2 strongly recommended).

## Definition of "training-ready"
- `tests/test_forward.py` green, plus new tests for A1, A2, A4, B1.
- A1 verified by gradient test; A2 prints real mAP/mIoU/top-1 on a val split.
- One real-data overfit-a-tiny-subset run drives detection loss → near zero and mAP up (proves the
  full loop learns).
- E1 profile recorded (so you don't train into a dead-end export).

## Do-not-touch (already correct — don't let the agent "fix" these)
- nuScenes 2D boxes + `track_ids`/`future`/`future_mask` emission ([`nuscenes.py`](../cwdetr/data/nuscenes.py)).
- Sign-box rescaling on resize ([`transforms.py`](../cwdetr/data/transforms.py)).
- MSDeformAttn math, Hungarian matcher, iterative refinement wiring, focal-bias init, config
  dimension validation — all verified correct.
