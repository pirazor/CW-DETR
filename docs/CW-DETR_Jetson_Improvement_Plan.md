# CW-DETR Jetson Orin Nano Improvement Plan

## Goal

Build a measured multi-head perception model for Jetson Orin Nano Super. The target is not a
paper claim or a projected FPS number. The target is the best reproducible accuracy-latency
point that satisfies an 8 GB memory budget and an end-to-end batch-1 deployment budget.

The current repository is a repaired baseline scaffold. It is ready for controlled experiments,
but it is untrained. No CW-DETR accuracy or Jetson latency result should be treated as measured
until it is recorded by the protocol below.

## Release Gates

Every candidate must report:

| Area | Required metrics |
|---|---|
| Detection | BDD100K mAP50 and mAP50:95, with small-object AP |
| Segmentation | drivable mIoU, lane IoU, lane F1 |
| Signs | crop top-1, in-context recall, end-to-end sign accuracy |
| Tracking | HOTA, IDF1, MOTA or AMOTA, ID switches |
| Trajectory | minADE, minFDE, miss rate at 6 seconds |
| Runtime | TensorRT batch-1 FPS, p50 and p95 latency, peak RSS, engine size, power mode |

Run latency measurements on the actual Orin Nano Super in 25 W mode after warmup. Record
TensorRT and JetPack versions, precision, input size, calibration set hash, and `trtexec`
layer profile. Keep PyTorch workstation timing separate from Jetson TensorRT timing.

## Baselines

1. Reproduce the repaired `cwdetr-nano` shape suite.
2. Train a detector-only CW-DETR-N baseline at `384x640`.
3. Add segmentation, then signs, then temporal heads one at a time.
4. Compare against an RF-DETR-N detector baseline and a practical multi-task baseline such as
   YOLOP or YOLOPv2 on the same data split and target device where export is possible.
5. Reject any architecture change that lacks an ablation against the immediately preceding
   candidate.

## Accuracy Workstream

### A. Stabilize the shared trunk

- Start with DINOv3 ConvNeXt-Tiny frozen for the first detection stage.
- Unfreeze progressively with a lower backbone learning rate.
- Use teacher distillation only after the detector-only baseline converges.
- Distill both stride-16 features and detection logits. Add segmentation-logit distillation after
  the dense heads are stable.

### B. Protect small objects and lane detail

- Benchmark an optional stride-4 detail branch used only by segmentation and sign ROI pooling.
- Keep the deformable decoder on `{8,16,32}` first; route stride-4 detail into detection only if
  small-object AP improves enough to justify the bandwidth cost.
- Mine hard examples for distant signs, motorcycles, pedestrians, night scenes, rain, glare, and
  partial occlusion.

### C. Finish temporal training

- Add a 2-to-5 frame clip sampler for nuScenes and BDD100K videos where available.
- Propagate track queries across adjacent frames during training.
- Supervise track-query identity preservation with track IDs.
- Train the trajectory head on matched tracked objects with masked future waypoints.
- Retain ByteTrack as the deterministic deployment baseline and compare it against learned track
  queries before enabling the learned tracker in production.

## Speed Workstream

### A. Measure before pruning

Profile the repaired baseline with `trtexec --dumpProfile`. Split latency into backbone,
projector, decoder, segmentation, and host postprocessing. Optimize the largest measured cost
first.

### B. Sweep the decoder budget

Run a Pareto sweep over:

| Knob | Values |
|---|---|
| Hidden width | `192`, `224`, `256` |
| Decoder layers | `2`, `3` |
| Object queries | `100`, `150`, `200`, `300` |
| Sampling points | `2`, `4` |
| Input size | `320x576`, `384x640`, `448x768` |

Use staged query counts: a low fixed budget for normal frames and a larger budget only when scene
density requires it, if TensorRT engine management remains simple enough.

### C. Quantize deliberately

- Build an FP16 reference engine first.
- Apply INT8 PTQ with a stratified calibration set covering weather, lighting, road type, and
  object density.
- Keep precision-sensitive sampling and final logits in FP16 only when the measured accuracy loss
  requires it.
- If PTQ loses too much accuracy, use quantization-aware fine-tuning for the ConvNeXt backbone,
  projector, and convolutional heads.

### D. Reduce work outside the network

- Keep the TensorRT engine batch-1 and static-shape for the first production path.
- Run sign ROI classification only when coarse sign detections exist.
- Run trajectory updates at a lower cadence than detection when metrics allow it.
- Use ByteTrack host updates asynchronously and avoid a PyTorch dependency in the TensorRT
  runtime.

## Experiment Order

1. Detector-only ConvNeXt-Tiny baseline.
2. Add segmentation and verify the shared trunk improves or preserves detection AP.
3. Add sign ROI classification and a stride-4 detail branch ablation.
4. Complete clip-based temporal training and compare learned queries with ByteTrack.
5. Export FP16, validate TensorRT parser compatibility, and record the first Orin profile.
6. Run the decoder/input Pareto sweep.
7. Calibrate INT8, run PTQ ablations, then QAT only if needed.
8. Freeze the best measured Orin Nano profile as `cwdetr-nano-v1`.

## Definition Of Done

`cwdetr-nano-v1` is complete only when a tagged checkpoint, config, calibration manifest, exported
ONNX model, TensorRT engine build log, evaluation report, and Orin Nano latency profile can be
reproduced from a clean checkout.
