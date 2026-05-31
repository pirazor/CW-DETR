# CW-DETR Training Readiness Report

Date: 2026-05-31

This report records verified evidence for the training-ready baseline work. It
does not claim measured accuracy or Jetson performance. Checkpoints, TensorRT
engines, gated DINOv3 weights, and generated profiles are intentionally not
committed.

## Branch Stack

The focused implementation branches are stacked in this order:

1. `docs/training-readiness-plan`
2. `codex/training-readiness-correctness`
3. `codex/training-readiness-eval`
4. `codex/training-readiness-deploy`
5. `codex/training-readiness-dn`

DN-DETR remains config-gated and default-off. It should only be enabled for long
runs after the baseline passes the fixed tiny-subset gate.

## Verified Locally

Environment used for the offline verification:

```text
platform=Windows-10-10.0.26200-SP0
python=3.11.9
torch=2.12.0+cpu
torchvision=0.27.0+cpu
scipy=1.17.1
pytest=9.0.3
pillow=12.2.0
numpy=2.4.6
pyyaml=6.0.3
onnx=1.21.0
onnxscript=0.7.0
onnxruntime=1.26.0
```

Commands:

```powershell
py -3.11 -m compileall -q cwdetr tests
git diff --check
py -3.11 -m pytest tests -q
```

Observed result:

```text
34 passed
```

The suite covers encoder-gradient flow, detector supervision filtering,
taxonomy validation, mocked ConvNeXt and ViT backbone contracts, teacher eval
persistence, unsafe ViT windowing rejection, synthetic metrics, checkpoint-best
selection, augmentation geometry, seeded sampler sharding, warmup, EMA,
deployment ROI handling, and DN query masking/loss behavior.

## ONNX Runtime Smoke

The sign sidecar was exported and executed off-device:

```powershell
$env:PYTHONPATH='C:\tmp\cwdetr-pydeps'
$env:PYTHONUTF8='1'
py -3.11 -m cwdetr.export.export_sign_sidecar `
  --config configs/cwdetr_nano_orin.yaml `
  --out C:\tmp\cwdetr-smoke\cwdetr_sign.onnx `
  --max-rois 4
```

Observed ONNX Runtime contract:

```text
inputs:  sign_features [1, 256, 48, 80], rois [4, 5]
output:  sign_logits [4, 43]
```

A compact mocked-backbone core graph was also exported and executed with ONNX
Runtime. Its observed output contract was:

```text
scores        [1, 20, 13]
boxes         [1, 20, 4]
drivable      [1, 16, 24]
lane          [1, 16, 24]
sign_features [1, 32, 8, 12]
```

The sidecar export emitted the expected ONNX `RoiAlign` adaptive-sampling
conversion warning and executed successfully in ONNX Runtime.

## Pending External Gates

The following acceptance work requires assets or hardware not available in the
offline environment:

1. Load gated DINOv3 weights and rerun the real Hugging Face and Meta backbone
   contract suite.
2. Run a fixed real-data tiny-subset overfit experiment, then run BDD100K and
   selected-sign-taxonomy validation evaluation.
3. Keep DN-DETR disabled unless its ablation reaches `mAP50 >= 0.90` in at most
   90 percent of baseline steps without validation regression.
4. Build and profile TensorRT engines on Jetson Orin for core FP16, mixed INT8
   with FP16 fallback, and sign-sidecar FP16. Capture the generated layer
   profiles because TensorRT `GridSample` uses floating-point I/O.

Reproducible training and evaluation commands:

```bash
python -m cwdetr.engine.train --config configs/cwdetr_nano_orin.yaml \
  --bdd-root /data/bdd100k --gtsrb-root /data/GTSRB --epochs 50
python -m cwdetr.engine.evaluate --config configs/cwdetr_nano_orin.yaml \
  --ckpt checkpoints/best_detection_map.pth --bdd-root /data/bdd100k \
  --gtsrb-root /data/GTSRB
```

Reproducible on-device profile commands:

```bash
python -m cwdetr.export.profile_jetson --core-engine cwdetr_nano_fp16.plan \
  --precision fp16 --out profiles/nano_fp16.json
python -m cwdetr.export.profile_jetson --core-engine cwdetr_nano_int8.plan \
  --precision int8 --out profiles/nano_mixed_int8.json
python -m cwdetr.export.profile_jetson --core-engine cwdetr_nano_int8.plan \
  --sign-engine cwdetr_sign_fp16.plan --precision int8 \
  --out profiles/nano_mixed_int8_sign.json
```
