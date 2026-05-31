#!/usr/bin/env bash
# ============================================================================
# CW-DETR environment + reference-repo setup
#
# Clones the two upstream repos we build on top of:
#   1. roboflow/rf-detr        -> the real-time DETR we extend (Apache-2.0 core)
#   2. facebookresearch/dinov3 -> the foundation backbone we swap in
#
# It then runs a short "investigation" pass that prints exactly where RF-DETR
# wires in its DINOv2 backbone — the seam CW-DETR replaces with DINOv3.
#
# Usage:  bash setup_clone.sh
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="${ROOT}/third_party"
mkdir -p "${TP}"

echo "==> [1/4] Cloning upstream repositories into ${TP}"
if [ ! -d "${TP}/rf-detr" ]; then
  git clone --depth 1 https://github.com/roboflow/rf-detr.git "${TP}/rf-detr"
else
  echo "    rf-detr already present, skipping."
fi
if [ ! -d "${TP}/dinov3" ]; then
  git clone --depth 1 https://github.com/facebookresearch/dinov3.git "${TP}/dinov3"
else
  echo "    dinov3 already present, skipping."
fi

echo "==> [2/4] Creating virtual environment + installing deps"
python3 -m venv "${ROOT}/.venv"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
pip install --upgrade pip
pip install -r "${ROOT}/requirements.txt"
# Install rf-detr in editable mode so we can import its decoder/matcher utilities.
pip install -e "${TP}/rf-detr" || echo "WARN: rf-detr editable install failed (optional; we vendor what we need)."

echo "==> [3/4] INVESTIGATE: where does RF-DETR build its DINOv2 backbone?"
# RF-DETR's neural core is src/rfdetr/models/lwdetr.py + a backbone subpackage.
# These greps surface the DINOv2 construction sites we override with DINOv3.
echo "---- backbone module(s) ----"
find "${TP}/rf-detr/src/rfdetr/models" -maxdepth 2 -iname '*backbone*' -o -iname '*dinov2*' 2>/dev/null || true
echo "---- DINOv2 references ----"
grep -rni --include='*.py' -e 'dinov2' -e 'windowed' -e 'register_token' \
     "${TP}/rf-detr/src/rfdetr/models" 2>/dev/null | head -n 40 || true
echo "---- decoder / deformable attention ----"
grep -rni --include='*.py' -e 'deformable' -e 'class LWDETR' -e 'MSDeform' \
     "${TP}/rf-detr/src/rfdetr/models" 2>/dev/null | head -n 30 || true

echo "==> [4/4] DINOv3 weights note"
cat <<'EOF'
   The official facebookresearch/dinov3 source repo is public. Pretrained
   checkpoints still require accepting Meta's access terms. Choose either:

   A) Hugging Face Transformers backend (default config):
       huggingface-cli login
      then accept the model terms at:
       https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m
       https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m

   B) Official Meta repository backend:
      request checkpoint URLs from:
       https://github.com/facebookresearch/dinov3#pretrained-models
      then set source: meta_hub in the YAML config and provide local checkpoint
      paths with weights / teacher_weights, or export:
       DINOV3_BACKBONE_WEIGHTS=/path/or/approved/url
       DINOV3_TEACHER_WEIGHTS=/path/or/approved/url
      Set pretrained: false only if an ungated random initialization is desired.

   Sanity check the install:
       python -m tests.test_forward --config configs/cwdetr_nano_orin.yaml
Done.
EOF
