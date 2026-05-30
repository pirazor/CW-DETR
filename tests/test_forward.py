"""Offline shape/forward sanity tests for CW-DETR.

DINOv3 weights are gated and heavy, so the full-model test swaps in a DummyBackbone
(via mock) producing correctly-shaped random feature maps. Every other module is
exercised with synthetic tensors. Run either as pytest or directly:

    python -m tests.test_forward --config configs/cwdetr_nano_orin.yaml
    pytest tests/test_forward.py -q
"""
from __future__ import annotations

import argparse
from unittest import mock

import torch
import torch.nn as nn

from cwdetr.config import load_config, CWDETRConfig
from cwdetr.models.decoder.deformable_decoder import DeformableTransformer
from cwdetr.models.decoder.deformable_attention import MSDeformAttn
from cwdetr.models.heads import DetectionHead, TrajectoryHead, SignClassificationHead
from cwdetr.models.matcher import HungarianMatcher
from cwdetr.models.criterion import MultiTaskCriterion


# --------------------------------------------------------------------------- #
class DummyBackbone(nn.Module):
    """Stand-in for DINOv3: emits maps at the configured channels/strides."""

    def __init__(self, cfg):
        super().__init__()
        self.out_channels = list(cfg.out_channels) or [192, 384, 768]
        self.out_strides = list(cfg.out_strides) or [8, 16, 32]
        self.stems = nn.ModuleList(
            nn.Conv2d(3, c, k, stride=k) for c, k in zip(self.out_channels, self.out_strides))
        self.teacher = None

    def forward(self, x):
        return [stem(x) for stem in self.stems]

    def teacher_features(self, x):
        return None


def _synthetic_srcs(b=2, c=256, base_hw=(48, 80)):
    h, w = base_hw
    return [torch.randn(b, c, h, w), torch.randn(b, c, h // 2, w // 2),
            torch.randn(b, c, h // 4, w // 4)]


# --------------------------------------------------------------------------- #
def test_deform_attn():
    b, q, c = 2, 100, 256
    attn = MSDeformAttn(c, n_levels=3, n_heads=8, n_points=4)
    shapes = [(48, 80), (24, 40), (12, 20)]
    lv = sum(h * w for h, w in shapes)
    value = torch.randn(b, lv, c)
    ref = torch.rand(b, q, 3, 2)
    out = attn(torch.randn(b, q, c), ref, value, shapes)
    assert out.shape == (b, q, c), out.shape
    print("  [ok] MSDeformAttn", tuple(out.shape))


def test_transformer_and_detection_head():
    cfg = CWDETRConfig().model.decoder
    cfg.hidden_dim = 256
    tr = DeformableTransformer(cfg)
    head = DetectionHead(256, num_classes=13, num_decoder_layers=cfg.num_layers)
    tr.decoder.bbox_embed = head.bbox_embed
    out = tr(_synthetic_srcs())
    det = head(out["hs"], out["inter_references"])
    assert det["pred_logits"].shape == (2, cfg.num_queries, 13)
    assert det["pred_boxes"].shape == (2, cfg.num_queries, 4)
    assert len(det["aux_outputs"]) == cfg.num_layers - 1
    print("  [ok] transformer+det head", tuple(det["pred_logits"].shape))


def test_matcher_and_criterion():
    crit = MultiTaskCriterion(num_classes=13)
    outputs = {"detection": {
        "pred_logits": torch.randn(2, 300, 13),
        "pred_boxes": torch.rand(2, 300, 4),
        "aux_outputs": [],
    }}
    targets = {"detection": [
        {"labels": torch.tensor([0, 5]), "boxes": torch.rand(2, 4)},
        {"labels": torch.tensor([11]), "boxes": torch.rand(1, 4)},
    ]}
    losses = crit(outputs, targets)
    assert torch.isfinite(losses["total"]), losses
    print("  [ok] matcher+criterion total=%.3f" % float(losses["total"]))


def test_trajectory_and_sign_heads():
    traj = TrajectoryHead(256, history_len=10, future_len=12, num_modes=6)
    t, logits = traj(torch.randn(5, 256), torch.randn(5, 10, 2))
    assert t.shape == (5, 6, 12, 2) and logits.shape == (5, 6)
    sign = SignClassificationHead(256, num_sign_classes=43, roi_size=7)
    feat = torch.randn(1, 256, 48, 80)
    rois = torch.tensor([[0, 10, 10, 60, 60], [0, 100, 80, 160, 140]], dtype=torch.float32)
    out = sign(feat, rois, feat_stride=8)
    assert out.shape == (2, 43), out.shape
    print("  [ok] trajectory + sign heads")


def test_full_model_with_dummy_backbone():
    cfg = load_config(ARGS.config) if ARGS and ARGS.config else CWDETRConfig()
    with mock.patch("cwdetr.models.cwdetr.DINOv3Backbone", DummyBackbone):
        from cwdetr.models.cwdetr import build_cwdetr
        model = build_cwdetr(cfg).eval()
        img = torch.randn(2, 3, cfg.input.height, cfg.input.width)
        with torch.no_grad():
            out = model(img)
        assert out["detection"]["pred_logits"].shape[0] == 2
        if "segmentation" in out:
            ds = out["segmentation"]["drivable_logits"]
            assert ds.shape[-2] == cfg.input.height // cfg.model.heads.segmentation.mask_stride
        # sign ROIs path
        rois = model.detected_sign_rois(out, score_thresh=-1.0)  # force some rois
        print("  [ok] full model fwd:",
              tuple(out["detection"]["pred_logits"].shape),
              "seg" if "segmentation" in out else "no-seg",
              f"rois={tuple(rois.shape)}")


ARGS = None
TESTS = [test_deform_attn, test_transformer_and_detection_head,
         test_matcher_and_criterion, test_trajectory_and_sign_heads,
         test_full_model_with_dummy_backbone]


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cwdetr_nano_orin.yaml")
    ARGS = ap.parse_args()
    print("Running CW-DETR sanity tests...")
    for t in TESTS:
        t()
    print("All sanity tests passed.")


if __name__ == "__main__":
    main()
