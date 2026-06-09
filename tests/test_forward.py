"""Offline shape/forward sanity tests for CW-DETR.

DINOv3 pretrained weights are access-controlled and heavy, so the full-model test swaps in a DummyBackbone
(via mock) producing correctly-shaped random feature maps. Every other module is
exercised with synthetic tensors. Run either as pytest or directly:

    python -m tests.test_forward --config configs/cwdetr_nano_orin.yaml
    pytest tests/test_forward.py -q
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from cwdetr.config import BackboneCfg, load_config, CWDETRConfig, ModelCfg
from cwdetr.data.multitask_dataset import ConcatMultiTaskDataset
from cwdetr.data.transforms import RandomHFlip
from cwdetr.engine.train import build_param_groups
from cwdetr.models.decoder.deformable_decoder import (DeformableTransformer,
                                                       gen_encoder_output_proposals)
from cwdetr.models.decoder.deformable_attention import MSDeformAttn
from cwdetr.models.backbone.dinov3_backbone import DINOv3Backbone
from cwdetr.models.heads import (DetectionHead, SignClassificationHead,
                                 TrackInstances, TrackQueryHead, TrajectoryHead)
from cwdetr.models.criterion import MultiTaskCriterion
from cwdetr.tracking.bytetrack import BYTETracker


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


class DummyMetaHubEncoder(nn.Module):
    """Official Meta-repo stand-in exposing the intermediate-layer API."""

    channels = [96, 192, 384, 768]

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def get_intermediate_layers(self, x, n=1, reshape=False):
        stages = list(range(4 - n, 4)) if isinstance(n, int) else list(n)
        b, _, h, w = x.shape
        return tuple(self.scale * torch.randn(b, self.channels[stage],
                                              h // (2 ** (stage + 2)),
                                              w // (2 ** (stage + 2)))
                     for stage in stages)


class DummyMetaHubViTEncoder(nn.Module):
    embed_dim = 768

    def get_intermediate_layers(self, x, n=1, reshape=False):
        assert n == 1
        b, _, h, w = x.shape
        return (torch.randn(b, self.embed_dim, h // 16, w // 16),)


class DummySignDataset(Dataset):
    def __init__(self, taxonomy):
        self.sign_taxonomy = taxonomy

    def __len__(self):
        return 1

    def __getitem__(self, _):
        return {"sign_labels": torch.tensor([0])}


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


def test_config_load_nested_dataclasses():
    cfg = load_config("configs/cwdetr_nano_orin.yaml")
    assert isinstance(cfg.model, ModelCfg)
    assert cfg.model.decoder.hidden_dim == 256
    assert cfg.model.heads.sign_classification.source_det_class == 11
    print("  [ok] typed config load")


def test_meta_hub_backbone_loader():
    cfg = BackboneCfg(
        source="meta_hub", meta_model="dinov3_convnext_tiny",
        weights="checkpoint.pth", out_indices=[1, 2, 3],
        out_channels=[192, 384, 768], out_strides=[8, 16, 32])
    factory = mock.Mock(return_value=DummyMetaHubEncoder())
    module = SimpleNamespace(dinov3_convnext_tiny=factory)
    with mock.patch.object(DINOv3Backbone, "_resolve_meta_repo", return_value="repo"), \
            mock.patch("importlib.import_module", return_value=module):
        backbone = DINOv3Backbone(cfg)
        feats = backbone(torch.randn(1, 3, 64, 96))
    factory.assert_called_once_with(pretrained=True, weights="checkpoint.pth")
    assert [tuple(f.shape) for f in feats] == [
        (1, 192, 8, 12), (1, 384, 4, 6), (1, 768, 2, 3)]
    backbone.teacher = DummyMetaHubEncoder()
    backbone._teacher_mode = "meta_hub"
    assert backbone.teacher_features(torch.randn(1, 3, 64, 96)).shape == (1, 768, 2, 3)
    print("  [ok] official Meta repository backbone")


def test_meta_hub_random_init_without_weights():
    factory = mock.Mock(return_value=DummyMetaHubEncoder())
    module = SimpleNamespace(dinov3_convnext_tiny=factory)
    with mock.patch.object(DINOv3Backbone, "_resolve_meta_repo", return_value="repo"), \
            mock.patch("importlib.import_module", return_value=module):
        DINOv3Backbone(BackboneCfg(source="meta_hub", pretrained=False))
    factory.assert_called_once_with(pretrained=False)
    print("  [ok] official Meta repository random init")


def test_backbone_train_backbone_false_freezes_encoder():
    cfg = BackboneCfg(
        source="meta_hub", meta_model="dinov3_convnext_tiny",
        weights="checkpoint.pth", out_indices=[1, 2, 3],
        out_channels=[192, 384, 768], out_strides=[8, 16, 32],
        train_backbone=False)
    module = SimpleNamespace(dinov3_convnext_tiny=lambda **_: DummyMetaHubEncoder())
    with mock.patch.object(DINOv3Backbone, "_resolve_meta_repo", return_value="repo"), \
            mock.patch("importlib.import_module", return_value=module):
        backbone = DINOv3Backbone(cfg)
    assert not any(parameter.requires_grad for parameter in backbone.encoder.parameters())
    backbone.encoder.train()
    backbone.train()
    assert not backbone.encoder.training
    print("  [ok] frozen DINOv3 encoder stays eval-only")


def test_meta_hub_vit_uses_final_map_and_validates_fpn_contract():
    cfg = BackboneCfg(
        type="dinov3_vit", source="meta_hub", meta_model="dinov3_vitb16",
        weights="checkpoint.pth", out_channels=[192, 384, 768],
        out_strides=[8, 16, 32], embed_dim=768, windowed_attention=False)
    factory = mock.Mock(return_value=DummyMetaHubViTEncoder())
    module = SimpleNamespace(dinov3_vitb16=factory)
    with mock.patch.object(DINOv3Backbone, "_resolve_meta_repo", return_value="repo"), \
            mock.patch("importlib.import_module", return_value=module):
        backbone = DINOv3Backbone(cfg)
        feats = backbone(torch.randn(1, 3, 64, 96))
    assert [tuple(f.shape) for f in feats] == [
        (1, 192, 8, 12), (1, 384, 4, 6), (1, 768, 2, 3)]
    print("  [ok] official Meta ViT final map + FPN contract")


def test_backbone_stride_contract_rejects_wrong_config():
    cfg = BackboneCfg(
        source="meta_hub", meta_model="dinov3_convnext_tiny",
        weights="checkpoint.pth", out_indices=[1, 2, 3],
        out_channels=[192, 384, 768], out_strides=[4, 8, 16])
    module = SimpleNamespace(dinov3_convnext_tiny=lambda **_: DummyMetaHubEncoder())
    with mock.patch.object(DINOv3Backbone, "_resolve_meta_repo", return_value="repo"), \
            mock.patch("importlib.import_module", return_value=module):
        backbone = DINOv3Backbone(cfg)
        try:
            backbone(torch.randn(1, 3, 64, 96))
        except ValueError as exc:
            assert "violates contract" in str(exc)
        else:
            raise AssertionError("wrong backbone stride config must fail")
    print("  [ok] backbone stride contract")


def test_hf_convnext_automodel_fallback_selects_stride_contract():
    image = torch.randn(2, 3, 384, 640)
    hidden_states = (
        torch.randn(2, 96, 96, 160),
        torch.randn(2, 96, 96, 160),
        torch.randn(2, 192, 48, 80),
        torch.randn(2, 384, 24, 40),
        torch.randn(2, 768, 12, 20),
    )
    feats = DINOv3Backbone._select_convnext_features(
        image, hidden_states, [192, 384, 768], [8, 16, 32])
    assert [tuple(feat.shape) for feat in feats] == [
        (2, 192, 48, 80), (2, 384, 24, 40), (2, 768, 12, 20)]
    print("  [ok] HF ConvNeXt AutoModel fallback stage selection")


def test_vit_windowing_fails_closed():
    try:
        DINOv3Backbone(BackboneCfg(type="dinov3_vit", windowed_attention=True))
    except ValueError as exc:
        assert "RoPE" in str(exc)
    else:
        raise AssertionError("unsafe ViT windowing must fail closed")
    print("  [ok] unsafe ViT windowing disabled")


def test_frozen_teacher_stays_in_eval_mode():
    cfg = BackboneCfg(
        source="meta_hub", meta_model="dinov3_convnext_tiny",
        weights="checkpoint.pth", out_indices=[1, 2, 3],
        out_channels=[192, 384, 768], out_strides=[8, 16, 32])
    module = SimpleNamespace(dinov3_convnext_tiny=lambda **_: DummyMetaHubEncoder())
    with mock.patch.object(DINOv3Backbone, "_resolve_meta_repo", return_value="repo"), \
            mock.patch("importlib.import_module", return_value=module):
        backbone = DINOv3Backbone(cfg)
    backbone.teacher = nn.Dropout()
    backbone.train()
    assert not backbone.teacher.training
    print("  [ok] frozen teacher remains eval-only")


def test_encoder_proposal_centers():
    _, proposals = gen_encoder_output_proposals(torch.zeros(1, 4, 8), [(2, 2)])
    centers = proposals.sigmoid()[0, :, :2]
    expected = torch.tensor([[0.25, 0.25], [0.75, 0.25],
                             [0.25, 0.75], [0.75, 0.75]])
    assert torch.allclose(centers, expected), centers
    print("  [ok] encoder proposal centers")


def test_transformer_and_detection_head():
    cfg = CWDETRConfig().model.decoder
    cfg.hidden_dim = 256
    tr = DeformableTransformer(cfg, num_classes=13)
    head = DetectionHead(256, num_classes=13, num_decoder_layers=cfg.num_layers)
    tr.decoder.bbox_embed = head.bbox_embed
    out = tr(_synthetic_srcs())
    det = head(out["hs"], out["inter_references"])
    assert det["pred_logits"].shape == (2, cfg.num_queries, 13)
    assert det["pred_boxes"].shape == (2, cfg.num_queries, 4)
    assert len(det["aux_outputs"]) == cfg.num_layers - 1
    print("  [ok] transformer+det head", tuple(det["pred_logits"].shape))


def test_encoder_proposal_head_receives_gradients():
    cfg = CWDETRConfig().model.decoder
    tr = DeformableTransformer(cfg, num_classes=13)
    head = DetectionHead(256, num_classes=13, num_decoder_layers=cfg.num_layers)
    tr.decoder.bbox_embed = head.bbox_embed
    dec = tr(_synthetic_srcs(b=1))
    det = head(dec["hs"], dec["inter_references"])
    crit = MultiTaskCriterion(num_classes=13)
    outputs = {"detection": det, "enc_outputs": dec["enc_outputs"]}
    targets = {"detection": [
        {"labels": torch.tensor([0, 5]), "boxes": torch.rand(2, 4)}
    ]}
    crit(outputs, targets)["total"].backward()
    for parameter in (tr.enc_class_embed.weight, tr.enc_bbox.layers[-1].weight,
                      tr.enc_output.weight):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0
    print("  [ok] supervised encoder proposal gradients")


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
    print("  [ok] matcher+criterion total=%.3f" % float(losses["total"].detach()))


def test_criterion_skips_absent_segmentation_targets():
    crit = MultiTaskCriterion(num_classes=13)
    outputs = {
        "detection": {
            "pred_logits": torch.randn(1, 8, 13),
            "pred_boxes": torch.rand(1, 8, 4),
            "aux_outputs": [],
        },
        "segmentation": {
            "drivable_logits": torch.randn(1, 3, 8, 8),
            "lane_logits": torch.randn(1, 2, 8, 8),
        },
    }
    targets = {
        "detection": [{"labels": torch.tensor([0]), "boxes": torch.rand(1, 4)}],
        "drivable": None,
        "lane": None,
    }
    losses = crit(outputs, targets)
    assert losses["segmentation"] is None
    assert torch.isfinite(losses["total"])
    print("  [ok] criterion skips absent segmentation targets")


def test_criterion_skips_crop_only_detection_but_trains_signs():
    crit = MultiTaskCriterion(num_classes=13)
    logits = torch.randn(1, 8, 13, requires_grad=True)
    boxes = torch.rand(1, 8, 4, requires_grad=True)
    sign_logits = torch.randn(1, 43, requires_grad=True)
    outputs = {
        "detection": {"pred_logits": logits, "pred_boxes": boxes, "aux_outputs": []},
        "sign_logits": sign_logits,
    }
    targets = {
        "detection": [{
            "labels": torch.zeros(0, dtype=torch.long),
            "boxes": torch.zeros(0, 4),
            "train_detection": False,
        }],
        "sign_labels": torch.tensor([3]),
    }
    losses = crit(outputs, targets)
    losses["total"].backward()
    assert losses["detection"] == 0
    assert losses["sign"] > 0
    assert logits.grad is not None and logits.grad.abs().sum() == 0
    assert boxes.grad is not None and boxes.grad.abs().sum() == 0
    assert sign_logits.grad is not None and sign_logits.grad.abs().sum() > 0
    print("  [ok] crop-only sign data bypasses detector loss")


def test_mixed_sign_taxonomies_require_explicit_mapping():
    try:
        ConcatMultiTaskDataset([DummySignDataset("gtsrb43"),
                                DummySignDataset("mapillary")])
    except ValueError as exc:
        assert "sign_label_maps" in str(exc)
    else:
        raise AssertionError("mixed fine-sign taxonomies must require explicit maps")
    dataset = ConcatMultiTaskDataset(
        [DummySignDataset("gtsrb43"), DummySignDataset("mapillary")],
        sign_label_maps={"gtsrb43": {0: 4}, "mapillary": {0: 8}})
    assert dataset[0]["sign_labels"].tolist() == [4]
    assert dataset[1]["sign_labels"].tolist() == [8]
    print("  [ok] mixed sign taxonomy validation")


def test_optimizer_groups_include_criterion_parameters():
    model = nn.Linear(4, 4)
    crit = MultiTaskCriterion(num_classes=13, teacher_dim=8, student_dim=4)
    groups = build_param_groups(model, 1e-3, crit)
    grouped = {id(p) for group in groups for p in group["params"]}
    assert all(id(p) in grouped for p in crit.parameters())
    print("  [ok] optimizer includes criterion parameters")


def test_track_query_miss_tolerance():
    head = TrackQueryHead(8, score_thresh=0.5, miss_tolerance=1)
    prev = TrackInstances(
        query_embed=torch.zeros(1, 8),
        query_pos=torch.zeros(1, 8),
        ref_points=torch.rand(1, 4),
        obj_ids=torch.tensor([7]),
        scores=torch.tensor([0.9]),
        disappear=torch.zeros(1),
    )
    missed_once = head.update(prev, torch.zeros(1, 8), torch.rand(1, 4),
                              torch.tensor([0.0]), num_track=1)
    assert len(missed_once) == 1
    assert missed_once.obj_ids.tolist() == [7]
    assert missed_once.disappear.tolist() == [1.0]
    missed_twice = head.update(missed_once, torch.zeros(1, 8), torch.rand(1, 4),
                               torch.tensor([0.0]), num_track=1)
    assert len(missed_twice) == 0
    print("  [ok] track query miss tolerance")


def test_bytetrack_class_gating():
    tracker = BYTETracker(high_thresh=0.5, match_thresh=0.5)
    boxes = np.asarray([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    scores = np.asarray([0.9], dtype=np.float32)
    tracker.update(boxes, scores, np.asarray([0]))
    tracker.update(boxes, scores, np.asarray([1]))
    assert len(tracker.tracks) == 2
    assert sorted(t.cls for t in tracker.tracks) == [0, 1]
    print("  [ok] ByteTrack class gating")


def test_image_trajectory_flip():
    sample = {
        "image": Image.new("RGB", (8, 4)),
        "boxes": torch.tensor([[0.25, 0.5, 0.2, 0.2]]),
        "sign_boxes": torch.tensor([[1.0, 0.0, 3.0, 2.0]]),
        "future": torch.tensor([[[0.25, 0.1]]]),
        "trajectory_space": "image",
    }
    flipped = RandomHFlip(1.0)(sample)
    assert flipped["boxes"][0, 0] == 0.75
    assert flipped["future"][0, 0, 0] == -0.25
    print("  [ok] image trajectory flip")


def test_trajectory_and_sign_heads():
    traj = TrajectoryHead(256, history_len=10, future_len=12, num_modes=6)
    t, logits = traj(torch.randn(5, 256), torch.randn(5, 10, 2))
    assert t.shape == (5, 6, 12, 2) and logits.shape == (5, 6)
    sign = SignClassificationHead(256, num_sign_classes=43, roi_size=7)
    feat = torch.randn(1, 256, 48, 80)
    rois = torch.tensor([[0, 10, 10, 60, 60], [0, 100, 80, 160, 140]], dtype=torch.float32)
    out = sign(feat, rois, feat_stride=8)
    assert out.shape == (2, 43), out.shape
    assert sum(p.numel() for p in sign.parameters()) < 100_000
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
TESTS = [test_deform_attn, test_config_load_nested_dataclasses, test_meta_hub_backbone_loader,
         test_meta_hub_random_init_without_weights, test_backbone_train_backbone_false_freezes_encoder,
         test_meta_hub_vit_uses_final_map_and_validates_fpn_contract,
         test_backbone_stride_contract_rejects_wrong_config,
         test_hf_convnext_automodel_fallback_selects_stride_contract, test_vit_windowing_fails_closed,
         test_frozen_teacher_stays_in_eval_mode, test_encoder_proposal_centers,
         test_transformer_and_detection_head, test_encoder_proposal_head_receives_gradients,
         test_matcher_and_criterion, test_criterion_skips_absent_segmentation_targets,
         test_criterion_skips_crop_only_detection_but_trains_signs,
         test_mixed_sign_taxonomies_require_explicit_mapping,
         test_optimizer_groups_include_criterion_parameters, test_track_query_miss_tolerance,
         test_bytetrack_class_gating, test_image_trajectory_flip, test_trajectory_and_sign_heads,
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
