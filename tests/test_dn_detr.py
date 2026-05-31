from __future__ import annotations

from unittest import mock

import torch
import torch.nn as nn

from cwdetr.config import CWDETRConfig, DecoderCfg
from cwdetr.models.criterion import MultiTaskCriterion
from cwdetr.models.decoder.deformable_decoder import DeformableTransformer
from cwdetr.models.heads import DetectionHead


def _decoder_cfg():
    return DecoderCfg(hidden_dim=32, num_heads=4, num_layers=2,
                      dim_feedforward=64, num_queries=6, num_feature_levels=3,
                      dn_enabled=True, dn_num_groups=2, label_noise_ratio=0.5,
                      box_noise_scale=0.4)


def _srcs():
    return [torch.randn(1, 32, 8, 8), torch.randn(1, 32, 4, 4),
            torch.randn(1, 32, 2, 2)]


def _targets():
    return [{"labels": torch.tensor([0, 2]),
             "boxes": torch.tensor([[0.4, 0.4, 0.2, 0.2],
                                    [0.7, 0.7, 0.1, 0.1]])}]


class _TinyBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.out_channels = list(cfg.out_channels)
        self.out_strides = list(cfg.out_strides)
        self.stems = nn.ModuleList(
            nn.Conv2d(3, channels, stride, stride=stride)
            for channels, stride in zip(self.out_channels, self.out_strides))
        self.teacher = None

    def forward(self, images):
        return [stem(images) for stem in self.stems]

    def teacher_features(self, _):
        return None


def test_dn_query_groups_are_isolated_from_matching_queries():
    transformer = DeformableTransformer(_decoder_cfg(), num_classes=3)
    _, _, _, mask, meta = transformer._build_dn_queries(
        _targets(), dtype=torch.float32, device=torch.device("cpu"))
    assert meta["pad_size"] == 4
    assert meta["valid"].sum() == 4
    # Matching queries cannot attend any denoising slots.
    assert mask[4:, :4].all()
    # Group 0 and group 1 cannot attend each other.
    assert mask[:2, 2:4].all()
    assert mask[2:4, :2].all()
    # Slots inside one denoising group remain mutually visible.
    assert not mask[:2, :2].any()


def test_dn_loss_backpropagates_to_noisy_label_embeddings():
    transformer = DeformableTransformer(_decoder_cfg(), num_classes=3).train()
    head = DetectionHead(32, num_classes=3, num_decoder_layers=2)
    transformer.decoder.bbox_embed = head.bbox_embed
    decoded = transformer(_srcs(), dn_targets=_targets())
    full = head(decoded["hs"], decoded["inter_references"])
    pad_size = decoded["dn_meta"]["pad_size"]
    outputs = {
        "detection": {
            "pred_logits": full["pred_logits"][:, pad_size:],
            "pred_boxes": full["pred_boxes"][:, pad_size:],
            "aux_outputs": [
                {"pred_logits": aux["pred_logits"][:, pad_size:],
                 "pred_boxes": aux["pred_boxes"][:, pad_size:]}
                for aux in full["aux_outputs"]
            ],
        },
        "enc_outputs": decoded["enc_outputs"],
        "dn_outputs": {
            "pred_logits": full["pred_logits"][:, :pad_size],
            "pred_boxes": full["pred_boxes"][:, :pad_size],
            "aux_outputs": [
                {"pred_logits": aux["pred_logits"][:, :pad_size],
                 "pred_boxes": aux["pred_boxes"][:, :pad_size]}
                for aux in full["aux_outputs"]
            ],
        },
        "dn_meta": decoded["dn_meta"],
    }
    loss = MultiTaskCriterion(num_classes=3)(outputs, {"detection": _targets()})
    assert loss["dn/detection"] > 0
    loss["total"].backward()
    assert transformer.dn_label_embed.weight.grad is not None
    assert transformer.dn_label_embed.weight.grad.abs().sum() > 0


def test_model_hides_dn_slots_and_inference_stays_export_neutral():
    cfg = CWDETRConfig()
    cfg.input.height = cfg.input.width = 64
    cfg.model.hidden_dim = cfg.model.projector.hidden_dim = cfg.model.decoder.hidden_dim = 32
    cfg.model.backbone.out_channels = [16, 32, 64]
    cfg.model.decoder.num_heads = 4
    cfg.model.decoder.num_layers = 2
    cfg.model.decoder.dim_feedforward = 64
    cfg.model.decoder.num_queries = 6
    cfg.model.decoder.dn_enabled = True
    cfg.model.decoder.dn_num_groups = 2
    with mock.patch("cwdetr.models.cwdetr.DINOv3Backbone", _TinyBackbone):
        from cwdetr.models.cwdetr import build_cwdetr
        model = build_cwdetr(cfg)
    train_outputs = model.train()(torch.randn(1, 3, 64, 64),
                                  detection_targets=_targets())
    assert train_outputs["detection"]["pred_logits"].shape[1] == 6
    assert train_outputs["dn_outputs"]["pred_logits"].shape[1] == 4
    eval_outputs = model.eval()(torch.randn(1, 3, 64, 64))
    assert "dn_outputs" not in eval_outputs
    assert eval_outputs["detection"]["pred_logits"].shape[1] == 6
