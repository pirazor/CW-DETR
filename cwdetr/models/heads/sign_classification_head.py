"""Traffic-sign sub-classification head.

The detection head finds *where* signs are (one coarse 'traffic_sign' class); this
head decides *which* sign it is (e.g. GTSRB's 43 classes or Mapillary's ~400).
Rather than run a second network, it ROI-aligns crops from a shared feature map
using the detection boxes — so fine sign typing is nearly free on top of detection.

Cascade design (decoupled where/what) keeps the detector's class space tiny
(stable training, fewer confusions) while still delivering fine-grained signs.

Training: ROIs come from ground-truth sign boxes (teacher forcing).
Inference: ROIs come from boxes the detector assigned to ``source_det_class``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torchvision.ops import roi_align
except Exception:  # torchvision optional at import time
    roi_align = None


def _group_count(channels: int) -> int:
    """Choose a small GroupNorm divisor for compact classifier widths."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _DepthwiseSeparableResidual(nn.Module):
    """Cheap local ROI refinement without hidden_dim x hidden_dim 3x3 kernels."""

    def __init__(self, channels: int):
        super().__init__()
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class SignClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, num_sign_classes: int, roi_size: int = 7):
        super().__init__()
        self.roi_size = roi_size
        self.num_sign_classes = num_sign_classes
        classifier_dim = max(32, min(96, hidden_dim // 4))
        groups = _group_count(classifier_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(hidden_dim, classifier_dim, 1, bias=False),
            nn.GroupNorm(groups, classifier_dim),
            nn.ReLU(inplace=True),
            _DepthwiseSeparableResidual(classifier_dim),
            _DepthwiseSeparableResidual(classifier_dim),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(classifier_dim * 2 * 2, classifier_dim), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(classifier_dim, num_sign_classes),
        )

    def forward(self, feat: torch.Tensor, rois: torch.Tensor,
                feat_stride: int) -> torch.Tensor:
        """
        feat:       [B, C, H, W] shared feature map (use finest projector level)
        rois:       [K, 5] = (batch_idx, x1, y1, x2, y2) in IMAGE pixel coords
        returns:    [K, num_sign_classes]
        """
        if rois.numel() == 0:
            return feat.new_zeros((0, self.num_sign_classes))
        assert roi_align is not None, "torchvision is required for the sign head"
        pooled = roi_align(feat, rois, output_size=self.roi_size,
                           spatial_scale=1.0 / feat_stride, aligned=True)
        return self.classifier(self.encoder(pooled))

    @staticmethod
    def boxes_to_rois(boxes_xyxy: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """Concat (batch_idx, x1,y1,x2,y2) into the [K,5] roi_align format."""
        if boxes_xyxy.numel() == 0:
            return boxes_xyxy.new_zeros((0, 5))
        return torch.cat([batch_idx[:, None].float(), boxes_xyxy], dim=1)
