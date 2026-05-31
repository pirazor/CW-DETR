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


class SignClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, num_sign_classes: int, roi_size: int = 7):
        super().__init__()
        self.roi_size = roi_size
        self.num_sign_classes = num_sign_classes
        self.encoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * roi_size * roi_size, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_sign_classes),
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
