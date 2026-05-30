"""Lane-line + drivable-area segmentation head.

A semantic-FPN decoder (Panoptic-FPN style) over the projector's multi-scale
feature maps. Two sibling predictors share the fused features:
  * drivable area  -> {background, direct, alternative}   (BDD100K)
  * lane lines     -> {background, lane}  (binary) or typed lanes (set lane_classes)

This head is the biggest beneficiary of the DINOv3 swap: Gram-anchored dense
features give clean, low-noise patch embeddings, which translate directly into
sharper lane/area masks than DINOv2 produced.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FuseLevel(nn.Sequential):
    def __init__(self, dim):
        super().__init__(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, dim // 2), dim), nn.ReLU(inplace=True),
        )


class SegmentationHead(nn.Module):
    def __init__(self, hidden_dim: int, drivable_classes: int, lane_classes: int,
                 mask_stride: int = 4):
        super().__init__()
        self.mask_stride = mask_stride
        self.lateral = _FuseLevel(hidden_dim)
        self.fuse = _FuseLevel(hidden_dim)
        self.drivable = nn.Conv2d(hidden_dim, drivable_classes, 1)
        self.lane = nn.Conv2d(hidden_dim, lane_classes, 1)

    def forward(self, feats: List[torch.Tensor], image_hw) -> Dict[str, torch.Tensor]:
        """feats: projector maps low-stride(high-res) -> high-stride(low-res)."""
        target_hw = feats[0].shape[-2:]
        fused = self.lateral(feats[0])
        for f in feats[1:]:
            fused = fused + F.interpolate(self.lateral(f), size=target_hw,
                                          mode="bilinear", align_corners=False)
        fused = self.fuse(fused)

        out_h = image_hw[0] // self.mask_stride
        out_w = image_hw[1] // self.mask_stride
        fused = F.interpolate(fused, size=(out_h, out_w), mode="bilinear",
                              align_corners=False)
        return {
            "drivable_logits": self.drivable(fused),   # [B, Cd, H/s, W/s]
            "lane_logits": self.lane(fused),           # [B, Cl, H/s, W/s]
        }
