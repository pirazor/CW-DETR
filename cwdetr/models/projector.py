"""Multi-scale projector / neck for CW-DETR.

RF-DETR feeds its single-scale backbone into an LW-DETR-style projector that
produces the multi-scale feature maps the deformable decoder samples from. We
keep that role but make the projector backbone-agnostic: it accepts either the
ConvNeXt hierarchical pyramid (3 maps, different channels) or the ViT simple
feature pyramid (3 maps) and emits exactly ``num_levels`` maps, all unified to
``hidden_dim`` channels, fused with a light top-down + bottom-up (PAN/C2f) path.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Sequential):
    def __init__(self, c_in, c_out, k=3, s=1, p=None, act=True):
        p = (k // 2) if p is None else p
        layers = [nn.Conv2d(c_in, c_out, k, s, p, bias=False),
                  nn.GroupNorm(min(32, c_out // 2 if c_out >= 2 else 1), c_out)]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class C2fBlock(nn.Module):
    """CSP-style fusion block (the 'C2f' used in the LW-DETR/YOLO lineage)."""

    def __init__(self, dim: int, n: int = 2):
        super().__init__()
        self.cv1 = ConvGNAct(dim, dim, k=1)
        self.m = nn.ModuleList(ConvGNAct(dim // 2, dim // 2) for _ in range(n))
        self.cv2 = ConvGNAct(dim // 2 * (n + 2), dim, k=1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        for blk in self.m:
            y.append(blk(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class MultiScaleProjector(nn.Module):
    def __init__(self, in_channels: List[int], in_strides: List[int],
                 hidden_dim: int, num_levels: int):
        super().__init__()
        self.in_strides = list(in_strides)
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim

        # 1x1 lateral projections to a common width.
        self.input_proj = nn.ModuleList(
            ConvGNAct(c, hidden_dim, k=1, act=False) for c in in_channels
        )
        n_in = len(in_channels)

        # Top-down then bottom-up fusion (PAN), with C2f blocks at each merge.
        self.td_blocks = nn.ModuleList(C2fBlock(hidden_dim) for _ in range(n_in - 1))
        self.bu_downsamples = nn.ModuleList(
            ConvGNAct(hidden_dim, hidden_dim, k=3, s=2) for _ in range(n_in - 1)
        )
        self.bu_blocks = nn.ModuleList(C2fBlock(hidden_dim) for _ in range(n_in - 1))

        # Extra coarser levels (stride 64...) if the decoder wants more than we have.
        self.extra = nn.ModuleList(
            ConvGNAct(hidden_dim, hidden_dim, k=3, s=2)
            for _ in range(max(0, num_levels - n_in))
        )

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        # feats: low-stride (high-res) -> high-stride (low-res)
        laterals = [proj(f) for proj, f in zip(self.input_proj, feats)]

        # top-down
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:],
                               mode="nearest")
            laterals[i - 1] = self.td_blocks[i - 1](laterals[i - 1] + up)

        # bottom-up
        outs = [laterals[0]]
        for i in range(len(laterals) - 1):
            down = self.bu_downsamples[i](outs[-1])
            outs.append(self.bu_blocks[i](laterals[i + 1] + down))

        # extra coarse levels
        for layer in self.extra:
            outs.append(layer(outs[-1]))

        return outs[: self.num_levels]
