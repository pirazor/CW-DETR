"""Multi-scale deformable attention (Zhu et al., 2021) — pure-PyTorch core.

We deliberately use the ``grid_sample``-based reference implementation rather
than the custom CUDA op. It is mathematically identical and, crucially, it
exports cleanly to ONNX opset >= 16 and is supported by TensorRT, so the same
module trains on a workstation and deploys to Jetson without a custom plugin.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ms_deform_attn_core_pytorch(value, value_spatial_shapes,
                                sampling_locations, attention_weights):
    """
    value:                [B, sum(H*W), n_heads, head_dim]
    value_spatial_shapes: list of (H, W) per level
    sampling_locations:   [B, Lq, n_heads, n_levels, n_points, 2]  in [0, 1]
    attention_weights:    [B, Lq, n_heads, n_levels, n_points]
    returns:              [B, Lq, n_heads*head_dim]
    """
    b, _, n_heads, head_dim = value.shape
    _, lq, _, n_levels, n_points, _ = sampling_locations.shape
    split_sizes = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_sizes, dim=1)
    sampling_grids = 2 * sampling_locations - 1                      # -> [-1, 1]
    sampling_value_list = []
    for lid, (h, w) in enumerate(value_spatial_shapes):
        # [B, H*W, n_heads, head_dim] -> [B*n_heads, head_dim, H, W]
        value_l = (value_list[lid]
                   .flatten(2).transpose(1, 2)
                   .reshape(b * n_heads, head_dim, h, w))
        # [B, Lq, n_heads, n_points, 2] -> [B*n_heads, Lq, n_points, 2]
        grid_l = (sampling_grids[:, :, :, lid]
                  .transpose(1, 2).flatten(0, 1))
        sampled = F.grid_sample(value_l, grid_l, mode="bilinear",
                                padding_mode="zeros", align_corners=False)
        sampling_value_list.append(sampled)                          # [B*nh, hd, Lq, P]
    # weight & sum over levels and points
    attention_weights = (attention_weights
                         .transpose(1, 2)
                         .reshape(b * n_heads, 1, lq, n_levels * n_points))
    out = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
    out = out.sum(-1).view(b, n_heads * head_dim, lq)
    return out.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        # Initialise sampling offsets on a small circle per head (Deformable DETR).
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= (i + 1)
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query, reference_points, value,
                value_spatial_shapes, value_padding_mask=None):
        """
        query:            [B, Lq, C]
        reference_points: [B, Lq, n_levels, 2]  or  [B, Lq, n_levels, 4]
        value:            [B, Lv, C]
        """
        b, lq, _ = query.shape
        b, lv, _ = value.shape

        value = self.value_proj(value)
        if value_padding_mask is not None:
            value = value.masked_fill(value_padding_mask[..., None], 0.0)
        value = value.view(b, lv, self.n_heads, self.head_dim)

        offsets = self.sampling_offsets(query).view(
            b, lq, self.n_heads, self.n_levels, self.n_points, 2)
        attn = self.attention_weights(query).view(
            b, lq, self.n_heads, self.n_levels * self.n_points)
        attn = attn.softmax(-1).view(
            b, lq, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            # normalize x-offset by level width, y-offset by level height -> (W, H)
            offset_norm = torch.stack(
                [torch.tensor([w, h], device=query.device, dtype=query.dtype)
                 for h, w in value_spatial_shapes])              # (n_levels, 2) = (W, H)
            offset_norm = offset_norm.view(1, 1, 1, self.n_levels, 1, 2)
            sampling_locations = (reference_points[:, :, None, :, None, :]
                                  + offsets / offset_norm)
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError("reference_points last dim must be 2 or 4")

        out = ms_deform_attn_core_pytorch(
            value, value_spatial_shapes, sampling_locations, attn)
        return self.output_proj(out)
