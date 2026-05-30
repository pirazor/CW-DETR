"""Shared deformable-DETR decoder for CW-DETR.

Design choice: like RF-DETR, the (single-scale) backbone + projector play the
role of the encoder, so there is *no* separate deformable encoder. The decoder
cross-attends straight into the projector's multi-scale memory. Every task head
consumes the SAME decoder output, which is what makes the multi-task model cheap
— the expensive backbone+decoder run once and five heads share it.

Includes: sine positional encoding, two-stage proposal generation, iterative
box refinement (DINO ``look_forward_twice``), and a ``self_attn_mask`` hook used
later by track queries (MOTR-style) and DN-DETR denoising groups.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cwdetr.models.decoder.deformable_attention import MSDeformAttn


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """Standard DETR sine positional embedding -> [B, C, H, W]."""

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        b, _, h, w = x.shape
        if mask is None:
            mask = torch.zeros((b, h, w), dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), 4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), 4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


def gen_encoder_output_proposals(memory, spatial_shapes):
    """Deformable-DETR two-stage proposal grid from flattened memory."""
    b = memory.shape[0]
    proposals, cur = [], 0
    for h, w in spatial_shapes:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.5, h - 0.5, h, device=memory.device),
            torch.linspace(0.5, w - 0.5, w, device=memory.device), indexing="ij")
        grid = torch.stack([grid_x, grid_y], -1)               # (h, w, 2)
        scale = torch.tensor([w, h], device=memory.device).view(1, 1, 2)
        grid = (grid.unsqueeze(0).expand(b, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05 * (2.0 ** len(proposals))
        proposals.append(torch.cat([grid, wh], -1).view(b, -1, 4))
        cur += h * w
    output_proposals = torch.cat(proposals, 1)
    valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(~valid, float("inf"))
    output_memory = memory.masked_fill(~valid, 0.0)
    return output_memory, output_proposals


def get_proposal_pos_embed(proposals, num_pos_feats=128, temperature=10000):
    """Sinusoidal embedding of (cx, cy, w, h) proposals -> content query init."""
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    proposals = proposals.sigmoid() * scale
    pos = proposals[:, :, :, None] / dim_t
    pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), 4).flatten(2)
    return pos


# --------------------------------------------------------------------------- #
# decoder
# --------------------------------------------------------------------------- #
class DeformableDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, n_levels, n_points, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes,
                src_padding_mask=None, self_attn_mask=None):
        q = k = tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))
        tgt2 = self.cross_attn(tgt + query_pos, reference_points, src,
                               src_spatial_shapes, src_padding_mask)
        tgt = self.norm2(tgt + self.dropout(tgt2))
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout(tgt2))
        return tgt


class DeformableDecoder(nn.Module):
    def __init__(self, layer_kwargs, num_layers, look_forward_twice=True):
        super().__init__()
        self.layers = nn.ModuleList(
            DeformableDecoderLayer(**layer_kwargs) for _ in range(num_layers))
        self.num_layers = num_layers
        self.look_forward_twice = look_forward_twice
        # Set externally by the detection head for iterative box refinement.
        self.bbox_embed: Optional[nn.ModuleList] = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, valid_ratios,
                query_pos=None, src_padding_mask=None, self_attn_mask=None):
        output = tgt
        intermediate, intermediate_ref = [], []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                ref_in = (reference_points[:, :, None]
                          * torch.cat([valid_ratios, valid_ratios], -1)[:, None])
            else:
                ref_in = reference_points[:, :, None] * valid_ratios[:, None]
            output = layer(output, query_pos, ref_in, src, src_spatial_shapes,
                           src_padding_mask, self_attn_mask)

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)                 # [B, Lq, 4] box deltas
                if reference_points.shape[-1] == 4:
                    new_ref = (tmp + inverse_sigmoid(reference_points)).sigmoid()
                else:                                              # 2-d point -> promote to box
                    xy = (tmp[..., :2] + inverse_sigmoid(reference_points)).sigmoid()
                    wh = tmp[..., 2:].sigmoid()
                    new_ref = torch.cat([xy, wh], dim=-1)
            else:
                new_ref = reference_points

            intermediate.append(output)
            intermediate_ref.append(new_ref)                       # grad-carrying, for aux loss
            # look_forward_twice (DINO): let gradients flow into the next layer too.
            reference_points = new_ref if self.look_forward_twice else new_ref.detach()

        return torch.stack(intermediate), torch.stack(intermediate_ref)


class DeformableTransformer(nn.Module):
    """Glue: flatten projector memory, build queries (two-stage), run decoder."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden_dim
        self.d_model = d
        self.num_queries = cfg.num_queries
        self.num_levels = cfg.num_feature_levels
        self.two_stage = cfg.two_stage

        self.pos_embed = PositionEmbeddingSine(d // 2)
        self.level_embed = nn.Parameter(torch.zeros(cfg.num_feature_levels, d))
        nn.init.normal_(self.level_embed)

        self.decoder = DeformableDecoder(
            layer_kwargs=dict(d_model=d, n_heads=cfg.num_heads, dim_ff=cfg.dim_feedforward,
                              n_levels=cfg.num_feature_levels, n_points=cfg.dec_n_points),
            num_layers=cfg.num_layers, look_forward_twice=cfg.look_forward_twice)

        if self.two_stage:
            self.enc_output = nn.Linear(d, d)
            self.enc_output_norm = nn.LayerNorm(d)
            self.enc_objectness = nn.Linear(d, 1)
            self.enc_bbox = MLP(d, d, 4, 3)
            self.pos_trans = nn.Linear(d, d)
            self.pos_trans_norm = nn.LayerNorm(d)
        else:
            self.query_embed = nn.Embedding(cfg.num_queries, d * 2)
            self.reference_points = nn.Linear(d, 2)

    @staticmethod
    def _flatten(srcs, pos_embeds, level_embed):
        src_flatten, pos_flatten, shapes = [], [], []
        for lvl, (src, pos) in enumerate(zip(srcs, pos_embeds)):
            b, c, h, w = src.shape
            shapes.append((h, w))
            src_flatten.append(src.flatten(2).transpose(1, 2))            # [B, HW, C]
            pos_flatten.append(pos.flatten(2).transpose(1, 2) + level_embed[lvl].view(1, 1, -1))
        return (torch.cat(src_flatten, 1), torch.cat(pos_flatten, 1), shapes)

    def forward(self, srcs: List[torch.Tensor],
                extra_queries: Optional[torch.Tensor] = None,
                extra_query_pos: Optional[torch.Tensor] = None,
                extra_reference_points: Optional[torch.Tensor] = None,
                self_attn_mask: Optional[torch.Tensor] = None):
        """``extra_*`` carry track queries (prepended to the object queries)."""
        pos_embeds = [self.pos_embed(s) for s in srcs]
        src_flatten, lvl_pos_flatten, spatial_shapes = self._flatten(
            srcs, pos_embeds, self.level_embed)
        b = src_flatten.shape[0]
        valid_ratios = torch.ones(b, self.num_levels, 2, device=src_flatten.device)

        enc_out = {}
        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(
                src_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            obj = self.enc_objectness(output_memory)                       # [B, L, 1]
            coord = self.enc_bbox(output_memory) + output_proposals        # [B, L, 4]
            topk = min(self.num_queries, obj.shape[1])
            topk_idx = torch.topk(obj[..., 0], topk, dim=1)[1]
            ref = torch.gather(coord, 1, topk_idx[..., None].expand(-1, -1, 4)).sigmoid()
            ref = ref.detach()
            query_pos = self.pos_trans_norm(self.pos_trans(
                get_proposal_pos_embed(ref, self.d_model // 2)))
            tgt = torch.gather(output_memory, 1,
                               topk_idx[..., None].expand(-1, -1, self.d_model)).detach()
            enc_out = {"objectness": obj, "coords": coord.sigmoid()}
        else:
            qe = self.query_embed.weight                                   # [Q, 2d]
            query_pos, tgt = torch.split(qe, self.d_model, dim=1)
            query_pos = query_pos[None].expand(b, -1, -1)
            tgt = tgt[None].expand(b, -1, -1)
            ref = self.reference_points(query_pos).sigmoid()               # [B, Q, 2]

        # Prepend track queries (MOTR-style) if provided.
        if extra_queries is not None:
            tgt = torch.cat([extra_queries, tgt], dim=1)
            query_pos = torch.cat([extra_query_pos, query_pos], dim=1)
            ref = torch.cat([extra_reference_points, ref], dim=1)

        init_reference = ref
        hs, inter_ref = self.decoder(
            tgt, ref, src_flatten, spatial_shapes, valid_ratios,
            query_pos=query_pos, self_attn_mask=self_attn_mask)
        return {
            "hs": hs,                          # [n_layers, B, Lq, C]
            "init_reference": init_reference,  # [B, Lq, 2 or 4]
            "inter_references": inter_ref,     # [n_layers, B, Lq, 2 or 4]
            "memory": src_flatten,             # [B, sum(HW), C]  (for seg head)
            "spatial_shapes": spatial_shapes,
            "enc_outputs": enc_out,
        }
