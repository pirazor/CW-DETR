"""Detection head (class + box) with iterative box refinement.

Follows the Deformable-DETR / DINO pattern: the per-layer ``bbox_embed`` is
*shared* with the decoder so that the boxes used for refinement are exactly the
boxes the loss supervises. The model wires ``decoder.bbox_embed = head.bbox_embed``
after construction. Boxes therefore come straight out of the decoder's
``inter_references``; this head adds the classification logits.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn

from cwdetr.models.decoder.deformable_decoder import MLP


class DetectionHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, num_decoder_layers: int,
                 prior_prob: float = 0.01):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_decoder_layers

        # Per-layer class + box heads (not weight-shared; modestly more accurate).
        self.class_embed = nn.ModuleList(
            nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers))
        self.bbox_embed = nn.ModuleList(
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_decoder_layers))

        bias = -math.log((1 - prior_prob) / prior_prob)   # focal-loss prior
        for cls in self.class_embed:
            nn.init.constant_(cls.bias, bias)
        for box in self.bbox_embed:
            nn.init.constant_(box.layers[-1].weight, 0.0)
            nn.init.constant_(box.layers[-1].bias, 0.0)

    def forward(self, hs: torch.Tensor, inter_references: torch.Tensor) -> Dict:
        """hs: [L, B, Lq, C]; inter_references: [L, B, Lq, 4] (sigmoid boxes)."""
        logits = torch.stack([self.class_embed[l](hs[l]) for l in range(self.num_layers)])
        boxes = inter_references                       # already refined boxes
        out = {"pred_logits": logits[-1], "pred_boxes": boxes[-1]}
        out["aux_outputs"] = [
            {"pred_logits": logits[l], "pred_boxes": boxes[l]}
            for l in range(self.num_layers - 1)
        ]
        # expose query embeddings for the sign / trajectory heads
        out["query_embed"] = hs[-1]                    # [B, Lq, C]
        return out
