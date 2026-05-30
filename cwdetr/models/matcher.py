"""Hungarian matcher for the detection set loss (DETR-style, focal cost)."""
from __future__ import annotations

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from cwdetr.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = focal_alpha
        self.gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        """outputs: pred_logits [B,Q,C], pred_boxes [B,Q,4] (cxcywh, normalized).
        targets: list of dicts with 'labels' [n] and 'boxes' [n,4] (cxcywh)."""
        bs, q = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()    # [B*Q, C]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)               # [B*Q, 4]

        tgt_ids = torch.cat([t["labels"] for t in targets])
        tgt_bbox = torch.cat([t["boxes"] for t in targets])
        if tgt_ids.numel() == 0:
            return [(torch.as_tensor([], dtype=torch.long),
                     torch.as_tensor([], dtype=torch.long)) for _ in range(bs)]

        # focal classification cost
        neg = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
        pos = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos[:, tgt_ids] - neg[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                         box_cxcywh_to_xyxy(tgt_bbox))

        C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class
             + self.cost_giou * cost_giou).view(bs, q, -1).cpu()

        sizes = [len(t["boxes"]) for t in targets]
        indices = [linear_sum_assignment(c[i])
                   for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.long),
                 torch.as_tensor(j, dtype=torch.long)) for i, j in indices]
