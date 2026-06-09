"""Multi-task loss for CW-DETR.

Composes five task losses and balances them with learned homoscedastic
uncertainty weights (Kendall, Gal & Cipolla, 2018): each task contributes
``0.5 * exp(-s) * L + 0.5 * s`` where ``s`` is a learnable log-variance. This
avoids hand-tuning five loss coefficients and lets the network down-weight
tasks it is currently uncertain about — important when batches mix datasets that
only carry labels for a subset of tasks.

Per-task losses
  detection   : sigmoid-focal class + L1 + GIoU, with deep-supervision aux losses
  segmentation: cross-entropy + Dice, for drivable area and lane lines
  sign        : cross-entropy over fine sign classes
  trajectory  : winner-takes-all min-ADE regression + mode classification
  distill     : cosine feature distillation from the frozen DINOv3 teacher
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cwdetr.models.matcher import HungarianMatcher
from cwdetr.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss.sum() if reduction == "sum" else loss.mean()


def dice_loss(logits, targets, eps=1.0):
    """logits/targets: [B, C, H, W] one-hot targets. Soft Dice over classes."""
    prob = logits.softmax(1)
    num = 2 * (prob * targets).sum((0, 2, 3))
    den = prob.sum((0, 2, 3)) + targets.sum((0, 2, 3))
    return (1 - (num + eps) / (den + eps)).mean()


class UncertaintyWeighter(nn.Module):
    def __init__(self, task_names: List[str]):
        super().__init__()
        self.task_names = task_names
        self.log_vars = nn.Parameter(torch.zeros(len(task_names)))

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for i, name in enumerate(self.task_names):
            if name in losses and losses[name] is not None:
                s = self.log_vars[i]
                total = total + 0.5 * torch.exp(-s) * losses[name] + 0.5 * s
        return total


class MultiTaskCriterion(nn.Module):
    def __init__(self, num_classes: int, teacher_dim: Optional[int] = None,
                 student_dim: int = 256, distill_weight: float = 1.0,
                 focal_alpha: float = 0.25, dn_loss_weight: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(focal_alpha=focal_alpha)
        self.focal_alpha = focal_alpha
        self.det_weights = {"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0}
        self.distill_weight = distill_weight
        self.dn_loss_weight = dn_loss_weight
        self.weighter = UncertaintyWeighter(
            ["detection", "segmentation", "sign", "trajectory"])
        if teacher_dim is not None:
            self.distill_proj = nn.Conv2d(student_dim, teacher_dim, 1)
        else:
            self.distill_proj = None

    # ---- detection -------------------------------------------------------- #
    @staticmethod
    def _src_perm(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _select_detection_batch(det_out, keep):
        selected = {
            "pred_logits": det_out["pred_logits"][keep],
            "pred_boxes": det_out["pred_boxes"][keep],
        }
        selected["aux_outputs"] = [
            {"pred_logits": aux["pred_logits"][keep], "pred_boxes": aux["pred_boxes"][keep]}
            for aux in det_out.get("aux_outputs", [])
        ]
        return selected

    def loss_detection(self, det_out, targets) -> Dict[str, torch.Tensor]:
        keep = [i for i, target in enumerate(targets)
                if target.get("train_detection", True)]
        if len(keep) != len(targets):
            det_out = self._select_detection_batch(det_out, keep)
            targets = [targets[i] for i in keep]
        if not targets:
            zero = det_out["pred_logits"].sum() * 0.0 + det_out["pred_boxes"].sum() * 0.0
            return {"detection": zero, "det/loss_ce": zero,
                    "det/loss_bbox": zero, "det/loss_giou": zero,
                    "det/precision": zero, "det/mean_score": zero,
                    "det/num_targets": zero}

        all_outputs = [{"pred_logits": det_out["pred_logits"],
                        "pred_boxes": det_out["pred_boxes"]}] + det_out.get("aux_outputs", [])
        target_count = sum(len(t["labels"]) for t in targets)
        num_boxes = max(1, target_count)
        total = {"loss_ce": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
        stats = {
            "precision": det_out["pred_logits"].new_zeros(()),
            "mean_score": det_out["pred_logits"].new_zeros(()),
            "num_targets": det_out["pred_logits"].new_tensor(float(target_count)),
        }
        for output_index, out in enumerate(all_outputs):
            indices = self.matcher(out, targets)
            bidx, sidx = self._src_perm(indices)

            # classification (focal, one-hot over matched)
            logits = out["pred_logits"]
            target_classes = torch.full(logits.shape[:2], self.num_classes,
                                        dtype=torch.long, device=logits.device)
            tgt_lbl = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)]) \
                if len(bidx) else torch.zeros(0, dtype=torch.long, device=logits.device)
            if len(bidx):
                target_classes[bidx, sidx] = tgt_lbl
            if output_index == 0:
                with torch.no_grad():
                    if len(bidx):
                        matched_prob = logits.sigmoid()[bidx, sidx]
                        scores, pred_labels = matched_prob.max(-1)
                        stats["precision"] = (pred_labels == tgt_lbl).float().mean()
                        stats["mean_score"] = scores.mean()
            onehot = F.one_hot(target_classes, self.num_classes + 1)[..., :-1].float()
            total["loss_ce"] += sigmoid_focal_loss(
                logits, onehot, self.focal_alpha) / num_boxes

            # boxes
            if len(bidx):
                src_boxes = out["pred_boxes"][bidx, sidx]
                tgt_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)])
                total["loss_bbox"] += F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / num_boxes
                total["loss_giou"] += (1 - torch.diag(generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes)))).sum() / num_boxes
        det_loss = sum(self.det_weights[k] * v for k, v in total.items())
        return {"detection": det_loss,
                **{f"det/{key}": value for key, value in total.items()},
                "det/precision": stats["precision"],
                "det/mean_score": stats["mean_score"],
                "det/num_targets": stats["num_targets"]}

    def loss_dn_detection(self, dn_out, meta) -> Dict[str, torch.Tensor]:
        all_outputs = [{"pred_logits": dn_out["pred_logits"],
                        "pred_boxes": dn_out["pred_boxes"]}] + dn_out.get("aux_outputs", [])
        valid = meta["valid"]
        num_boxes = max(1, int(valid.sum()))
        labels = meta["labels"][valid]
        target_boxes = meta["boxes"][valid]
        total = {"loss_ce": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
        for output in all_outputs:
            logits = output["pred_logits"][valid]
            boxes = output["pred_boxes"][valid]
            if logits.numel() == 0:
                zero = output["pred_logits"].sum() * 0.0
                total = {key: value + zero for key, value in total.items()}
                continue
            onehot = F.one_hot(labels, self.num_classes).float()
            total["loss_ce"] += sigmoid_focal_loss(
                logits, onehot, self.focal_alpha) / num_boxes
            total["loss_bbox"] += F.l1_loss(boxes, target_boxes, reduction="sum") / num_boxes
            total["loss_giou"] += (1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(target_boxes)))).sum() / num_boxes
        detection = sum(self.det_weights[key] * value for key, value in total.items())
        return {"dn/detection": detection,
                **{f"dn/{key}": value for key, value in total.items()}}

    # ---- segmentation ----------------------------------------------------- #
    def loss_segmentation(self, seg_out, drivable_gt, lane_gt) -> torch.Tensor:
        loss = 0.0
        if drivable_gt is not None:
            dl = seg_out["drivable_logits"]
            dl = F.interpolate(dl, size=drivable_gt.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss + F.cross_entropy(dl, drivable_gt)
            loss = loss + dice_loss(dl, F.one_hot(drivable_gt, dl.shape[1]).permute(0, 3, 1, 2).float())
        if lane_gt is not None:
            ll = seg_out["lane_logits"]
            ll = F.interpolate(ll, size=lane_gt.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss + F.cross_entropy(ll, lane_gt)
            loss = loss + dice_loss(ll, F.one_hot(lane_gt, ll.shape[1]).permute(0, 3, 1, 2).float())
        return loss

    # ---- sign ------------------------------------------------------------- #
    @staticmethod
    def loss_sign(sign_logits, sign_labels) -> torch.Tensor:
        if sign_logits.numel() == 0:
            return sign_logits.new_zeros(())
        return F.cross_entropy(sign_logits, sign_labels)

    # ---- trajectory (winner-takes-all) ------------------------------------ #
    @staticmethod
    def loss_trajectory(traj_out, fut_gt, fut_mask) -> torch.Tensor:
        traj = traj_out["traj"]                      # [N, M, T, 2]
        logits = traj_out["mode_logits"]             # [N, M]
        if traj.shape[0] == 0:
            return traj.new_zeros(())
        err = ((traj - fut_gt[:, None]) ** 2).sum(-1).sqrt()       # [N, M, T]
        err = (err * fut_mask[:, None]).sum(-1) / fut_mask.sum(-1).clamp(min=1)[:, None]
        best = err.argmin(1)                                       # [N]
        reg = err[torch.arange(err.shape[0], device=err.device), best].mean()  # min-ADE
        cls = F.cross_entropy(logits, best)
        return reg + cls

    # ---- distillation ----------------------------------------------------- #
    def loss_distill(self, student_feat, teacher_feat) -> torch.Tensor:
        if student_feat is None or teacher_feat is None or self.distill_proj is None:
            return self.weighter.log_vars.new_zeros(())
        s = self.distill_proj(student_feat)
        if s.shape[-2:] != teacher_feat.shape[-2:]:
            s = F.interpolate(s, size=teacher_feat.shape[-2:], mode="bilinear", align_corners=False)
        s = F.normalize(s, dim=1)
        t = F.normalize(teacher_feat, dim=1)
        return (1 - (s * t).sum(1)).mean()

    # ---- compose ---------------------------------------------------------- #
    def forward(self, outputs: Dict, targets: Dict,
                student_feat=None, teacher_feat=None) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        det = self.loss_detection(outputs["detection"], targets["detection"])
        losses.update(det)
        enc_out = outputs.get("enc_outputs")
        if enc_out and "pred_logits" in enc_out:
            enc = self.loss_detection(enc_out, targets["detection"])
            losses["detection"] = losses["detection"] + enc["detection"]
            losses.update({key.replace("det/", "enc_det/"): value
                           for key, value in enc.items() if key.startswith("det/")})
        if "dn_outputs" in outputs:
            dn = self.loss_dn_detection(outputs["dn_outputs"], outputs["dn_meta"])
            losses["detection"] = losses["detection"] + self.dn_loss_weight * dn["dn/detection"]
            losses.update(dn)

        has_seg_target = targets.get("drivable") is not None or targets.get("lane") is not None
        losses["segmentation"] = (self.loss_segmentation(
            outputs["segmentation"], targets.get("drivable"), targets.get("lane"))
            if "segmentation" in outputs and has_seg_target else None)
        losses["sign"] = (self.loss_sign(outputs["sign_logits"], targets["sign_labels"])
                          if "sign_logits" in outputs and "sign_labels" in targets else None)
        losses["trajectory"] = (self.loss_trajectory(
            outputs["trajectory"], targets["future"], targets["future_mask"])
            if "trajectory" in outputs and "future" in targets else None)

        weighted = self.weighter({k: losses[k] for k in
                                  ["detection", "segmentation", "sign", "trajectory"]})
        distill = self.loss_distill(student_feat, teacher_feat) * self.distill_weight
        losses["distill"] = distill
        losses["total"] = weighted + distill
        return losses
