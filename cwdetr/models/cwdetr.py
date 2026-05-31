"""CW-DETR — the assembled multi-task model.

    image ─► DINOv3 backbone ─► multi-scale projector ─► shared deformable decoder
                                                              │
        ┌───────────────┬───────────────┬──────────────┬─────┴───────────┐
        ▼               ▼               ▼              ▼                 ▼
   detection      track queries   segmentation    sign cls         trajectory
   (cls + box)    (MOTR-style)   (lane/drivable)  (ROI crops)      (multimodal)

The backbone + decoder run **once**; every head reads the shared decoder output
(or shared feature maps). That sharing is the whole point — five tasks for not
much more than the cost of one detector.

Batching: detection / segmentation / sign run for any batch size. Query-based
tracking and trajectory are *online* (batch size 1, one video stream), which is
how MOT is actually deployed; ``track_instances`` carries state across frames.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from cwdetr.config import CWDETRConfig
from cwdetr.models.backbone import DINOv3Backbone
from cwdetr.models.projector import MultiScaleProjector
from cwdetr.models.decoder.deformable_decoder import DeformableTransformer
from cwdetr.models.heads import (DetectionHead, TrackQueryHead, TrackInstances,
                                 SegmentationHead, SignClassificationHead,
                                 TrajectoryHead)
from cwdetr.utils.box_ops import box_cxcywh_to_xyxy


class CWDETR(nn.Module):
    def __init__(self, cfg: CWDETRConfig):
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        hd = m.hidden_dim
        self.input_hw = (cfg.input.height, cfg.input.width)

        # ---- shared trunk -------------------------------------------------- #
        self.backbone = DINOv3Backbone(m.backbone)
        self.projector = MultiScaleProjector(
            in_channels=self.backbone.out_channels,
            in_strides=self.backbone.out_strides,
            hidden_dim=hd, num_levels=m.decoder.num_feature_levels)
        self.finest_stride = self.backbone.out_strides[0]
        self.transformer = DeformableTransformer(
            m.decoder, num_classes=m.heads.detection.num_classes)

        # ---- heads --------------------------------------------------------- #
        self.detection_head = DetectionHead(hd, m.heads.detection.num_classes,
                                            m.decoder.num_layers)
        # share box head with decoder for iterative refinement
        self.transformer.decoder.bbox_embed = self.detection_head.bbox_embed

        self.track_head = (TrackQueryHead(hd, m.heads.tracking.score_thresh,
                                          max_active=m.heads.tracking.max_active_tracks)
                           if m.heads.tracking.enabled and
                           m.heads.tracking.mode == "track_query" else None)
        self.seg_head = (SegmentationHead(hd, m.heads.segmentation.drivable_classes,
                                          m.heads.segmentation.lane_classes,
                                          m.heads.segmentation.mask_stride)
                         if m.heads.segmentation.enabled else None)
        self.sign_head = (SignClassificationHead(hd, m.heads.sign_classification.num_sign_classes,
                                                 m.heads.sign_classification.roi_size)
                          if m.heads.sign_classification.enabled else None)
        self.traj_head = (TrajectoryHead(hd, m.heads.trajectory.history_len,
                                         m.heads.trajectory.future_len,
                                         m.heads.trajectory.num_modes)
                          if m.heads.trajectory.enabled else None)
        self.source_det_class = m.heads.sign_classification.source_det_class

    # ----------------------------------------------------------------------- #
    def forward(self, images: torch.Tensor,
                track_instances: Optional[TrackInstances] = None,
                sign_rois: Optional[torch.Tensor] = None,
                traj_history: Optional[torch.Tensor] = None,
                run_seg: bool = True) -> Dict:
        """images: [B, 3, H, W] normalized.  Returns a dict of per-task outputs."""
        feats = self.backbone(images)              # list of maps
        srcs = self.projector(feats)               # num_levels maps @ hidden_dim

        # prepend track queries (online, B==1)
        extra_q = extra_qp = extra_ref = None
        num_track = 0
        if self.track_head is not None and track_instances is not None and len(track_instances) > 0:
            num_track = len(track_instances)
            extra_q = track_instances.query_embed[None]      # [1, T, C]
            extra_qp = track_instances.query_pos[None]
            extra_ref = track_instances.ref_points[None]     # [1, T, 4]

        dec = self.transformer(srcs, extra_q, extra_qp, extra_ref)
        det = self.detection_head(dec["hs"], dec["inter_references"])

        results: Dict = {
            "detection": det,
            "num_track": num_track,
            "enc_outputs": dec["enc_outputs"],
            "_srcs": srcs,                          # kept for loss-time ROI pooling
            "_image_hw": images.shape[-2:],
        }

        if self.seg_head is not None and run_seg:
            results["segmentation"] = self.seg_head(srcs, images.shape[-2:])

        # sign sub-classification (teacher-forced ROIs during training)
        if self.sign_head is not None and sign_rois is not None:
            results["sign_logits"] = self.sign_head(srcs[0], sign_rois, self.finest_stride)

        # trajectory for the propagated track queries
        if self.traj_head is not None and num_track > 0 and traj_history is not None:
            track_embed = det["query_embed"][0, :num_track]   # [T, C]  (B==1)
            traj, mode_logits = self.traj_head(track_embed, traj_history)
            results["trajectory"] = {"traj": traj, "mode_logits": mode_logits}

        return results

    # ----------------------------------------------------------------------- #
    @torch.no_grad()
    def step_track(self, results: Dict, prev_tracks: Optional[TrackInstances]) -> TrackInstances:
        """Advance the online tracker by one frame (B==1)."""
        assert self.track_head is not None
        det = results["detection"]
        logits, boxes = det["pred_logits"][0], det["pred_boxes"][0]    # [Lq, C], [Lq, 4]
        scores = logits.sigmoid().max(-1).values
        embeds = det["query_embed"][0]
        if prev_tracks is None:
            prev_tracks = TrackInstances.empty(self.cfg.model.hidden_dim, boxes.device)
        return self.track_head.update(prev_tracks, embeds, boxes, scores,
                                      results["num_track"])

    @torch.no_grad()
    def detected_sign_rois(self, results: Dict, score_thresh: float = 0.4) -> torch.Tensor:
        """Build ROIs (image coords) from boxes the detector called 'traffic_sign'."""
        det = results["detection"]
        logits, boxes = det["pred_logits"], det["pred_boxes"]          # [B, Lq, *]
        h, w = results.get("_image_hw", self.input_hw)
        rois = []
        for b in range(logits.shape[0]):
            scores = logits[b].sigmoid()
            cls = scores.argmax(-1)
            keep = (cls == self.source_det_class) & (scores.max(-1).values > score_thresh)
            xyxy = box_cxcywh_to_xyxy(boxes[b][keep]) * torch.tensor([w, h, w, h], device=boxes.device)
            if xyxy.numel():
                bidx = torch.full((xyxy.shape[0], 1), float(b), device=boxes.device)
                rois.append(torch.cat([bidx, xyxy], dim=1))
        return torch.cat(rois, 0) if rois else boxes.new_zeros((0, 5))


def build_cwdetr(cfg: CWDETRConfig) -> CWDETR:
    return CWDETR(cfg)
