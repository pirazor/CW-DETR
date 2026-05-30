"""Query-based tracking head (MOTR / MOTRv2 style).

Tracking is folded into the detection query mechanism rather than bolted on as a
post-processor. Each frame, surviving *track queries* (one per active object) are
prepended to the fresh *object queries* before the decoder runs. A track query
keeps attending to the same physical object across frames, so identity is carried
implicitly — no IoU/Re-ID matching needed in the end-to-end path. A Query
Interaction Module (QIM) updates the surviving queries' embeddings between frames.

This module owns only the *temporal state machinery*. The per-frame class/box
predictions for track queries come from the shared DetectionHead, exactly like
object queries. A lightweight ByteTrack fallback (cwdetr/tracking/bytetrack.py)
is available for the deploy path where a deterministic associator is preferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from cwdetr.models.decoder.deformable_decoder import MLP


@dataclass
class TrackInstances:
    """Mutable per-frame set of active tracks."""
    query_embed: torch.Tensor      # [N, C]  content query for next frame
    query_pos: torch.Tensor        # [N, C]  positional query for next frame
    ref_points: torch.Tensor       # [N, 4]  last box (cx,cy,w,h), as next reference
    obj_ids: torch.Tensor          # [N]     persistent track ids
    scores: torch.Tensor           # [N]     last confidence
    disappear: torch.Tensor        # [N]     consecutive missed frames

    @classmethod
    def empty(cls, dim: int, device) -> "TrackInstances":
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        return cls(z(0, dim), z(0, dim), z(0, 4),
                   torch.zeros(0, dtype=torch.long, device=device), z(0), z(0))

    def __len__(self):
        return self.query_embed.shape[0]


class QueryInteractionModule(nn.Module):
    """Self-attention + FFN over surviving track queries (temporal refinement)."""

    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(inplace=True),
                                 nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        if q.shape[0] == 0:
            return q
        x = q[None]                                   # [1, N, C]
        a, _ = self.self_attn(x, x, x)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x[0]


class TrackQueryHead(nn.Module):
    def __init__(self, hidden_dim: int, score_thresh: float = 0.5,
                 miss_tolerance: int = 5, max_active: int = 200):
        super().__init__()
        self.dim = hidden_dim
        self.score_thresh = score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_active = max_active
        self.qim = QueryInteractionModule(hidden_dim)
        # Maps an updated query embedding to its next-frame positional query.
        self.pos_update = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self._next_id = 0

    def reset(self):
        self._next_id = 0

    def _new_ids(self, n: int, device) -> torch.Tensor:
        ids = torch.arange(self._next_id, self._next_id + n, device=device)
        self._next_id += n
        return ids

    @torch.no_grad()
    def update(self, prev: TrackInstances, query_embed: torch.Tensor,
               ref_points: torch.Tensor, scores: torch.Tensor,
               num_track: int) -> TrackInstances:
        """Build next frame's TrackInstances from this frame's decoder output.

        The first ``num_track`` queries were the propagated track queries; the
        rest are new object queries. Surviving track queries + newly-confident
        object queries become the next frame's tracks.
        """
        device = query_embed.device
        keep_track = scores[:num_track] > (self.score_thresh * 0.5)   # hysteresis
        new_det = scores[num_track:] > self.score_thresh

        # carry over surviving tracks
        surv_embed = query_embed[:num_track][keep_track]
        surv_ids = prev.obj_ids[keep_track] if len(prev) == num_track else \
            self._new_ids(int(keep_track.sum()), device)
        surv_ref = ref_points[:num_track][keep_track]
        surv_scores = scores[:num_track][keep_track]

        # spawn new tracks from confident detections
        new_embed = query_embed[num_track:][new_det]
        new_ref = ref_points[num_track:][new_det]
        new_scores = scores[num_track:][new_det]
        new_ids = self._new_ids(int(new_det.sum()), device)

        embed = torch.cat([surv_embed, new_embed], 0)
        ref = torch.cat([surv_ref, new_ref], 0)
        ids = torch.cat([surv_ids, new_ids], 0)
        sc = torch.cat([surv_scores, new_scores], 0)

        # cap active tracks
        if embed.shape[0] > self.max_active:
            top = torch.topk(sc, self.max_active)[1]
            embed, ref, ids, sc = embed[top], ref[top], ids[top], sc[top]

        # temporal refinement of the carried embeddings
        embed = self.qim(embed)
        query_pos = self.pos_update(embed)
        return TrackInstances(
            query_embed=embed, query_pos=query_pos, ref_points=ref,
            obj_ids=ids, scores=sc,
            disappear=torch.zeros(embed.shape[0], device=device))
