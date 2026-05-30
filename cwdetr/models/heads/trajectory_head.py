"""Multimodal trajectory / path-prediction head.

Conditioned on (a) a tracked object's decoder query embedding — which already
encodes appearance, class and current position — and (b) its short motion history
from the tracker, this head predicts ``num_modes`` plausible future paths plus a
probability over modes (Wayformer / MTR-lite). Trained winner-takes-all: only the
closest mode to the GT future is regressed, which avoids mode collapse.

Outputs future positions as cumulative per-step offsets (cumsum) for smoothness,
in image plane (Nano tier) or BEV/ego frame (Base tier, nuScenes).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from cwdetr.models.decoder.deformable_decoder import MLP


class TrajectoryHead(nn.Module):
    def __init__(self, hidden_dim: int, history_len: int = 10, future_len: int = 12,
                 num_modes: int = 6):
        super().__init__()
        self.future_len = future_len
        self.num_modes = num_modes

        self.history_encoder = nn.GRU(input_size=2, hidden_size=hidden_dim,
                                      num_layers=1, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim))
        self.mode_query = nn.Embedding(num_modes, hidden_dim)
        self.traj_decoder = MLP(hidden_dim, hidden_dim, future_len * 2, 3)
        self.mode_head = nn.Linear(hidden_dim, 1)

    def forward(self, query_embed: torch.Tensor,
                history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query_embed: [N, C]  per-track decoder embedding
        history:     [N, history_len, 2]  past (x,y), zero-padded if short
        returns:     trajectories [N, num_modes, future_len, 2],
                     mode_logits  [N, num_modes]
        """
        n = query_embed.shape[0]
        if n == 0:
            return (query_embed.new_zeros((0, self.num_modes, self.future_len, 2)),
                    query_embed.new_zeros((0, self.num_modes)))

        _, h_n = self.history_encoder(history)            # h_n: [1, N, C]
        motion = h_n[-1]                                  # [N, C]
        ctx = self.fuse(torch.cat([query_embed, motion], dim=-1))   # [N, C]

        modes = self.mode_query.weight[None] + ctx[:, None]         # [N, M, C]
        deltas = self.traj_decoder(modes).view(n, self.num_modes, self.future_len, 2)
        trajectories = torch.cumsum(deltas, dim=2)        # offsets -> positions
        mode_logits = self.mode_head(modes).squeeze(-1)   # [N, M]
        return trajectories, mode_logits
