"""RF-DETR / ViTDet-style windowed attention for the DINOv3 ViT tier.

Global self-attention over a high-resolution ViT token grid is the dominant cost
of a ViT detector. RF-DETR (following ViTDet) keeps attention global only in a
few blocks and restricts the rest to non-overlapping local windows, which makes
cost grow linearly rather than quadratically with the number of tokens.

This module provides:
  * ``window_partition`` / ``window_unpartition`` — the standard grid<->windows ops.
  * ``convert_to_windowed`` — best-effort wrapper that makes the chosen DINOv3 ViT
    blocks attend within windows while leaving the remaining blocks global.

Caveats (intentionally explicit — this is research scaffolding)
---------------------------------------------------------------
* DINOv3 uses RoPE. When a block is windowed, rotary positions should ideally be
  recomputed per window. The wrapper below restricts the *token set* to a window
  but defers to the block's own positional handling; for production accuracy,
  prefer fine-tuning from the official ``facebookresearch/dinov3`` ViT (cloned by
  setup_clone.sh) where per-window RoPE can be wired in directly, or simply use
  the ConvNeXt backbone tier (no attention -> no windowing needed) on Jetson.
* CLS + register tokens are passed through windowed blocks unchanged and are
  refreshed by the global blocks.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def window_partition(x: torch.Tensor, window: int):
    """[B, H, W, C] -> ([B*nW, window, window, C], (Hp, Wp)) with bottom/right pad."""
    b, h, w, c = x.shape
    pad_h = (window - h % window) % window
    pad_w = (window - w % window) % window
    if pad_h or pad_w:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w
    x = x.view(b, hp // window, window, wp // window, window, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window, window, c)
    return windows, (hp, wp)


def window_unpartition(windows: torch.Tensor, window: int, pad_hw, hw):
    """Inverse of ``window_partition``; crops padding back to (H, W)."""
    hp, wp = pad_hw
    h, w = hw
    b = windows.shape[0] // (hp * wp // window // window)
    x = windows.view(b, hp // window, wp // window, window, window, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, hp, wp, -1)
    if hp != h or wp != w:
        x = x[:, :h, :w, :].contiguous()
    return x


class _WindowedBlock(nn.Module):
    """Wraps a ViT block so its attention runs within local windows.

    Assumes the wrapped block's ``forward`` takes hidden_states of shape
    ``[B, S + Hp*Wp, C]`` (S = 1 CLS + R register tokens) and returns the same
    shape. We split special vs. patch tokens, fold patches into windows, run the
    block per window (special tokens duplicated into each window so intra-window
    attention can still see global context), then scatter back.
    """

    def __init__(self, block: nn.Module, window: int, num_special: int):
        super().__init__()
        self.block = block
        self.window = window
        self.num_special = num_special
        self.grid_hw = None  # set by the backbone before each forward (Hp, Wp)

    def forward(self, hidden_states, *args, **kwargs):
        if self.grid_hw is None:
            return self.block(hidden_states, *args, **kwargs)
        hp, wp = self.grid_hw
        s = self.num_special
        special, patches = hidden_states[:, :s], hidden_states[:, s:]
        b, n, c = patches.shape
        patches = patches.view(b, hp, wp, c)
        windows, pad_hw = window_partition(patches, self.window)      # [b*nW, ws, ws, c]
        nw = windows.shape[0] // b
        win_tokens = windows.view(b * nw, self.window * self.window, c)
        # Broadcast special tokens into each window for global context.
        special_rep = special.repeat_interleave(nw, dim=0)
        seq = torch.cat([special_rep, win_tokens], dim=1)
        out = self.block(seq, *args, **kwargs)
        out = out[0] if isinstance(out, tuple) else out
        win_out = out[:, s:].reshape(b * nw, self.window, self.window, c)
        patches_out = window_unpartition(win_out, self.window, pad_hw, (hp, wp))
        patches_out = patches_out.reshape(b, hp * wp, c)
        # Average the per-window copies of the special tokens back to one set.
        special_out = out[:, :s].view(b, nw, s, c).mean(dim=1)
        return torch.cat([special_out, patches_out], dim=1)


def _find_blocks(model: nn.Module):
    """Locate the transformer block list across naming conventions."""
    for attr in ("blocks", "layer", "layers"):
        mod = model
        for part in attr.split("."):
            mod = getattr(mod, part, None)
            if mod is None:
                break
        if isinstance(mod, (nn.ModuleList, list)):
            return mod
    # HF often nests under .encoder
    enc = getattr(model, "encoder", None)
    if enc is not None:
        for attr in ("layer", "layers", "blocks"):
            mod = getattr(enc, attr, None)
            if isinstance(mod, (nn.ModuleList, list)):
                return mod
    raise AttributeError("Could not locate ViT transformer blocks for windowing.")


def convert_to_windowed(model: nn.Module, window_block_indices: List[int],
                        window_size: int, num_special: int = 5) -> nn.Module:
    """Replace the listed blocks with windowed wrappers (in place). ``num_special``
    defaults to 5 (1 CLS + 4 register tokens) for DINOv3."""
    blocks = _find_blocks(model)
    wrapped = 0
    for i in list(window_block_indices):
        if 0 <= i < len(blocks):
            blocks[i] = _WindowedBlock(blocks[i], window_size, num_special)
            wrapped += 1
    model._cwdetr_windowed_blocks = [b for b in blocks if isinstance(b, _WindowedBlock)]
    print(f"[windowed-attn] wrapped {wrapped} block(s) at window={window_size}.")
    return model


def set_grid_hw(model: nn.Module, hp: int, wp: int):
    """Backbone calls this each forward so windowed blocks know the token grid."""
    for blk in getattr(model, "_cwdetr_windowed_blocks", []):
        blk.grid_hw = (hp, wp)
