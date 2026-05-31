"""DINOv3 backbone for CW-DETR.

Replaces RF-DETR's DINOv2 (patch-14, interpolated positional embeddings) with
DINOv3 (patch-16, RoPE, register tokens, Gram-anchored dense features).

Why this is the core upgrade
----------------------------
* **Patch-16 vs patch-14** — at a given input size DINOv3 produces ~23% fewer
  tokens, so ViT self-attention (quadratic in token count) is markedly cheaper:
  a direct latency win on Jetson.
* **RoPE positional encoding** — DINOv3 uses rotary position embeddings instead
  of learned/interpolated ones, so the backbone is natively resolution-agnostic.
  RF-DETR had to interpolate DINOv2's positional grid for every new resolution;
  with DINOv3 we change input size freely (crucial for the wide ADAS aspect
  ratios and for multi-resolution training).
* **Gram anchoring** — DINOv3's training keeps patch-to-patch similarity stable,
  yielding far cleaner dense features. Dense quality is exactly what the
  segmentation and small-object detection heads consume.
* **ConvNeXt variants** — distilled from ViT-7B, fully convolutional, and far
  friendlier to TensorRT INT8 than a ViT. We default the Nano tier to
  ConvNeXt-Tiny for that reason.

Two backbone families are supported behind one interface:

  type == "dinov3_convnext" : hierarchical, returns maps at strides {8,16,32}.
  type == "dinov3_vit"      : single-scale (stride 16); a ViTDet-style "simple
                              feature pyramid" lifts it to {8,16,32}.

Both return a list of feature maps ``[B, C_l, H_l, W_l]`` (low-to-high stride),
which the projector then unifies to ``hidden_dim`` channels.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from cwdetr.config import BackboneCfg


# --------------------------------------------------------------------------- #
# ViTDet "simple feature pyramid" (Li et al., 2022): build a multi-scale
# pyramid from a single stride-16 ViT feature map via strided (de)convs.
# --------------------------------------------------------------------------- #
class SimpleFeaturePyramid(nn.Module):
    def __init__(self, in_dim: int, out_dims: List[int], strides: List[int]):
        super().__init__()
        assert len(out_dims) == len(strides)
        self.blocks = nn.ModuleList()
        for out_dim, stride in zip(out_dims, strides):
            if stride == 8:        # upsample x2  (16 -> 8)
                layer = nn.Sequential(
                    nn.ConvTranspose2d(in_dim, in_dim // 2, 2, stride=2),
                    nn.GroupNorm(16, in_dim // 2), nn.GELU(),
                    nn.Conv2d(in_dim // 2, out_dim, 1),
                )
            elif stride == 16:     # identity scale
                layer = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1))
            elif stride == 32:     # downsample x2 (16 -> 32)
                layer = nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
                    nn.GroupNorm(16, out_dim), nn.GELU(),
                )
            else:
                raise ValueError(f"Unsupported FPN stride {stride}")
            self.blocks.append(layer)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [blk(x) for blk in self.blocks]


class DINOv3Backbone(nn.Module):
    def __init__(self, cfg: BackboneCfg):
        super().__init__()
        self.cfg = cfg
        self.kind = cfg.type
        self.patch_size = 16
        self.num_register_tokens = cfg.num_register_tokens

        self._load_encoder(cfg)

        # Output spec consumed by the projector.
        self.out_strides: List[int] = list(cfg.out_strides)
        if self.kind == "dinov3_vit":
            self.simple_fpn = SimpleFeaturePyramid(
                in_dim=cfg.embed_dim,
                out_dims=list(cfg.out_channels) if cfg.out_channels else [cfg.embed_dim] * 3,
                strides=self.out_strides,
            )
            self.out_channels = list(cfg.out_channels)
        else:
            self.out_channels = list(cfg.out_channels)

        self._maybe_freeze(cfg)

        # Optional frozen teacher for Gram-anchored feature distillation.
        self.teacher: Optional[nn.Module] = None
        if cfg.gram_anchor_distill and cfg.teacher_hf_name:
            self._load_teacher(cfg.teacher_hf_name)

    # ----- loading -------------------------------------------------------- #
    def _load_encoder(self, cfg: BackboneCfg):
        """Prefer transformers.AutoBackbone (clean multi-scale .feature_maps);
        fall back to AutoModel + manual token reshape for ViT."""
        self._mode = None
        try:
            from transformers import AutoBackbone
            self.encoder = AutoBackbone.from_pretrained(
                cfg.hf_name, out_indices=cfg.out_indices
            )
            self._mode = "autobackbone"
            # Reconcile channel spec with what the model actually reports.
            if getattr(self.encoder, "channels", None):
                cfg.out_channels = list(self.encoder.channels)
        except Exception as exc:  # noqa: BLE001  (we deliberately degrade gracefully)
            print(f"[backbone] AutoBackbone unavailable ({exc}); using AutoModel path.")
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(cfg.hf_name)
            self._mode = "automodel"

        if cfg.windowed_attention and self.kind == "dinov3_vit":
            # Patch selected ViT blocks to compute attention within local windows
            # (RF-DETR / ViTDet scheme). See windowed_attention.convert_to_windowed.
            try:
                from cwdetr.models.backbone.windowed_attention import convert_to_windowed
                convert_to_windowed(
                    self.encoder,
                    window_block_indices=cfg.window_block_indices,
                    window_size=cfg.window_size,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[backbone] windowed-attention patch skipped ({exc}); "
                      f"using global attention.")

    def _load_teacher(self, teacher_hf_name: str):
        from transformers import AutoModel
        self.teacher = AutoModel.from_pretrained(teacher_hf_name)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def _maybe_freeze(self, cfg: BackboneCfg):
        n = cfg.freeze_stages if self.kind == "dinov3_convnext" else cfg.freeze_blocks
        if n <= 0:
            return
        # Freeze the patch embedding + the first ``n`` stages/blocks. Names vary
        # across transformers versions, so freeze by module-index heuristic.
        frozen = 0
        for name, module in self.encoder.named_children():
            for sub in module.modules():
                for p in sub.parameters(recurse=False):
                    p.requires_grad_(False)
            frozen += 1
            if frozen >= n:
                break

    # ----- forward -------------------------------------------------------- #
    def _vit_tokens_to_map(self, tokens: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        """[B, 1 + R + N, C] -> [B, C, hp, wp] dropping CLS + register tokens."""
        patches = tokens[:, 1 + self.num_register_tokens:, :]      # [B, N, C]
        b, n, c = patches.shape
        patches = patches.transpose(1, 2).reshape(b, c, hp, wp)    # [B, C, hp, wp]
        return patches.contiguous()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """x: normalized image tensor [B, 3, H, W] -> list of feature maps."""
        h, w = x.shape[-2:]
        hp, wp = h // self.patch_size, w // self.patch_size

        if self.kind == "dinov3_convnext":
            if self._mode == "autobackbone":
                feats = list(self.encoder(x).feature_maps)
            else:
                out = self.encoder(x, output_hidden_states=True)
                feats = [out.hidden_states[i] for i in self.cfg.out_indices]
            return feats

        # ----- ViT path: single stride-16 map -> simple feature pyramid ---- #
        if self.cfg.windowed_attention:
            from cwdetr.models.backbone.windowed_attention import set_grid_hw
            set_grid_hw(self.encoder, hp, wp)
        if self._mode == "autobackbone":
            # Use the last requested layer's map (already [B, C, hp, wp]).
            fmap = self.encoder(x).feature_maps[-1]
        else:
            out = self.encoder(x, output_hidden_states=True)
            tokens = out.hidden_states[self.cfg.out_layer_indices[-1]]
            fmap = self._vit_tokens_to_map(tokens, hp, wp)
        return self.simple_fpn(fmap)

    @torch.no_grad()
    def teacher_features(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Dense patch features from the frozen DINOv3 teacher, [B, C, hp, wp].
        Used by the Gram-anchored feature-distillation loss (training only)."""
        if self.teacher is None:
            return None
        h, w = x.shape[-2:]
        hp, wp = h // self.patch_size, w // self.patch_size
        out = self.teacher(x, output_hidden_states=True)
        return self._vit_tokens_to_map(out.last_hidden_state, hp, wp)
