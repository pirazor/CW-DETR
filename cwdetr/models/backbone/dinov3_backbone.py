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

import importlib
import os
from pathlib import Path
import sys
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
        self._encoder_channels: Optional[List[int]] = None

        if self.kind == "dinov3_vit" and cfg.windowed_attention:
            raise ValueError(
                "DINOv3 ViT windowed_attention is disabled until local-window RoPE "
                "handling is implemented correctly")

        self._load_encoder(cfg)

        # Output spec consumed by the projector.
        self.out_strides: List[int] = list(cfg.out_strides)
        if self.kind == "dinov3_vit":
            self.out_channels = (list(cfg.out_channels) if cfg.out_channels
                                 else [cfg.embed_dim] * len(self.out_strides))
            self.simple_fpn = SimpleFeaturePyramid(
                in_dim=cfg.embed_dim,
                out_dims=self.out_channels,
                strides=self.out_strides,
            )
        else:
            self.out_channels = list(self._encoder_channels or cfg.out_channels)

        self._maybe_freeze(cfg)

        # Optional frozen teacher for Gram-anchored feature distillation.
        self.teacher: Optional[nn.Module] = None
        if cfg.gram_anchor_distill:
            self._load_teacher(cfg)

    # ----- loading -------------------------------------------------------- #
    def _load_encoder(self, cfg: BackboneCfg):
        """Load from Hugging Face Transformers or Meta's official repository."""
        self._mode = None
        if cfg.source == "meta_hub":
            self.encoder = self._load_meta_model(
                cfg.meta_repo, cfg.meta_model or self._default_meta_model(cfg.type),
                cfg.weights or os.getenv("DINOV3_BACKBONE_WEIGHTS"),
                pretrained=cfg.pretrained, weights_hint="DINOV3_BACKBONE_WEIGHTS")
            self._mode = "meta_hub"
            return

        try:
            from transformers import AutoBackbone
            self.encoder = AutoBackbone.from_pretrained(
                cfg.hf_name, out_indices=(cfg.out_indices if self.kind == "dinov3_convnext"
                                          else [-1])
            )
            self._mode = "autobackbone"
            # ConvNeXt emits encoder stages directly. ViT channels describe the
            # generated simple-FPN outputs and must not be overwritten.
            if self.kind == "dinov3_convnext" and getattr(self.encoder, "channels", None):
                self._encoder_channels = list(self.encoder.channels)
        except Exception as exc:  # noqa: BLE001  (we deliberately degrade gracefully)
            print(f"[backbone] AutoBackbone unavailable ({exc}); using AutoModel path.")
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(cfg.hf_name)
            self._mode = "automodel"

    @staticmethod
    def _default_meta_model(kind: str) -> str:
        return "dinov3_convnext_tiny" if kind == "dinov3_convnext" else "dinov3_vitb16"

    @staticmethod
    def _resolve_meta_repo(meta_repo: str) -> str:
        repo = Path(os.getenv("DINOV3_REPO", meta_repo)).expanduser()
        if not repo.is_absolute():
            repo = Path(__file__).resolve().parents[3] / repo
        if not (repo / "hubconf.py").is_file():
            raise FileNotFoundError(
                f"DINOv3 repo not found at {repo}. Run setup_clone.sh or set DINOV3_REPO.")
        return str(repo.resolve())

    @classmethod
    def _load_meta_model(cls, meta_repo: str, meta_model: str,
                         weights: Optional[str], pretrained: bool = True,
                         weights_hint: str = "DINOV3_BACKBONE_WEIGHTS") -> nn.Module:
        if pretrained and not weights:
            raise ValueError(
                "Meta repository backend requires an approved checkpoint path or URL. "
                f"Set weights in the config or {weights_hint}, or use pretrained: false.")
        kwargs = {"pretrained": pretrained}
        if weights:
            kwargs["weights"] = weights
        repo = cls._resolve_meta_repo(meta_repo)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        backbones = importlib.import_module("dinov3.hub.backbones")
        try:
            factory = getattr(backbones, meta_model)
        except AttributeError as exc:
            raise ValueError(f"Unsupported Meta DINOv3 backbone: {meta_model}") from exc
        return factory(**kwargs)

    def _load_teacher(self, cfg: BackboneCfg):
        teacher_source = cfg.teacher_source or cfg.source
        if teacher_source == "meta_hub":
            weights = cfg.teacher_weights or os.getenv("DINOV3_TEACHER_WEIGHTS")
            self.teacher = self._load_meta_model(
                cfg.meta_repo, cfg.teacher_meta_model, weights,
                weights_hint="DINOV3_TEACHER_WEIGHTS")
            self._teacher_mode = "meta_hub"
        else:
            if not cfg.teacher_hf_name:
                raise ValueError("teacher_hf_name is required for Hugging Face distillation")
            from transformers import AutoModel
            self.teacher = AutoModel.from_pretrained(cfg.teacher_hf_name)
            self._teacher_mode = "huggingface"
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher_out_channels = self._feature_width(self.teacher)

    @staticmethod
    def _feature_width(model: nn.Module) -> int:
        config = getattr(model, "config", None)
        for owner in (config, model):
            for attr in ("hidden_size", "embed_dim", "num_features"):
                value = getattr(owner, attr, None)
                if isinstance(value, int):
                    return value
        raise ValueError("could not infer DINOv3 teacher feature width")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

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

    def _validate_features(self, x: torch.Tensor,
                           feats: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(feats) != len(self.out_channels) or len(feats) != len(self.out_strides):
            raise ValueError(
                "DINOv3 backbone feature-count mismatch: "
                f"got {len(feats)} maps, expected channels={self.out_channels} "
                f"at strides={self.out_strides}")
        h, w = x.shape[-2:]
        for index, (feat, channels, stride) in enumerate(
                zip(feats, self.out_channels, self.out_strides)):
            expected_hw = (h // stride, w // stride)
            if feat.ndim != 4 or feat.shape[1] != channels or feat.shape[-2:] != expected_hw:
                raise ValueError(
                    f"DINOv3 backbone map {index} violates contract: got "
                    f"{tuple(feat.shape)}, expected [B, {channels}, "
                    f"{expected_hw[0]}, {expected_hw[1]}] at stride {stride}")
        return feats

    @staticmethod
    def _select_convnext_features(x: torch.Tensor, hidden_states,
                                  out_channels: List[int],
                                  out_strides: List[int]) -> List[torch.Tensor]:
        """Resolve HF AutoModel stage maps by contract instead of tuple position.

        Transformers versions differ in whether ConvNeXt hidden states include
        an embedding/stem output. Matching channels and spatial stride avoids an
        off-by-one stage selection when AutoBackbone is unavailable.
        """
        candidates = [state for state in hidden_states
                      if torch.is_tensor(state) and state.ndim == 4]
        height, width = x.shape[-2:]
        selected = []
        for channels, stride in zip(out_channels, out_strides):
            expected_hw = (height // stride, width // stride)
            matches = [state for state in candidates
                       if state.shape[1] == channels and state.shape[-2:] == expected_hw]
            if not matches:
                observed = [tuple(state.shape) for state in candidates]
                raise ValueError(
                    "DINOv3 ConvNeXt AutoModel fallback cannot resolve feature "
                    f"[B, {channels}, {expected_hw[0]}, {expected_hw[1]}] at stride "
                    f"{stride}; observed hidden states: {observed}")
            # Prefer the final match if a version exposes both pre-stage and
            # post-stage tensors at the same resolution.
            selected.append(matches[-1])
        return selected

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """x: normalized image tensor [B, 3, H, W] -> list of feature maps."""
        h, w = x.shape[-2:]
        hp, wp = h // self.patch_size, w // self.patch_size

        if self.kind == "dinov3_convnext":
            if self._mode == "meta_hub":
                feats = list(self.encoder.get_intermediate_layers(
                    x, n=self.cfg.out_indices, reshape=True))
                return self._validate_features(x, feats)
            if self._mode == "autobackbone":
                feats = list(self.encoder(x).feature_maps)
            else:
                out = self.encoder(x, output_hidden_states=True)
                feats = self._select_convnext_features(
                    x, out.hidden_states, self.out_channels, self.out_strides)
            return self._validate_features(x, feats)

        # ----- ViT path: single stride-16 map -> simple feature pyramid ---- #
        if self._mode == "autobackbone":
            # Use the last requested layer's map (already [B, C, hp, wp]).
            fmap = self.encoder(x).feature_maps[-1]
        elif self._mode == "automodel":
            out = self.encoder(x, output_hidden_states=True)
            tokens = out.last_hidden_state
            fmap = self._vit_tokens_to_map(tokens, hp, wp)
        else:
            fmap = self.encoder.get_intermediate_layers(x, n=1, reshape=True)[-1]
        return self._validate_features(x, self.simple_fpn(fmap))

    @torch.no_grad()
    def teacher_features(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Dense patch features from the frozen DINOv3 teacher, [B, C, hp, wp].
        Used by the Gram-anchored feature-distillation loss (training only)."""
        if self.teacher is None:
            return None
        if self._teacher_mode == "meta_hub":
            return self.teacher.get_intermediate_layers(x, n=1, reshape=True)[-1]
        h, w = x.shape[-2:]
        hp, wp = h // self.patch_size, w // self.patch_size
        out = self.teacher(x, output_hidden_states=True)
        return self._vit_tokens_to_map(out.last_hidden_state, hp, wp)
