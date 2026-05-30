"""Typed configuration for CW-DETR, loaded from the YAML files in configs/.

Dataclasses mirror the YAML schema. ``load_config`` parses a YAML file into a
``CWDETRConfig`` while tolerating missing keys (defaults below) and ignoring any
unknown keys, so configs can evolve without breaking older checkpoints.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional

import yaml


# --------------------------------------------------------------------------- #
# Leaf configs
# --------------------------------------------------------------------------- #
@dataclass
class InputCfg:
    height: int = 384
    width: int = 640
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class BackboneCfg:
    type: str = "dinov3_convnext"          # dinov3_convnext | dinov3_vit
    hf_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    # ConvNeXt path
    out_indices: List[int] = field(default_factory=lambda: [1, 2, 3])
    out_channels: List[int] = field(default_factory=lambda: [192, 384, 768])
    out_strides: List[int] = field(default_factory=lambda: [8, 16, 32])
    freeze_stages: int = 0
    # ViT path
    embed_dim: int = 768
    depth: int = 12
    num_register_tokens: int = 4
    out_layer_indices: List[int] = field(default_factory=lambda: [5, 8, 11])
    simple_fpn: bool = True
    windowed_attention: bool = True
    window_block_indices: List[int] = field(default_factory=list)
    window_size: int = 8
    freeze_blocks: int = 0
    # shared
    drop_path_rate: float = 0.1
    gram_anchor_distill: bool = False
    teacher_hf_name: Optional[str] = None


@dataclass
class ProjectorCfg:
    type: str = "c2f_multiscale"
    num_levels: int = 3
    hidden_dim: int = 256


@dataclass
class DecoderCfg:
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 3
    dim_feedforward: int = 1024
    num_queries: int = 300
    num_feature_levels: int = 3
    dec_n_points: int = 4
    two_stage: bool = True
    look_forward_twice: bool = True
    box_noise_scale: float = 1.0


@dataclass
class DetectionHeadCfg:
    enabled: bool = True
    num_classes: int = 13


@dataclass
class TrackingCfg:
    enabled: bool = True
    mode: str = "track_query"              # track_query | bytetrack_only
    max_active_tracks: int = 200
    score_thresh: float = 0.5
    match_thresh: float = 0.8


@dataclass
class SegmentationCfg:
    enabled: bool = True
    drivable_classes: int = 3
    lane_classes: int = 2
    mask_stride: int = 4


@dataclass
class SignClsCfg:
    enabled: bool = True
    num_sign_classes: int = 43
    source_det_class: int = 11
    roi_size: int = 7


@dataclass
class TrajectoryCfg:
    enabled: bool = True
    history_len: int = 10
    future_len: int = 12
    step_dt: float = 0.5
    num_modes: int = 6
    space: str = "image"                   # image | bev


@dataclass
class HeadsCfg:
    detection: DetectionHeadCfg = field(default_factory=DetectionHeadCfg)
    tracking: TrackingCfg = field(default_factory=TrackingCfg)
    segmentation: SegmentationCfg = field(default_factory=SegmentationCfg)
    sign_classification: SignClsCfg = field(default_factory=SignClsCfg)
    trajectory: TrajectoryCfg = field(default_factory=TrajectoryCfg)


@dataclass
class ModelCfg:
    hidden_dim: int = 256
    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    projector: ProjectorCfg = field(default_factory=ProjectorCfg)
    decoder: DecoderCfg = field(default_factory=DecoderCfg)
    heads: HeadsCfg = field(default_factory=HeadsCfg)


@dataclass
class DeployCfg:
    target: str = "jetson_orin_nano"
    precision: str = "int8"                # int8 | fp16 | fp32
    fp16_fallback: bool = True
    max_workspace_gb: int = 4
    calib_images: int = 500
    expected_fps_int8: Optional[int] = None
    expected_fps_fp16: Optional[int] = None


@dataclass
class CWDETRConfig:
    name: str = "cwdetr-nano"
    description: str = ""
    input: InputCfg = field(default_factory=InputCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    deploy: DeployCfg = field(default_factory=DeployCfg)


# --------------------------------------------------------------------------- #
# Recursive dict -> dataclass (defaults applied, unknown keys dropped)
# --------------------------------------------------------------------------- #
def _from_dict(cls, data: Optional[Dict[str, Any]]):
    if data is None:
        return cls()
    if not is_dataclass(cls):
        return data
    kwargs: Dict[str, Any] = {}
    type_hints = {f.name: f.type for f in fields(cls)}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = type_hints[f.name]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(ftype, value)
        else:
            kwargs[f.name] = value
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        # Non-fatal: surfaces typos without crashing training jobs.
        print(f"[config] WARN: ignoring unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**kwargs)


def load_config(path: str) -> CWDETRConfig:
    """Load a YAML config file into a typed ``CWDETRConfig``."""
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    return _from_dict(CWDETRConfig, raw)


def config_to_dict(cfg: CWDETRConfig) -> Dict[str, Any]:
    """Round-trip back to a plain dict (for logging / checkpoint metadata)."""
    return dataclasses.asdict(cfg)
