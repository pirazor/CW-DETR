"""Typed configuration for CW-DETR, loaded from the YAML files in configs/.

Dataclasses mirror the YAML schema. ``load_config`` parses a YAML file into a
``CWDETRConfig`` while tolerating missing keys (defaults below) and ignoring any
unknown keys, so configs can evolve without breaking older checkpoints.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, get_type_hints

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
class AugmentationCfg:
    enabled: bool = True
    scale_min: float = 0.8
    scale_max: float = 1.2
    crop_prob: float = 0.5
    hflip_prob: float = 0.5
    photometric_prob: float = 0.8
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05


@dataclass
class BackboneCfg:
    type: str = "dinov3_convnext"          # dinov3_convnext | dinov3_vit
    source: str = "huggingface"            # huggingface | meta_hub
    # Hugging Face Transformers backend
    hf_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    # Official facebookresearch/dinov3 repository backend. ``weights`` may be a
    # local checkpoint or an approved Meta URL. Environment overrides:
    # DINOV3_REPO and DINOV3_BACKBONE_WEIGHTS.
    meta_repo: str = "third_party/dinov3"
    meta_model: Optional[str] = None
    pretrained: bool = True
    weights: Optional[str] = None
    # ConvNeXt path
    out_indices: List[int] = field(default_factory=lambda: [1, 2, 3])
    out_channels: List[int] = field(default_factory=lambda: [192, 384, 768])
    out_strides: List[int] = field(default_factory=lambda: [8, 16, 32])
    train_backbone: bool = True
    freeze_stages: int = 0
    # ViT path
    embed_dim: int = 768
    depth: int = 12
    num_register_tokens: int = 4
    out_layer_indices: List[int] = field(default_factory=lambda: [5, 8, 11])
    simple_fpn: bool = True
    windowed_attention: bool = False
    window_block_indices: List[int] = field(default_factory=list)
    window_size: int = 8
    freeze_blocks: int = 0
    # shared
    drop_path_rate: float = 0.1
    gram_anchor_distill: bool = False
    teacher_hf_name: Optional[str] = None
    teacher_source: Optional[str] = None     # defaults to source
    teacher_meta_model: str = "dinov3_vitb16"
    teacher_weights: Optional[str] = None    # or DINOV3_TEACHER_WEIGHTS


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
    dn_enabled: bool = False
    dn_num_groups: int = 5
    label_noise_ratio: float = 0.2
    box_noise_scale: float = 1.0
    dn_loss_weight: float = 1.0


@dataclass
class DetectionHeadCfg:
    enabled: bool = True
    num_classes: int = 13


@dataclass
class TrackingCfg:
    enabled: bool = False
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
    enabled: bool = False
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
    augmentation: AugmentationCfg = field(default_factory=AugmentationCfg)
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
    # ``from __future__ import annotations`` stores dataclass field annotations
    # as strings. Resolve them before checking for nested dataclasses.
    type_hints = get_type_hints(cls)
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
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = _from_dict(CWDETRConfig, raw)
    validate_config(cfg)
    return cfg


def config_to_dict(cfg: CWDETRConfig) -> Dict[str, Any]:
    """Round-trip back to a plain dict (for logging / checkpoint metadata)."""
    return dataclasses.asdict(cfg)


def validate_config(cfg: CWDETRConfig) -> None:
    """Fail early on shape mismatches that would otherwise surface deep in a run."""
    m = cfg.model
    b = m.backbone
    d = m.decoder
    aug = cfg.augmentation
    if cfg.input.height <= 0 or cfg.input.width <= 0:
        raise ValueError("input height and width must be positive")
    if not 0 < aug.scale_min <= aug.scale_max:
        raise ValueError("augmentation scale_min and scale_max must be positive and ordered")
    for name in ("crop_prob", "hflip_prob", "photometric_prob"):
        if not 0 <= getattr(aug, name) <= 1:
            raise ValueError(f"augmentation {name} must be in [0, 1]")
    if m.hidden_dim != m.projector.hidden_dim or m.hidden_dim != d.hidden_dim:
        raise ValueError("model, projector, and decoder hidden_dim values must match")
    if d.hidden_dim % d.num_heads:
        raise ValueError("decoder hidden_dim must be divisible by num_heads")
    if d.num_feature_levels != m.projector.num_levels:
        raise ValueError("decoder num_feature_levels must match projector num_levels")
    if d.dn_enabled and not d.two_stage:
        raise ValueError("DN-DETR currently requires the two-stage decoder")
    if d.dn_num_groups <= 0:
        raise ValueError("decoder dn_num_groups must be positive")
    if not 0 <= d.label_noise_ratio <= 1:
        raise ValueError("decoder label_noise_ratio must be in [0, 1]")
    if d.box_noise_scale < 0 or d.dn_loss_weight < 0:
        raise ValueError("decoder box_noise_scale and dn_loss_weight must be non-negative")
    if len(b.out_channels) != len(b.out_strides):
        raise ValueError("backbone out_channels and out_strides must have the same length")
    if b.type not in ("dinov3_convnext", "dinov3_vit"):
        raise ValueError("backbone type must be 'dinov3_convnext' or 'dinov3_vit'")
    if b.type == "dinov3_vit" and b.windowed_attention:
        raise ValueError("DINOv3 ViT windowed_attention is not RoPE-correct and must remain disabled")
    if b.source not in ("huggingface", "meta_hub"):
        raise ValueError("backbone source must be 'huggingface' or 'meta_hub'")
    if b.teacher_source not in (None, "huggingface", "meta_hub"):
        raise ValueError("teacher_source must be 'huggingface', 'meta_hub', or null")
    if len(b.out_channels) > d.num_feature_levels:
        raise ValueError("backbone cannot emit more levels than the decoder consumes")
    if b.out_strides and any(
            size % max(b.out_strides) for size in (cfg.input.height, cfg.input.width)):
        raise ValueError("input height and width must be divisible by the coarsest backbone stride")
    det_classes = m.heads.detection.num_classes
    source_cls = m.heads.sign_classification.source_det_class
    if m.heads.sign_classification.enabled and not 0 <= source_cls < det_classes:
        raise ValueError("sign source_det_class must index the detection taxonomy")
