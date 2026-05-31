"""Export the conditional traffic-sign classifier as a static TensorRT sidecar."""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from cwdetr.config import load_config
from cwdetr.models.heads import SignClassificationHead


class SignSidecarInfer(nn.Module):
    """Static-capacity wrapper. The host slices logits to the valid ROI count."""

    def __init__(self, head: SignClassificationHead, feat_stride: int):
        super().__init__()
        self.head = head
        self.feat_stride = feat_stride

    def forward(self, sign_features, rois):
        return self.head(sign_features, rois, self.feat_stride)


def _load_head_state(head, ckpt):
    if not ckpt:
        return
    checkpoint = torch.load(ckpt, map_location="cpu")
    state = checkpoint.get("ema", checkpoint.get("model", checkpoint))
    prefix = "sign_head."
    head_state = {name[len(prefix):]: value for name, value in state.items()
                  if name.startswith(prefix)}
    if not head_state:
        raise ValueError("checkpoint does not contain sign_head parameters")
    head.load_state_dict(head_state)


def export(config: str, ckpt: str, out_path: str, max_rois: int = 32, opset: int = 18):
    cfg = load_config(config)
    model_cfg = cfg.model
    sign_cfg = model_cfg.heads.sign_classification
    if not sign_cfg.enabled:
        raise ValueError("sign classification must be enabled to export the sidecar")
    head = SignClassificationHead(model_cfg.hidden_dim, sign_cfg.num_sign_classes,
                                  sign_cfg.roi_size).eval()
    _load_head_state(head, ckpt)

    stride = model_cfg.backbone.out_strides[0]
    height, width = cfg.input.height // stride, cfg.input.width // stride
    features = torch.randn(1, model_cfg.hidden_dim, height, width)
    rois = torch.zeros(max_rois, 5)
    rois[:, 1:] = torch.tensor([0.0, 0.0, float(cfg.input.width), float(cfg.input.height)])
    wrapper = SignSidecarInfer(head, stride).eval()
    torch.onnx.export(
        wrapper, (features, rois), out_path,
        input_names=["sign_features", "rois"], output_names=["sign_logits"],
        opset_version=opset, do_constant_folding=True, external_data=False,
    )
    print(f"exported sign sidecar ONNX -> {out_path}  (max_rois={max_rois})")
    try:
        import onnx
        from onnxslim import slim
        onnx.save(slim(onnx.load(out_path)), out_path)
        print("simplified sign sidecar graph in place with onnxslim.")
    except Exception as exc:  # noqa: BLE001
        print(f"warning: ONNX simplification skipped ({exc}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--out", default="cwdetr_sign.onnx")
    parser.add_argument("--max-rois", type=int, default=32)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    export(args.config, args.ckpt, args.out, args.max_rois, args.opset)
