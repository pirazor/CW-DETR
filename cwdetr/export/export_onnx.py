"""Export the CW-DETR per-frame graph to ONNX.

We export the *feed-forward perception graph* — backbone -> projector -> decoder
-> detection + segmentation heads — producing dense tensors. Tracking (track
queries / ByteTrack) and trajectory rollout are stateful and run on the host loop
around the engine, so they are intentionally excluded from the static graph.

The deformable attention uses ``grid_sample`` (ONNX opset >= 16), so no custom
TensorRT plugin is needed — this is why CW-DETR deploys to Jetson with the stock
TensorRT ONNX parser. Use opset 18 for current PyTorch exporters and keep the
Jetson engine static at batch size 1.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from cwdetr.config import load_config
from cwdetr.models.cwdetr import build_cwdetr


class CWDETRInfer(nn.Module):
    """Thin inference wrapper returning export-friendly tensors."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        out = self.model(image, run_seg=True)
        det = out["detection"]
        results = (det["pred_logits"].sigmoid(), det["pred_boxes"])
        if "segmentation" in out:
            results = results + (out["segmentation"]["drivable_logits"].argmax(1).to(torch.int32),
                                 out["segmentation"]["lane_logits"].argmax(1).to(torch.int32))
        return results


def export(config: str, ckpt: str, out_path: str, opset: int = 18):
    cfg = load_config(config)
    model = build_cwdetr(cfg).eval()
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd.get("model", sd), strict=False)

    wrapper = CWDETRInfer(model).eval()
    dummy = torch.randn(1, 3, cfg.input.height, cfg.input.width)

    names_out = ["scores", "boxes"]
    if cfg.model.heads.segmentation.enabled:
        names_out.extend(["drivable", "lane"])
    torch.onnx.export(
        wrapper, dummy, out_path,
        input_names=["image"], output_names=names_out,
        opset_version=opset, do_constant_folding=True, external_data=False,
    )
    print(f"exported ONNX -> {out_path}  (opset {opset})")
    try:
        import onnx
        from onnxslim import slim
        onnx.save(slim(onnx.load(out_path)), out_path)
        print("simplified ONNX graph in place with onnxslim.")
    except Exception as exc:  # noqa: BLE001
        print(f"warning: ONNX simplification skipped ({exc}).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="cwdetr.onnx")
    ap.add_argument("--opset", type=int, default=18)
    a = ap.parse_args()
    export(a.config, a.ckpt, a.out, a.opset)
