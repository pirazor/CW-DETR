from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn as nn

from cwdetr.export.export_onnx import CWDETRInfer
from cwdetr.export.export_sign_sidecar import SignSidecarInfer
from cwdetr.export.jetson_infer import build_sign_rois, classify_signs
from cwdetr.export.profile_jetson import _extract_latency_ms, profile


class _CoreModel(nn.Module):
    def forward(self, image, run_seg=True):
        batch = image.shape[0]
        return {
            "detection": {
                "pred_logits": torch.zeros(batch, 2, 3),
                "pred_boxes": torch.zeros(batch, 2, 4),
            },
            "segmentation": {
                "drivable_logits": torch.zeros(batch, 2, 2, 2),
                "lane_logits": torch.zeros(batch, 2, 2, 2),
            },
            "_srcs": [torch.ones(batch, 4, 2, 2)],
        }


class _SignHead(nn.Module):
    def forward(self, features, rois, feat_stride):
        return torch.ones(rois.shape[0], feat_stride)


class _SignRuntime:
    def infer(self, _):
        logits = np.zeros((2, 3), dtype=np.float32)
        logits[:, 2] = 1.0
        return {"sign_logits": logits.ravel()}


def test_core_export_wrapper_optionally_exposes_sign_features():
    image = torch.zeros(1, 3, 4, 4)
    basic = CWDETRInfer(_CoreModel())(image)
    with_features = CWDETRInfer(_CoreModel(), include_sign_features=True)(image)
    assert len(basic) == 4
    assert len(with_features) == 5
    assert with_features[-1].shape == (1, 4, 2, 2)


def test_sign_sidecar_wrapper_and_roi_padding():
    sidecar = SignSidecarInfer(_SignHead(), feat_stride=8)
    logits = sidecar(torch.zeros(1, 4, 2, 2), torch.zeros(3, 5))
    assert logits.shape == (3, 8)

    boxes = np.asarray([[0, 0, 2, 2], [1, 1, 3, 3], [2, 2, 4, 4]], dtype=np.float32)
    classes = np.asarray([0, 11, 11])
    rois, count = build_sign_rois(boxes, classes, max_rois=1)
    assert count == 1
    assert rois.shape == (1, 5)
    assert np.allclose(rois[0, 1:], boxes[1])
    typed = classify_signs(_SignRuntime(), np.zeros(1), boxes, classes,
                           max_rois=2, num_classes=3)
    assert typed.tolist() == [2, 2]


def test_jetson_profile_dry_run_records_hashes_and_commands(tmp_path):
    core = tmp_path / "core.plan"
    sign = tmp_path / "sign.plan"
    core.write_bytes(b"core")
    sign.write_bytes(b"sign")
    manifest = profile(str(core), str(tmp_path / "profile.json"), precision="int8",
                       sign_engine=str(sign), dry_run=True)
    assert len(manifest["profiles"]) == 2
    assert manifest["profiles"][0]["sha256"] == hashlib.sha256(b"core").hexdigest()
    assert "--int8" in manifest["profiles"][0]["command"]
    assert "--fp16" in manifest["profiles"][0]["command"]
    assert manifest["profiles"][1]["precision"] == "fp16"


def test_trtexec_latency_parser():
    parsed = _extract_latency_ms(
        "GPU Compute Time: min = 1 ms, max = 4 ms, mean = 2.5 ms, "
        "median = 2.0 ms, percentile(95%) = 3.8 ms")
    assert parsed == {"p50_ms": 2.0, "p95_ms": 3.8, "mean_ms": 2.5}
