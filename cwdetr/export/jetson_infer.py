"""Minimal Jetson runtime: load a TensorRT engine, run the perception graph, and
(optionally) attach the ByteTrack associator for IDs. This is the deterministic
deploy path; the end-to-end track-query path runs the PyTorch model instead.

Usage (on device):
    python -m cwdetr.export.jetson_infer --engine cwdetr_nano_int8.plan \
        --video /dev/video0 --height 384 --width 640
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from cwdetr.tracking import BYTETracker

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except Exception:
    trt = None


class TRTModel:
    def __init__(self, engine_path):
        assert trt is not None, "TensorRT/pycuda not available."
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings, self.host, self.dev, self.inputs = [], {}, {}, []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            if any(dim < 0 for dim in shape):
                raise ValueError("dynamic TensorRT shapes are not supported by this batch-1 runtime")
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(abs(np.prod(shape)))
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            self.host[name], self.dev[name] = host_mem, dev_mem
            self.bindings.append(int(dev_mem))
            self.ctx.set_tensor_address(name, int(dev_mem))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
    def infer(self, inputs):
        if isinstance(inputs, np.ndarray):
            if len(self.inputs) != 1:
                raise ValueError(f"expected inputs {self.inputs}, got one array")
            inputs = {self.inputs[0]: inputs}
        if set(inputs) != set(self.inputs):
            raise ValueError(f"expected TensorRT inputs {self.inputs}, got {sorted(inputs)}")
        for name, value in inputs.items():
            np.copyto(self.host[name], value.ravel())
            cuda.memcpy_htod_async(self.dev[name], self.host[name], self.stream)
        self.ctx.execute_async_v3(self.stream.handle)
        outs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                cuda.memcpy_dtoh_async(self.host[name], self.dev[name], self.stream)
                outs[name] = self.host[name]
        self.stream.synchronize()
        return outs


def postprocess(scores, boxes, hw, conf=0.4, num_classes=13, topk=300):
    s = scores.reshape(-1, num_classes)
    b = boxes.reshape(-1, 4)
    cls = s.argmax(1)
    conf_v = s.max(1)
    keep = conf_v > conf
    b = b[keep]
    cls = cls[keep]
    conf_v = conf_v[keep]
    if len(conf_v) > topk:
        top = np.argpartition(conf_v, -topk)[-topk:]
        b, cls, conf_v = b[top], cls[top], conf_v[top]
    xyxy = np.column_stack((b[:, 0] - 0.5 * b[:, 2],
                            b[:, 1] - 0.5 * b[:, 3],
                            b[:, 0] + 0.5 * b[:, 2],
                            b[:, 1] + 0.5 * b[:, 3]))
    xyxy *= np.array([hw[1], hw[0], hw[1], hw[0]], np.float32)
    return xyxy, conf_v, cls


def build_sign_rois(boxes_xyxy, classes, max_rois=32, source_class=11):
    sign_boxes = boxes_xyxy[classes == source_class][:max_rois]
    rois = np.zeros((max_rois, 5), dtype=np.float32)
    rois[:len(sign_boxes), 1:] = sign_boxes
    return rois, len(sign_boxes)


def classify_signs(sign_model, sign_features, boxes_xyxy, classes,
                   max_rois=32, num_classes=43):
    rois, count = build_sign_rois(boxes_xyxy, classes, max_rois)
    if not count:
        return np.zeros(0, dtype=np.int64)
    outputs = sign_model.infer({"sign_features": sign_features, "rois": rois})
    logits = outputs["sign_logits"].reshape(max_rois, num_classes)
    return logits[:count].argmax(1)


def preprocess_bgr(frame: np.ndarray, hw) -> np.ndarray:
    import cv2
    h, w = hw
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / \
        np.array([0.229, 0.224, 0.225], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--track", action="store_true")
    ap.add_argument("--sign-engine", default=None,
                    help="optional FP16 sign-classifier sidecar TensorRT engine")
    ap.add_argument("--sign-max-rois", type=int, default=32)
    ap.add_argument("--sign-num-classes", type=int, default=43)
    ap.add_argument("--video", default=None,
                    help="video path or camera index; omit for a synthetic benchmark")
    ap.add_argument("--frames", type=int, default=100)
    a = ap.parse_args()

    model = TRTModel(a.engine)
    sign_model = TRTModel(a.sign_engine) if a.sign_engine else None
    tracker = BYTETracker() if a.track else None
    capture = None
    if a.video is not None:
        import cv2
        source = int(a.video) if a.video.isdigit() else a.video
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"failed to open video source: {a.video}")

    try:
        for _ in range(a.frames):
            if capture is None:
                frame = np.random.randn(1, 3, a.height, a.width).astype(np.float32)
            else:
                ok, bgr = capture.read()
                if not ok:
                    break
                frame = preprocess_bgr(bgr, (a.height, a.width))
            t0 = time.time()
            outs = model.infer(frame)
            xyxy, conf, cls = postprocess(
                outs["scores"], outs["boxes"], (a.height, a.width))
            tracks = tracker.update(xyxy, conf, cls) if tracker is not None else []
            typed_classes = np.zeros(0, dtype=np.int64)
            if sign_model is not None:
                if "sign_features" not in outs:
                    raise ValueError("core engine must export sign_features for the sign sidecar")
                typed_classes = classify_signs(
                    sign_model, outs["sign_features"], xyxy, cls,
                    a.sign_max_rois, a.sign_num_classes)
            dt = (time.time() - t0) * 1000
            print(f"{dt:.1f} ms  ({1000/max(dt,1e-3):.0f} FPS)  "
                  f"dets={len(conf)} tracks={len(tracks)} typed_signs={len(typed_classes)}")
    finally:
        if capture is not None:
            capture.release()


if __name__ == "__main__":
    main()
