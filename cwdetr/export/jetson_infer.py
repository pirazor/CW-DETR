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
from cwdetr.utils.box_ops import box_cxcywh_to_xyxy

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
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings, self.host, self.dev = [], {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(abs(np.prod(shape)))
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            self.host[name], self.dev[name] = host_mem, dev_mem
            self.bindings.append(int(dev_mem))
            self.ctx.set_tensor_address(name, int(dev_mem))

    def infer(self, image_chw: np.ndarray):
        inp = self.engine.get_tensor_name(0)
        np.copyto(self.host[inp], image_chw.ravel())
        cuda.memcpy_htod_async(self.dev[inp], self.host[inp], self.stream)
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
    xyxy = box_cxcywh_to_xyxy(__import__("torch").as_tensor(b[keep])).numpy()
    xyxy *= np.array([hw[1], hw[0], hw[1], hw[0]], np.float32)
    return xyxy, conf_v[keep], cls[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--track", action="store_true")
    a = ap.parse_args()

    model = TRTModel(a.engine)
    tracker = BYTETracker() if a.track else None

    # Demo loop on random frames; swap in a real capture (cv2.VideoCapture).
    for _ in range(100):
        frame = np.random.randn(1, 3, a.height, a.width).astype(np.float32)
        t0 = time.time()
        outs = model.infer(frame)
        names = list(outs.keys())
        xyxy, conf, cls = postprocess(outs[names[0]], outs[names[1]], (a.height, a.width))
        if tracker is not None:
            tracks = tracker.update(xyxy, conf, cls)
        dt = (time.time() - t0) * 1000
        print(f"{dt:.1f} ms  ({1000/max(dt,1e-3):.0f} FPS)  dets={len(conf)}")


if __name__ == "__main__":
    main()
