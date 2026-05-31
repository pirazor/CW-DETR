"""Build a TensorRT engine from the exported ONNX (run ON the Jetson, or any
machine with matching TensorRT). Supports FP16 (Orin NX tier) and INT8 PTQ with
entropy calibration (Orin Nano tier).

INT8 notes for CW-DETR
  * The ConvNeXt backbone tier quantizes cleanly to INT8.
  * Keep the deformable-attention sampling + the final detection/seg logits in
    FP16 (precision-sensitive). Use the per-layer precision API or mark them via
    ``--fp16-fallback`` (default) so TensorRT keeps unsupported/sensitive layers
    in FP16 automatically.

Usage (on device):
    python -m cwdetr.export.build_tensorrt --onnx cwdetr.onnx --precision int8 \
        --calib-dir /data/calib --out cwdetr_nano_int8.plan
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

try:
    import tensorrt as trt
except Exception:  # not present off-device
    trt = None


def _load_calib_image(path, h, w):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((w, h))
    x = np.asarray(img, np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = (x - mean) / std
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


if trt is not None:
    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, calib_dir, h, w, cache="calib.cache", max_images=500):
            super().__init__()
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
            self.cuda = cuda
            files = sorted(glob.glob(os.path.join(calib_dir, "*")))
            self.files = [p for p in files
                          if os.path.splitext(p)[1].lower() in
                          (".jpg", ".jpeg", ".png", ".bmp", ".webp")][:max_images]
            if not self.files:
                raise ValueError(f"no calibration images found in {calib_dir}")
            self.h, self.w, self.cache_path = h, w, cache
            self.idx = 0
            self.device_input = cuda.mem_alloc(int(np.prod((1, 3, h, w)) * 4))

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.idx >= len(self.files):
                return None
            batch = _load_calib_image(self.files[self.idx], self.h, self.w)
            self.cuda.memcpy_htod(self.device_input, batch)
            self.idx += 1
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if os.path.exists(self.cache_path):
                return open(self.cache_path, "rb").read()

        def write_calibration_cache(self, cache):
            open(self.cache_path, "wb").write(cache)


def build_engine(onnx_path, precision="fp16", calib_dir=None, hw=(384, 640),
                 workspace_gb=4, fp16_fallback=True, out="model.plan"):
    assert trt is not None, "TensorRT not available in this environment."
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if precision in ("fp16", "int8") and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        assert builder.platform_has_fast_int8 and calib_dir, "INT8 needs calib-dir + INT8 HW"
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = EntropyCalibrator(calib_dir, hw[0], hw[1])
        if fp16_fallback and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)   # let TRT keep sensitive layers in FP16

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build a serialized engine")
    with open(out, "wb") as f:
        f.write(serialized)
    print(f"built TensorRT engine -> {out}  ({precision})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--precision", choices=["fp16", "int8", "fp32"], default="fp16")
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--workspace-gb", type=int, default=4)
    ap.add_argument("--no-fp16-fallback", action="store_false", dest="fp16_fallback")
    ap.set_defaults(fp16_fallback=True)
    ap.add_argument("--out", default="model.plan")
    a = ap.parse_args()
    build_engine(a.onnx, a.precision, a.calib_dir, (a.height, a.width),
                 a.workspace_gb, a.fp16_fallback, a.out)
