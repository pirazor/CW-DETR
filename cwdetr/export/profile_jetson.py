"""Record reproducible TensorRT profiles on a Jetson target."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text(path: str):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _tensorrt_version():
    try:
        import tensorrt as trt
        return trt.__version__
    except Exception:
        return None


def _extract_latency_ms(log: str):
    patterns = {
        "p50_ms": r"(?:median|percentile\(50%\))\s*=\s*([0-9.]+)\s*ms",
        "p95_ms": r"percentile\(95%\)\s*=\s*([0-9.]+)\s*ms",
        "mean_ms": r"mean\s*=\s*([0-9.]+)\s*ms",
    }
    return {
        key: (float(match.group(1)) if (match := re.search(pattern, log, re.I)) else None)
        for key, pattern in patterns.items()
    }


def _profile_engine(name: str, engine: str, out_dir: Path, trtexec: str,
                    precision: str, warmup_ms: int, duration_ms: int, dry_run: bool):
    log_path = out_dir / f"{name}.trtexec.log"
    layer_path = out_dir / f"{name}.layers.json"
    times_path = out_dir / f"{name}.times.json"
    command = [
        trtexec,
        f"--loadEngine={engine}",
        f"--warmUp={warmup_ms}",
        f"--duration={duration_ms}",
        "--dumpProfile",
        f"--exportProfile={layer_path}",
        f"--exportTimes={times_path}",
        "--useSpinWait",
    ]
    if precision == "fp16":
        command.append("--fp16")
    elif precision == "int8":
        command.extend(["--int8", "--fp16"])
    result = {"name": name, "engine": engine, "sha256": sha256_file(engine),
              "command": command, "log": str(log_path), "layers": str(layer_path),
              "times": str(times_path), "precision": precision}
    if dry_run:
        result["returncode"] = None
        result["latency"] = {}
        return result
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    log = completed.stdout + "\n" + completed.stderr
    log_path.write_text(log, encoding="utf-8")
    result["returncode"] = completed.returncode
    result["latency"] = _extract_latency_ms(log)
    if completed.returncode:
        raise RuntimeError(f"trtexec profile failed for {name}; see {log_path}")
    return result


def profile(core_engine: str, out: str, precision: str = "fp16",
            sign_engine: str | None = None, trtexec: str = "trtexec",
            warmup_ms: int = 5000, duration_ms: int = 30000,
            calibration_hash: str | None = None, dry_run: bool = False):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jetpack_release": _read_text("/etc/nv_tegra_release"),
        "tensorrt": _tensorrt_version(),
        "precision": precision,
        "calibration_hash": calibration_hash,
        "grid_sample_note": "INT8 profiles use FP16 fallback for floating-point GridSample layers.",
        "profiles": [],
    }
    manifest["profiles"].append(
        _profile_engine("core", core_engine, out_path.parent, trtexec,
                        precision, warmup_ms, duration_ms, dry_run))
    if sign_engine:
        manifest["profiles"].append(
            _profile_engine("sign_sidecar", sign_engine, out_path.parent, trtexec,
                            "fp16", warmup_ms, duration_ms, dry_run))
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote Jetson TensorRT profile manifest -> {out_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-engine", required=True)
    parser.add_argument("--sign-engine", default=None)
    parser.add_argument("--precision", choices=["fp16", "int8", "fp32"], default="fp16")
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--warmup-ms", type=int, default=5000)
    parser.add_argument("--duration-ms", type=int, default=30000)
    parser.add_argument("--calibration-hash", default=None)
    parser.add_argument("--out", default="profiles/jetson_profile.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    profile(args.core_engine, args.out, args.precision, args.sign_engine, args.trtexec,
            args.warmup_ms, args.duration_ms, args.calibration_hash, args.dry_run)
