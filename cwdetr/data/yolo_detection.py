"""YOLO-format object-detection dataset adapter.

The adapter intentionally consumes detection labels only. Image directories
follow the usual YOLO sibling layout::

    root/
      train/images/*.jpg
      train/labels/*.txt
      val/images/*.jpg
      val/labels/*.txt
      data.yaml

Each label row must be ``class_id cx cy width height`` with normalized box
coordinates. Missing label files represent valid background images. A small
manifest is cached beside ``data.yaml`` so remote filesystems such as a mounted
Google Drive only need one recursive image scan.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, TypeVar

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
TRANSIENT_IO_ERRNOS = {5, 16, 110, 116}
T = TypeVar("T")


def _retry_remote_io(action: Callable[[], T], description: str,
                     attempts: int = 5, initial_delay: float = 0.25) -> T:
    """Retry transient mounted-filesystem failures with a short backoff."""
    for attempt in range(attempts):
        try:
            return action()
        except OSError as exc:
            if exc.errno not in TRANSIENT_IO_ERRNOS or attempt + 1 == attempts:
                if exc.errno in TRANSIENT_IO_ERRNOS:
                    raise OSError(
                        exc.errno,
                        f"{description} failed after {attempts} attempts. "
                        "If this is a mounted Google Drive path, remount Drive and retry "
                        "the cell without rebuilding the dataset index.",
                        exc.filename) from exc
                raise
            time.sleep(initial_delay * (2 ** attempt))


def _ordered_names(raw_names, num_classes: int) -> List[str]:
    if isinstance(raw_names, list):
        names = [str(name) for name in raw_names]
    elif isinstance(raw_names, dict):
        try:
            normalized = {int(index): str(name) for index, name in raw_names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("YOLO data.yaml names keys must be integer class IDs") from exc
        expected = list(range(num_classes))
        if sorted(normalized) != expected:
            raise ValueError(f"YOLO data.yaml names must define contiguous IDs {expected}")
        names = [normalized[index] for index in expected]
    else:
        raise ValueError("YOLO data.yaml must define names as a list or class-ID mapping")
    if len(names) != num_classes:
        raise ValueError(
            f"YOLO data.yaml declares nc={num_classes} but defines {len(names)} names")
    return names


class YoloDetectionDataset(Dataset):
    """Read YOLO detection labels into CW-DETR's normalized target contract."""

    def __init__(self, data_yaml: str, split: str, transforms=None,
                 expected_num_classes: Optional[int] = None,
                 refresh_index: bool = False):
        self.data_yaml = Path(data_yaml).expanduser().resolve()
        self.split = split
        self.transforms = transforms
        spec = yaml.safe_load(_retry_remote_io(
            lambda: self.data_yaml.read_text(encoding="utf-8"),
            f"reading YOLO dataset config {self.data_yaml}")) or {}
        if split not in spec:
            raise ValueError(f"YOLO data.yaml does not define the {split!r} split")
        try:
            self.num_classes = int(spec["nc"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("YOLO data.yaml must define integer nc") from exc
        self.class_names = _ordered_names(spec.get("names"), self.num_classes)
        if expected_num_classes is not None and self.num_classes != expected_num_classes:
            raise ValueError(
                f"YOLO data.yaml declares nc={self.num_classes}, but the model detection "
                f"head expects {expected_num_classes} classes")

        root = Path(spec.get("path", self.data_yaml.parent)).expanduser()
        if not root.is_absolute():
            root = self.data_yaml.parent / root
        self.root = root.resolve()
        image_dir = Path(spec[split]).expanduser()
        if not image_dir.is_absolute():
            image_dir = self.root / image_dir
        self.image_dir = image_dir.resolve()
        if not _retry_remote_io(self.image_dir.is_dir,
                                f"checking YOLO image directory {self.image_dir}"):
            raise FileNotFoundError(f"YOLO image directory does not exist: {self.image_dir}")

        parts = list(self.image_dir.parts)
        image_indices = [index for index, part in enumerate(parts) if part == "images"]
        if not image_indices:
            raise ValueError(
                f"YOLO split path must contain an 'images' directory: {self.image_dir}")
        parts[image_indices[-1]] = "labels"
        self.label_dir = Path(*parts)
        self.index_path = self.data_yaml.parent / f".cwdetr-{self.data_yaml.stem}-{split}-images.txt"
        self.images = self._load_images(refresh_index)
        if not self.images:
            raise ValueError(f"YOLO split contains no supported images: {self.image_dir}")

    def _load_images(self, refresh_index: bool) -> List[Path]:
        index_exists = _retry_remote_io(
            self.index_path.exists, f"checking YOLO image manifest {self.index_path}")
        if index_exists and not refresh_index:
            images = []
            manifest = _retry_remote_io(
                lambda: self.index_path.read_text(encoding="utf-8"),
                f"reading YOLO image manifest {self.index_path}")
            for relative in manifest.splitlines():
                if not relative:
                    continue
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(f"invalid path in YOLO image manifest: {relative!r}")
                images.append(self.image_dir / relative_path)
            if images:
                return images

        self.images = sorted(
            path for path in self.image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if self.images:
            contents = "\n".join(
                path.relative_to(self.image_dir).as_posix() for path in self.images) + "\n"
            _retry_remote_io(
                lambda: self.index_path.write_text(contents, encoding="utf-8"),
                f"writing YOLO image manifest {self.index_path}")
        return self.images

    def __len__(self) -> int:
        return len(self.images)

    def _label_path(self, image_path: Path) -> Path:
        return (self.label_dir / image_path.relative_to(self.image_dir)).with_suffix(".txt")

    def _load_labels(self, image_path: Path):
        label_path = self._label_path(image_path)
        try:
            contents = _retry_remote_io(
                lambda: label_path.read_text(encoding="utf-8"),
                f"reading YOLO label file {label_path}")
        except FileNotFoundError:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, 4)
        labels, boxes = [], []
        for line_number, line in enumerate(contents.splitlines(), start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(
                    f"{label_path}:{line_number}: expected YOLO detection row "
                    "'class_id cx cy width height'")
            try:
                raw_class, *raw_box = [float(field) for field in fields]
            except ValueError as exc:
                raise ValueError(f"{label_path}:{line_number}: label values must be numeric") from exc
            class_id = int(raw_class)
            if raw_class != class_id or not 0 <= class_id < self.num_classes:
                raise ValueError(
                    f"{label_path}:{line_number}: class ID {raw_class} is outside "
                    f"[0, {self.num_classes - 1}]")
            if not all(math.isfinite(value) for value in raw_box):
                raise ValueError(f"{label_path}:{line_number}: box values must be finite")
            cx, cy, width, height = raw_box
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError(
                    f"{label_path}:{line_number}: normalized box values are outside YOLO bounds")
            labels.append(class_id)
            boxes.append(raw_box)
        return (torch.as_tensor(labels, dtype=torch.long),
                torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4))

    def __getitem__(self, index: int) -> Dict:
        image_path = self.images[index]
        def load_image():
            with Image.open(image_path) as opened:
                opened.load()
                return opened.convert("RGB")

        image = _retry_remote_io(load_image, f"reading YOLO image file {image_path}")
        width, height = image.size
        labels, boxes = self._load_labels(image_path)
        relative_path = image_path.relative_to(self.image_dir).as_posix()
        sample = {
            "image": image,
            "labels": labels,
            "boxes": boxes,
            "train_detection": True,
            "drivable": None,
            "lane": None,
            "sign_boxes": torch.zeros(0, 4),
            "sign_labels": torch.zeros(0, dtype=torch.long),
            "image_id": f"{self.split}/{relative_path}",
            "orig_size": (height, width),
            "dataset": "yolo_detection",
        }
        return self.transforms(sample) if self.transforms else sample
