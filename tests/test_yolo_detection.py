from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import numpy as np
import torch
from PIL import Image

from cwdetr.config import load_config
from cwdetr.data.yolo_detection import YoloDetectionDataset
from cwdetr.engine.evaluate import build_eval_dataset
from cwdetr.engine.train import build_datasets


NAMES = [
    "car", "truck", "bus", "train", "bike", "cyclist", "person",
    "traffic_light", "traffic_sign",
]


def _write_yaml(root: Path, names=NAMES) -> Path:
    yaml_path = root / "data.yaml"
    names_yaml = "\n".join(f"  {index}: {name}" for index, name in enumerate(names))
    yaml_path.write_text(
        "path: .\n"
        "train: train/images/\n"
        "val: val/images/\n"
        f"nc: {len(names)}\n"
        "names:\n"
        f"{names_yaml}\n",
        encoding="utf-8")
    return yaml_path


def _write_image(root: Path, split: str, name: str) -> Path:
    path = root / split / "images" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 10)).save(path)
    return path


def test_yolo_detection_dataset_reads_labels_and_background_images(tmp_path):
    yaml_path = _write_yaml(tmp_path)
    image = _write_image(tmp_path, "train", "nested/labelled.jpg")
    background = _write_image(tmp_path, "train", "background.jpg")
    label = tmp_path / "train" / "labels" / "nested" / "labelled.txt"
    label.parent.mkdir(parents=True)
    label.write_text("8 0.5 0.4 0.2 0.3\n0 0.1 0.2 0.05 0.1\n", encoding="utf-8")

    dataset = YoloDetectionDataset(str(yaml_path), "train", expected_num_classes=9)
    samples = {sample["image_id"]: sample for sample in dataset}
    labelled = samples["train/nested/labelled.jpg"]
    empty = samples["train/background.jpg"]

    assert image.exists() and background.exists()
    assert dataset.class_names == NAMES
    assert labelled["labels"].tolist() == [8, 0]
    assert torch.allclose(labelled["boxes"], torch.tensor([
        [0.5, 0.4, 0.2, 0.3], [0.1, 0.2, 0.05, 0.1]]))
    assert labelled["orig_size"] == (10, 20)
    assert labelled["train_detection"]
    assert empty["labels"].numel() == 0
    assert empty["boxes"].shape == (0, 4)


def test_yolo_detection_dataset_rejects_model_class_mismatch(tmp_path):
    yaml_path = _write_yaml(tmp_path)
    _write_image(tmp_path, "train", "sample.jpg")
    with pytest.raises(ValueError, match="model detection head expects 8"):
        YoloDetectionDataset(str(yaml_path), "train", expected_num_classes=8)


def test_yolo_detection_dataset_caches_and_refreshes_image_manifest(tmp_path):
    yaml_path = _write_yaml(tmp_path)
    _write_image(tmp_path, "train", "first.jpg")
    dataset = YoloDetectionDataset(str(yaml_path), "train", expected_num_classes=9)
    assert dataset.index_path.exists()
    assert len(dataset) == 1

    _write_image(tmp_path, "train", "second.jpg")
    cached = YoloDetectionDataset(str(yaml_path), "train", expected_num_classes=9)
    refreshed = YoloDetectionDataset(
        str(yaml_path), "train", expected_num_classes=9, refresh_index=True)
    assert len(cached) == 1
    assert len(refreshed) == 2


def test_yolo_detection_dataset_rejects_non_detection_rows(tmp_path):
    yaml_path = _write_yaml(tmp_path)
    _write_image(tmp_path, "train", "sample.jpg")
    label = tmp_path / "train" / "labels" / "sample.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2 0.7 0.7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected YOLO detection row"):
        YoloDetectionDataset(str(yaml_path), "train", expected_num_classes=9)


def test_yolo_detection_dataset_retries_transient_label_reads(tmp_path):
    yaml_path = _write_yaml(tmp_path)
    _write_image(tmp_path, "train", "sample.jpg")
    label = tmp_path / "train" / "labels" / "sample.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    original_read_text = Path.read_text
    failures = 0

    def flaky_read_text(path, *args, **kwargs):
        nonlocal failures
        if path == label and failures < 2:
            failures += 1
            raise OSError(5, "simulated mounted-drive failure", str(path))
        return original_read_text(path, *args, **kwargs)

    with mock.patch.object(Path, "read_text", flaky_read_text), \
            mock.patch("cwdetr.data.yolo_detection.time.sleep"):
        dataset = YoloDetectionDataset(str(yaml_path), "train", expected_num_classes=9)
    assert failures == 2
    assert dataset[0]["labels"].tolist() == [0]


def test_yolo_detection_dataset_uses_parsed_labels_during_getitem(tmp_path):
    yaml_path = _write_yaml(tmp_path)
    _write_image(tmp_path, "train", "sample.jpg")
    label = tmp_path / "train" / "labels" / "sample.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    dataset = YoloDetectionDataset(str(yaml_path), "train", expected_num_classes=9)
    original_read_text = Path.read_text

    def reject_label_read(path, *args, **kwargs):
        if path == label:
            raise AssertionError("getitem must not reopen cached YOLO label files")
        return original_read_text(path, *args, **kwargs)

    with mock.patch.object(Path, "read_text", reject_label_read):
        assert dataset[0]["labels"].tolist() == [0]


def test_yolo_detection_dataset_can_cache_images_as_float32_ram(tmp_path):
    yaml_path = _write_yaml(tmp_path)
    image_path = _write_image(tmp_path, "train", "sample.jpg")
    label = tmp_path / "train" / "labels" / "sample.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    dataset = YoloDetectionDataset(
        str(yaml_path), "train", expected_num_classes=9,
        image_cache="ram-float32")
    assert dataset.image_cache is not None
    assert dataset.image_cache[0].dtype == np.float32

    original_open = Image.open

    def reject_image_open(path, *args, **kwargs):
        if Path(path) == image_path:
            raise AssertionError("getitem must not reopen cached YOLO image files")
        return original_open(path, *args, **kwargs)

    with mock.patch.object(Image, "open", reject_image_open):
        sample = dataset[0]
    assert sample["orig_size"] == (10, 20)
    assert sample["labels"].tolist() == [0]


def test_yolo_detection_dataset_imports_existing_ultralytics_label_cache(tmp_path):
    yaml_path = _write_yaml(tmp_path)
    image = _write_image(tmp_path, "train", "sample.jpg")
    label_dir = tmp_path / "train" / "labels"
    label_dir.mkdir(parents=True)
    ultralytics_cache = label_dir.with_suffix(".cache")
    payload = {
        "labels": [{
            "im_file": str(image),
            "cls": np.asarray([[8.0]], dtype=np.float32),
            "bboxes": np.asarray([[0.5, 0.4, 0.2, 0.3]], dtype=np.float32),
        }],
    }
    with ultralytics_cache.open("wb") as handle:
        np.save(handle, payload)

    dataset = YoloDetectionDataset(str(yaml_path), "train", expected_num_classes=9)
    sample = dataset[0]
    assert sample["labels"].tolist() == [8]
    assert torch.allclose(sample["boxes"], torch.tensor([[0.5, 0.4, 0.2, 0.3]]))
    assert dataset.label_cache_path.exists()


def test_yolo_detection_only_config_and_eval_builder(tmp_path):
    yaml_path = _write_yaml(tmp_path)
    _write_image(tmp_path, "train", "sample.jpg")
    _write_image(tmp_path, "val", "sample.jpg")
    cfg = load_config("configs/cwdetr_nano_yolo_bdd_detection.yaml")
    assert cfg.model.heads.detection.num_classes == 9
    assert not cfg.model.heads.segmentation.enabled
    assert not cfg.model.heads.sign_classification.enabled
    dataset = build_eval_dataset(cfg, yolo_data=str(yaml_path))
    assert len(dataset) == 1
    assert dataset[0]["dataset"] == "yolo_detection"

    args = SimpleNamespace(
        bdd_root=None, yolo_data=str(yaml_path), gtsrb_root=None,
        nuscenes_root=None, nuscenes_version=None, nuscenes_split=None)
    train_dataset, weights = build_datasets(cfg, args)
    assert len(train_dataset) == 1
    assert weights == [2.0]
