"""Traffic-sign sub-classification datasets (GTSRB / Mapillary Traffic Sign).

GTSRB ships pre-cropped signs (one folder per class). For those, the whole image
is the ROI. Mapillary Traffic Sign provides full street images with sign boxes +
fine labels; use ``MapillaryTrafficSign`` for in-context ROIs. Both yield the
  CW-DETR sample contract, populating ``sign_boxes`` + ``sign_labels``. GTSRB
  crops intentionally do not supervise the coarse detector because they are not
  in-context road images.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict

import torch
from PIL import Image
from torch.utils.data import Dataset


class GTSRBSigns(Dataset):
    sign_taxonomy = "gtsrb43"

    def __init__(self, root: str, split: str = "train", transforms=None):
        self.transforms = transforms
        self.samples = []
        base = os.path.join(root, "Train" if split == "train" else "Test")
        if os.path.isdir(base):
            for cls in sorted(os.listdir(base)):
                cdir = os.path.join(base, cls)
                if not os.path.isdir(cdir) or not cls.isdigit():
                    continue
                for fn in os.listdir(cdir):
                    if fn.lower().endswith((".png", ".ppm", ".jpg")):
                        self.samples.append((os.path.join(cdir, fn), int(cls)))
        if not self.samples and split != "train":
            self._load_flat_test_csv(root, base)
        if not self.samples:
            raise FileNotFoundError(f"no GTSRB {split} samples found under {root}")

    def _load_flat_test_csv(self, root: str, base: str):
        csv_paths = [os.path.join(root, name) for name in ("Test.csv", "GT-final_test.csv")]
        csv_paths += [os.path.join(base, name) for name in ("Test.csv", "GT-final_test.csv")]
        csv_path = next((path for path in csv_paths if os.path.isfile(path)), None)
        if csv_path is None:
            return
        with open(csv_path, newline="", encoding="utf-8-sig") as handle:
            first_line = handle.readline()
            handle.seek(0)
            reader = csv.DictReader(handle, delimiter=";" if ";" in first_line else ",")
            for row in reader:
                relative = row.get("Path") or row.get("Filename")
                class_id = row.get("ClassId")
                if relative is None or class_id is None:
                    continue
                candidates = [os.path.join(root, relative), os.path.join(base, relative),
                              os.path.join(base, os.path.basename(relative))]
                image_path = next((path for path in candidates if os.path.isfile(path)), None)
                if image_path is not None:
                    self.samples.append((image_path, int(class_id)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        path, cls = self.samples[idx]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        sample = {
            "image": img,
            "labels": torch.zeros(0, dtype=torch.long),
            "boxes": torch.zeros(0, 4, dtype=torch.float32),
            "train_detection": False,
            "drivable": None, "lane": None,
            "sign_boxes": torch.tensor([[0, 0, w, h]], dtype=torch.float32),
            "sign_labels": torch.tensor([cls], dtype=torch.long),
            "image_id": path,
            "orig_size": (h, w),
            "dataset": "gtsrb",
        }
        return self.transforms(sample) if self.transforms else sample


class MapillaryTrafficSign(Dataset):
    """Full-image signs with boxes + fine labels (annotation JSON per image)."""

    def __init__(self, root: str, split: str, class_index: Dict[str, int], transforms=None,
                 taxonomy_name: str = "mapillary"):
        self.root, self.split = root, split
        self.transforms = transforms
        self.class_index = class_index
        self.sign_taxonomy = taxonomy_name
        with open(os.path.join(root, "splits", f"{split}.txt")) as f:
            self.ids = [ln.strip() for ln in f if ln.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict:
        img_id = self.ids[idx]
        img = Image.open(os.path.join(self.root, "images", f"{img_id}.jpg")).convert("RGB")
        with open(os.path.join(self.root, "annotations", f"{img_id}.json")) as f:
            ann = json.load(f)
        sboxes, slabels, dboxes, w, h = [], [], [], *img.size
        for obj in ann.get("objects", []):
            bb = obj["bbox"]
            x1, y1, x2, y2 = bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]
            cls = self.class_index.get(obj["label"])
            if cls is None:
                continue
            sboxes.append([x1, y1, x2, y2])
            slabels.append(cls)
            dboxes.append([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h])
        sample = {
            "image": img,
            "labels": torch.full((len(dboxes),), 11, dtype=torch.long),
            "boxes": torch.as_tensor(dboxes, dtype=torch.float32).reshape(-1, 4),
            "train_detection": True,
            "drivable": None, "lane": None,
            "sign_boxes": torch.as_tensor(sboxes, dtype=torch.float32).reshape(-1, 4),
            "sign_labels": torch.as_tensor(slabels, dtype=torch.long),
            "image_id": img_id,
            "orig_size": (h, w),
            "dataset": "mapillary",
        }
        return self.transforms(sample) if self.transforms else sample
