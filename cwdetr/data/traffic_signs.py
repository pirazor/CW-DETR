"""Traffic-sign sub-classification datasets (GTSRB / Mapillary Traffic Sign).

GTSRB ships pre-cropped signs (one folder per class). For those, the whole image
is the ROI. Mapillary Traffic Sign provides full street images with sign boxes +
fine labels; use ``MapillaryTrafficSign`` for in-context ROIs. Both yield the
CW-DETR sample contract, populating ``sign_boxes`` + ``sign_labels`` and a single
coarse ``traffic_sign`` detection box so the cascade is trained jointly.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import torch
from PIL import Image
from torch.utils.data import Dataset


class GTSRBSigns(Dataset):
    def __init__(self, root: str, split: str = "train", transforms=None):
        self.transforms = transforms
        self.samples = []
        base = os.path.join(root, "Train" if split == "train" else "Test")
        for cls in sorted(os.listdir(base)):
            cdir = os.path.join(base, cls)
            if not os.path.isdir(cdir):
                continue
            for fn in os.listdir(cdir):
                if fn.lower().endswith((".png", ".ppm", ".jpg")):
                    self.samples.append((os.path.join(cdir, fn), int(cls)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        path, cls = self.samples[idx]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        sample = {
            "image": img,
            "labels": torch.tensor([11], dtype=torch.long),       # coarse 'traffic_sign'
            "boxes": torch.tensor([[0.5, 0.5, 1.0, 1.0]], dtype=torch.float32),
            "drivable": None, "lane": None,
            "sign_boxes": torch.tensor([[0, 0, w, h]], dtype=torch.float32),
            "sign_labels": torch.tensor([cls], dtype=torch.long),
            "dataset": "gtsrb",
        }
        return self.transforms(sample) if self.transforms else sample


class MapillaryTrafficSign(Dataset):
    """Full-image signs with boxes + fine labels (annotation JSON per image)."""

    def __init__(self, root: str, split: str, class_index: Dict[str, int], transforms=None):
        self.root, self.split = root, split
        self.transforms = transforms
        self.class_index = class_index
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
            sboxes.append([x1, y1, x2, y2]); slabels.append(cls)
            dboxes.append([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h])
        sample = {
            "image": img,
            "labels": torch.full((len(dboxes),), 11, dtype=torch.long),
            "boxes": torch.as_tensor(dboxes, dtype=torch.float32).reshape(-1, 4),
            "drivable": None, "lane": None,
            "sign_boxes": torch.as_tensor(sboxes, dtype=torch.float32).reshape(-1, 4),
            "sign_labels": torch.as_tensor(slabels, dtype=torch.long),
            "dataset": "mapillary",
        }
        return self.transforms(sample) if self.transforms else sample
