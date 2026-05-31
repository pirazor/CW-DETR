"""BDD100K loader: 2D detection + drivable-area + lane-line segmentation.

Expects the standard BDD100K layout::

    root/
      images/100k/{train,val}/*.jpg
      labels/det_20/det_{train,val}.json          # detection (box2d)
      labels/drivable/masks/{train,val}/*.png      # drivable area (0 bg,1 direct,2 alt)
      labels/lane/masks/{train,val}/*.png          # lane lines (binary or typed)

Returns the CW-DETR sample contract (see data/multitask_dataset.py).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset

from cwdetr.data.taxonomy import BDD100K_MAP


class BDD100KDataset(Dataset):
    def __init__(self, root: str, split: str, transforms=None,
                 load_seg: bool = True, lane_binary: bool = True):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.load_seg = load_seg
        self.lane_binary = lane_binary
        self.img_dir = os.path.join(root, "images", "100k", split)
        det_json = os.path.join(root, "labels", "det_20", f"det_{split}.json")
        with open(det_json) as f:
            self.frames: List[Dict] = json.load(f)

    def __len__(self):
        return len(self.frames)

    def _load_mask(self, kind: str, name: str):
        path = os.path.join(self.root, "labels", kind, "masks", self.split,
                            name.replace(".jpg", ".png"))
        return Image.open(path) if os.path.exists(path) else None

    def __getitem__(self, idx: int) -> Dict:
        fr = self.frames[idx]
        name = fr["name"]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        w, h = img.size

        labels, boxes, sign_boxes = [], [], []
        for lab in fr.get("labels", []) or []:
            if "box2d" not in lab:
                continue
            cid = BDD100K_MAP.get(lab["category"])
            if cid is None:
                continue
            b = lab["box2d"]
            cx = (b["x1"] + b["x2"]) / 2 / w
            cy = (b["y1"] + b["y2"]) / 2 / h
            bw = (b["x2"] - b["x1"]) / w
            bh = (b["y2"] - b["y1"]) / h
            labels.append(cid)
            boxes.append([cx, cy, bw, bh])
            if cid == 11:  # traffic_sign -> also a sign ROI (label unknown w/o sub-class)
                sign_boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])

        sample = {
            "image": img,
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "drivable": self._load_mask("drivable", name) if self.load_seg else None,
            "lane": self._load_mask("lane", name) if self.load_seg else None,
            "sign_boxes": torch.as_tensor(sign_boxes, dtype=torch.float32).reshape(-1, 4),
            "sign_labels": torch.full((len(sign_boxes),), -1, dtype=torch.long),  # unlabeled sub-class
            "dataset": "bdd100k",
        }
        if self.transforms:
            sample = self.transforms(sample)
        if sample.get("lane") is not None and self.lane_binary:
            sample["lane"] = (sample["lane"] > 0).long()
        return sample
