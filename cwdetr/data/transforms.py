"""Minimal image+target transforms.

Operates on a sample dict: image (PIL.Image) plus optional boxes (cxcywh,
normalized — so resizing the image leaves them unchanged), drivable/lane masks
(PIL or np), and sign boxes (xyxy pixels). Kept deliberately small; swap in
albumentations for production augmentation.
"""
from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch
import torchvision.transforms.functional as TF


class Resize:
    def __init__(self, hw):
        self.h, self.w = hw

    def __call__(self, s: Dict) -> Dict:
        ow, oh = s["image"].size
        s["image"] = TF.resize(s["image"], [self.h, self.w])
        for k in ("drivable", "lane"):
            if s.get(k) is not None:
                s[k] = TF.resize(s[k], [self.h, self.w],
                                 interpolation=TF.InterpolationMode.NEAREST)
        if s.get("sign_boxes") is not None and len(s["sign_boxes"]):
            sx, sy = self.w / ow, self.h / oh
            s["sign_boxes"] = s["sign_boxes"] * torch.tensor([sx, sy, sx, sy])
        return s


class RandomHFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, s: Dict) -> Dict:
        if s.get("disable_hflip") or random.random() > self.p:
            return s
        s["image"] = TF.hflip(s["image"])
        for k in ("drivable", "lane"):
            if s.get(k) is not None:
                s[k] = TF.hflip(s[k])
        if s.get("boxes") is not None and len(s["boxes"]):
            s["boxes"][:, 0] = 1.0 - s["boxes"][:, 0]          # flip cx (normalized)
        if s.get("sign_boxes") is not None and len(s["sign_boxes"]):
            w = s["image"].size[0]
            x1 = w - s["sign_boxes"][:, 2]
            x2 = w - s["sign_boxes"][:, 0]
            s["sign_boxes"][:, 0], s["sign_boxes"][:, 2] = x1, x2
        if s.get("future") is not None and s.get("trajectory_space") == "image":
            s["future"][..., 0] *= -1
        return s


class ToTensorNormalize:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, s: Dict) -> Dict:
        s["image"] = TF.normalize(TF.to_tensor(s["image"]), self.mean, self.std)
        for k in ("drivable", "lane"):
            if s.get(k) is not None:
                s[k] = torch.as_tensor(np.array(s[k]), dtype=torch.long)
        return s


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, s):
        for t in self.transforms:
            s = t(s)
        return s


def build_transforms(cfg, train: bool) -> Compose:
    ts = [Resize((cfg.input.height, cfg.input.width))]
    if train:
        ts.append(RandomHFlip(0.5))
    ts.append(ToTensorNormalize(cfg.input.normalize_mean, cfg.input.normalize_std))
    return Compose(ts)
