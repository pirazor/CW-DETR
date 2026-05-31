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
from torchvision.transforms import ColorJitter


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


def _filter_aligned(s: Dict, keep: torch.Tensor, keys) -> None:
    for key in keys:
        value = s.get(key)
        if torch.is_tensor(value) and value.shape[0] == keep.shape[0]:
            s[key] = value[keep]


class RandomScaleCrop:
    """Large-scale jitter with crop/pad to a fixed output size."""

    def __init__(self, hw, scale_min=0.8, scale_max=1.2, crop_prob=0.5):
        self.h, self.w = hw
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.crop_prob = crop_prob
        self.fallback = Resize(hw)

    def __call__(self, s: Dict) -> Dict:
        # Temporal targets need clip-consistent geometry. Keep them deterministic
        # until the clip sampler owns shared augmentation parameters.
        if s.get("future") is not None:
            return self.fallback(s)

        ow, oh = s["image"].size
        scale = random.uniform(self.scale_min, self.scale_max)
        nh, nw = max(1, round(self.h * scale)), max(1, round(self.w * scale))
        sx, sy = nw / ow, nh / oh
        s["image"] = TF.resize(s["image"], [nh, nw])
        for key in ("drivable", "lane"):
            if s.get(key) is not None:
                s[key] = TF.resize(s[key], [nh, nw],
                                   interpolation=TF.InterpolationMode.NEAREST)

        random_offset = random.random() < self.crop_prob
        pad_x = max(0, self.w - nw)
        pad_y = max(0, self.h - nh)
        left = random.randint(0, pad_x) if random_offset and pad_x else pad_x // 2
        top = random.randint(0, pad_y) if random_offset and pad_y else pad_y // 2
        right, bottom = pad_x - left, pad_y - top
        if pad_x or pad_y:
            s["image"] = TF.pad(s["image"], [left, top, right, bottom], fill=0)
            for key in ("drivable", "lane"):
                if s.get(key) is not None:
                    s[key] = TF.pad(s[key], [left, top, right, bottom], fill=0)

        padded_w, padded_h = nw + pad_x, nh + pad_y
        max_crop_x, max_crop_y = padded_w - self.w, padded_h - self.h
        crop_x = random.randint(0, max_crop_x) if random_offset and max_crop_x else max_crop_x // 2
        crop_y = random.randint(0, max_crop_y) if random_offset and max_crop_y else max_crop_y // 2
        if max_crop_x or max_crop_y:
            s["image"] = TF.crop(s["image"], crop_y, crop_x, self.h, self.w)
            for key in ("drivable", "lane"):
                if s.get(key) is not None:
                    s[key] = TF.crop(s[key], crop_y, crop_x, self.h, self.w)

        if s.get("boxes") is not None and len(s["boxes"]):
            boxes = s["boxes"]
            cx, cy, bw, bh = boxes.unbind(-1)
            xyxy = torch.stack([(cx - bw / 2) * ow, (cy - bh / 2) * oh,
                                (cx + bw / 2) * ow, (cy + bh / 2) * oh], -1)
            xyxy = xyxy * xyxy.new_tensor([sx, sy, sx, sy])
            xyxy = xyxy + xyxy.new_tensor([left - crop_x, top - crop_y,
                                            left - crop_x, top - crop_y])
            xyxy[:, 0::2].clamp_(0, self.w)
            xyxy[:, 1::2].clamp_(0, self.h)
            keep = (xyxy[:, 2] - xyxy[:, 0] > 1) & (xyxy[:, 3] - xyxy[:, 1] > 1)
            xyxy = xyxy[keep]
            s["boxes"] = torch.stack([
                (xyxy[:, 0] + xyxy[:, 2]) / 2 / self.w,
                (xyxy[:, 1] + xyxy[:, 3]) / 2 / self.h,
                (xyxy[:, 2] - xyxy[:, 0]) / self.w,
                (xyxy[:, 3] - xyxy[:, 1]) / self.h,
            ], -1)
            _filter_aligned(s, keep, ("labels", "track_ids", "future", "future_mask"))

        if s.get("sign_boxes") is not None and len(s["sign_boxes"]):
            sign_boxes = s["sign_boxes"] * s["sign_boxes"].new_tensor([sx, sy, sx, sy])
            sign_boxes = sign_boxes + sign_boxes.new_tensor(
                [left - crop_x, top - crop_y, left - crop_x, top - crop_y])
            sign_boxes[:, 0::2].clamp_(0, self.w)
            sign_boxes[:, 1::2].clamp_(0, self.h)
            keep = ((sign_boxes[:, 2] - sign_boxes[:, 0] > 1)
                    & (sign_boxes[:, 3] - sign_boxes[:, 1] > 1))
            s["sign_boxes"] = sign_boxes[keep]
            _filter_aligned(s, keep, ("sign_labels",))
        return s


class PhotometricJitter:
    def __init__(self, p=0.8, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
        self.p = p
        self.jitter = ColorJitter(brightness=brightness, contrast=contrast,
                                  saturation=saturation, hue=hue)

    def __call__(self, s: Dict) -> Dict:
        if random.random() <= self.p:
            s["image"] = self.jitter(s["image"])
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
    hw = (cfg.input.height, cfg.input.width)
    aug = cfg.augmentation
    if train and aug.enabled:
        ts = [RandomScaleCrop(hw, aug.scale_min, aug.scale_max, aug.crop_prob),
              PhotometricJitter(aug.photometric_prob, aug.brightness, aug.contrast,
                                aug.saturation, aug.hue)]
    else:
        ts = [Resize(hw)]
    if train:
        ts.append(RandomHFlip(aug.hflip_prob))
    ts.append(ToTensorNormalize(cfg.input.normalize_mean, cfg.input.normalize_std))
    return Compose(ts)
