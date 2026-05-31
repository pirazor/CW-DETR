"""nuScenes loader for tracking + trajectory supervision (front-camera 2D).

Scaffold around nuscenes-devkit. Produces frame samples that additionally carry
per-object track ids and BEV/image future waypoints, used to train the track
head (identity) and trajectory head (multimodal futures). Heavy devkit calls are
isolated here so the rest of the pipeline stays light.

Contract additions on top of the base sample:
    track_ids   : LongTensor[n]        persistent instance id per GT box
    future      : Tensor[n, T, 2]      relative positions (normalized image or BEV metres)
    future_mask : Tensor[n, T]         1 where a future step is observed
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cwdetr.data.taxonomy import NUSCENES_MAP


class NuScenesSequences(Dataset):
    def __init__(self, root: str, version: str = "v1.0-trainval", split: str = "train",
                 camera: str = "CAM_FRONT", future_len: int = 12, step_dt: float = 0.5,
                 space: str = "bev", transforms=None):
        self.root, self.version, self.split = root, version, split
        self.camera, self.future_len, self.step_dt = camera, future_len, step_dt
        self.space, self.transforms = space, transforms
        if abs(step_dt - 0.5) > 1e-6:
            raise ValueError("nuScenes keyframes are sampled at 0.5 second intervals")
        if space not in ("image", "bev"):
            raise ValueError("trajectory space must be 'image' or 'bev'")
        self._nusc = None
        self._instance_ids: Dict[str, int] = {}
        self.samples: List[str] = []   # sample tokens; filled by _lazy_init
        self._lazy_init()

    def _lazy_init(self):
        try:
            from nuscenes.nuscenes import NuScenes
            from nuscenes.utils.splits import create_splits_scenes
            self._nusc = NuScenes(version=self.version, dataroot=self.root, verbose=False)
            self._instance_ids = {
                instance["token"]: idx for idx, instance in enumerate(self._nusc.instance)
            }
            split_scenes = set(create_splits_scenes()[self.split])
            self.samples = [
                s["token"] for s in self._nusc.sample
                if self._nusc.get("scene", s["scene_token"])["name"] in split_scenes
            ]
        except Exception as exc:  # devkit/data absent -> empty dataset, non-fatal
            print(f"[nuscenes] devkit/data unavailable ({exc}); dataset is empty.")
            self.samples = []

    def __len__(self):
        return len(self.samples)

    def _instance_id(self, token: str) -> int:
        return self._instance_ids[token]

    def _project_center(self, ann: Dict) -> Optional[torch.Tensor]:
        """Project an annotation center into its frame's configured camera."""
        from nuscenes.utils.geometry_utils import view_points
        from pyquaternion import Quaternion

        nusc = self._nusc
        sample = nusc.get("sample", ann["sample_token"])
        sd = nusc.get("sample_data", sample["data"][self.camera])
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        sensor = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        point = np.asarray(ann["translation"], dtype=np.float64)
        point = Quaternion(pose["rotation"]).inverse.rotate(point - np.asarray(pose["translation"]))
        point = Quaternion(sensor["rotation"]).inverse.rotate(
            point - np.asarray(sensor["translation"]))
        if point[2] <= 1e-4:
            return None
        xy = view_points(point[:, None], np.asarray(sensor["camera_intrinsic"]),
                         normalize=True)[:2, 0]
        return torch.as_tensor(xy, dtype=torch.float32)

    def _gather_future(self, ann: Dict, ego_rotation, current_center,
                       image_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Walk future annotations and return relative image or ego-frame waypoints."""
        from pyquaternion import Quaternion

        waypoints = torch.zeros(self.future_len, 2, dtype=torch.float32)
        mask = torch.zeros(self.future_len, dtype=torch.float32)
        token = ann["next"]
        for step in range(self.future_len):
            if not token:
                break
            rec = self._nusc.get("sample_annotation", token)
            if self.space == "image":
                xy = self._project_center(rec)
                if xy is not None:
                    scale = torch.tensor([image_hw[1], image_hw[0]], dtype=torch.float32)
                    waypoints[step] = (xy - current_center) / scale
                    mask[step] = 1.0
            else:
                delta = np.asarray(rec["translation"][:2]) - np.asarray(ann["translation"][:2])
                rotation = Quaternion(ego_rotation).rotation_matrix[:2, :2]
                waypoints[step] = torch.as_tensor(delta @ rotation, dtype=torch.float32)
                mask[step] = 1.0
            token = rec["next"]
        return waypoints, mask

    def __getitem__(self, idx: int) -> Dict:
        from nuscenes.utils.geometry_utils import BoxVisibility, view_points

        sample = self._nusc.get("sample", self.samples[idx])
        cam_token = sample["data"][self.camera]
        sd = self._nusc.get("sample_data", cam_token)
        pose = self._nusc.get("ego_pose", sd["ego_pose_token"])
        image_path, visible_boxes, intrinsic = self._nusc.get_sample_data(
            cam_token, box_vis_level=BoxVisibility.ANY)
        image = Image.open(image_path if os.path.isabs(image_path)
                           else os.path.join(self.root, image_path)).convert("RGB")
        w, h = image.size

        labels, boxes, track_ids, futures, future_masks = [], [], [], [], []
        for box in visible_boxes:
            cid = NUSCENES_MAP.get(box.name)
            if cid is None or box.token is None:
                continue
            corners = view_points(box.corners(), np.asarray(intrinsic), normalize=True)[:2]
            x1, y1 = np.maximum(corners.min(1), 0)
            x2, y2 = np.minimum(corners.max(1), [w, h])
            if x2 <= x1 or y2 <= y1:
                continue
            ann = self._nusc.get("sample_annotation", box.token)
            labels.append(cid)
            boxes.append([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                          (x2 - x1) / w, (y2 - y1) / h])
            track_ids.append(self._instance_id(ann["instance_token"]))
            future, future_mask = self._gather_future(
                ann, pose["rotation"],
                torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2], dtype=torch.float32),
                (h, w))
            futures.append(future)
            future_masks.append(future_mask)

        sample_out = {
            "image": image,
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "drivable": None,
            "lane": None,
            "sign_boxes": torch.zeros(0, 4),
            "sign_labels": torch.zeros(0, dtype=torch.long),
            "track_ids": torch.as_tensor(track_ids, dtype=torch.long),
            "future": (torch.stack(futures) if futures
                       else torch.zeros(0, self.future_len, 2)),
            "future_mask": (torch.stack(future_masks) if future_masks
                            else torch.zeros(0, self.future_len)),
            "trajectory_space": self.space,
            "disable_hflip": self.space == "bev",
            "dataset": "nuscenes",
        }
        return self.transforms(sample_out) if self.transforms else sample_out
