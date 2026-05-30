"""nuScenes loader for tracking + trajectory supervision (front-camera 2D).

Scaffold around nuscenes-devkit. Produces frame samples that additionally carry
per-object track ids and BEV/image future waypoints, used to train the track
head (identity) and trajectory head (multimodal futures). Heavy devkit calls are
isolated here so the rest of the pipeline stays light.

Contract additions on top of the base sample:
    track_ids   : LongTensor[n]        persistent instance id per GT box
    future      : Tensor[n, T, 2]      future positions (image px or BEV metres)
    future_mask : Tensor[n, T]         1 where a future step is observed
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from cwdetr.data.taxonomy import NUSCENES_MAP


class NuScenesSequences(Dataset):
    def __init__(self, root: str, version: str = "v1.0-trainval", split: str = "train",
                 camera: str = "CAM_FRONT", future_len: int = 12, step_dt: float = 0.5,
                 space: str = "bev", transforms=None):
        self.root, self.version, self.split = root, version, split
        self.camera, self.future_len, self.step_dt = camera, future_len, step_dt
        self.space, self.transforms = space, transforms
        self._nusc = None
        self.samples: List[str] = []   # sample tokens; filled by _lazy_init
        self._lazy_init()

    def _lazy_init(self):
        try:
            from nuscenes.nuscenes import NuScenes
            self._nusc = NuScenes(version=self.version, dataroot=self.root, verbose=False)
            self.samples = [s["token"] for s in self._nusc.sample]
        except Exception as exc:  # devkit/data absent -> empty dataset, non-fatal
            print(f"[nuscenes] devkit/data unavailable ({exc}); dataset is empty.")
            self.samples = []

    def __len__(self):
        return len(self.samples)

    def _gather_future(self, instance_token, sample) -> Optional[torch.Tensor]:
        """Walk the instance forward ``future_len`` annotated steps -> waypoints."""
        nusc = self._nusc
        waypoints = []
        ann_tokens = [a for a in sample["anns"]]
        cur = next((a for a in ann_tokens
                    if nusc.get("sample_annotation", a)["instance_token"] == instance_token), None)
        steps = 0
        while cur is not None and steps < self.future_len:
            rec = nusc.get("sample_annotation", cur)
            waypoints.append(rec["translation"][:2])     # global x,y (BEV)
            cur = rec["next"] if rec["next"] else None
            steps += 1
        if not waypoints:
            return None
        wp = torch.tensor(waypoints, dtype=torch.float32)
        if wp.shape[0] < self.future_len:                 # pad
            pad = wp[-1:].repeat(self.future_len - wp.shape[0], 1)
            wp = torch.cat([wp, pad], 0)
        return wp

    def __getitem__(self, idx: int) -> Dict:
        # Full per-sample assembly (image load, box projection, future walk) lives
        # here; omitted lines are mechanical nuscenes-devkit calls. The returned
        # dict must follow the contract documented in this module's docstring.
        raise NotImplementedError(
            "Fill in nuScenes per-sample assembly with the devkit; the target "
            "contract (track_ids/future/future_mask) is documented above.")
