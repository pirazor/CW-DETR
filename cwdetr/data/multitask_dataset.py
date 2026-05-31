"""Mixed multi-dataset training utilities.

Different datasets carry different labels (BDD100K: det+seg; nuScenes: det+track+
future; sign sets: det+fine-sign). We draw **dataset-homogeneous batches** (every
sample in a batch comes from one dataset) so each batch has a consistent label
set; the criterion simply skips tasks whose labels are absent, and the
uncertainty weighter keeps the active tasks balanced across steps.
"""
from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional

import torch
from torch.utils.data import Dataset, Sampler


class ConcatMultiTaskDataset(Dataset):
    def __init__(self, datasets: List[Dataset],
                 sign_label_maps: Optional[Mapping[str, Mapping[int, int]]] = None):
        self.datasets = datasets
        self.sign_label_maps = dict(sign_label_maps or {})
        sign_taxonomies = {
            dataset.sign_taxonomy for dataset in datasets
            if getattr(dataset, "sign_taxonomy", None) is not None
        }
        missing_maps = sign_taxonomies - set(self.sign_label_maps)
        if len(sign_taxonomies) > 1 and missing_maps:
            raise ValueError(
                "incompatible sign taxonomies require explicit sign_label_maps; "
                f"missing mappings for {sorted(missing_maps)}")
        self.lengths = [len(d) for d in datasets]
        self.offsets, c = [], 0
        for n in self.lengths:
            self.offsets.append(c)
            c += n
        self.total = c

    def __len__(self):
        return self.total

    def locate(self, gidx):
        for di in range(len(self.datasets) - 1, -1, -1):
            if gidx >= self.offsets[di]:
                return di, gidx - self.offsets[di]
        return 0, gidx

    def __getitem__(self, gidx):
        di, li = self.locate(gidx)
        dataset = self.datasets[di]
        sample = dataset[li]
        taxonomy = getattr(dataset, "sign_taxonomy", None)
        if taxonomy in self.sign_label_maps and sample.get("sign_labels") is not None:
            mapping = self.sign_label_maps[taxonomy]
            sample = dict(sample)
            sample["sign_labels"] = torch.as_tensor(
                [mapping.get(int(label), -1) for label in sample["sign_labels"]],
                dtype=torch.long)
        return sample


class MixedBatchSampler(Sampler):
    def __init__(self, concat: ConcatMultiTaskDataset, batch_size: int,
                 weights: List[float] = None, drop_last: bool = True):
        self.concat = concat
        self.bs = batch_size
        self.drop_last = drop_last
        n = len(concat.datasets)
        self.weights = weights or [1.0] * n

    def __iter__(self):
        # build per-dataset shuffled batch queues
        queues = []
        for di, length in enumerate(self.concat.lengths):
            idx = [self.concat.offsets[di] + i for i in range(length)]
            random.shuffle(idx)
            batches = [idx[i:i + self.bs] for i in range(0, length, self.bs)]
            if self.drop_last and batches and len(batches[-1]) < self.bs:
                batches.pop()
            queues.append(batches)
        total = sum(len(q) for q in queues)
        # interleave datasets by weight
        while total > 0:
            di = random.choices(range(len(queues)),
                                weights=[self.weights[i] * len(queues[i]) for i in range(len(queues))]
                                )[0] if any(queues) else None
            if di is None or not queues[di]:
                if not any(queues):
                    break
                continue
            yield queues[di].pop()
            total -= 1

    def __len__(self):
        return sum(max(0, length // self.bs) for length in self.concat.lengths)


def collate_fn(batch: List[Dict]) -> Dict:
    images = torch.stack([s["image"] for s in batch])
    detection = [{"labels": s["labels"], "boxes": s["boxes"],
                  "train_detection": s.get("train_detection", True)} for s in batch]

    has_drivable = all(s.get("drivable") is not None for s in batch)
    has_lane = all(s.get("lane") is not None for s in batch)
    drivable = torch.stack([s["drivable"] for s in batch]) if has_drivable else None
    lane = torch.stack([s["lane"] for s in batch]) if has_lane else None

    # labelled sign ROIs (image pixel coords) -> [K,5] + labels [K]
    sign_rois, sign_labels = [], []
    for bi, s in enumerate(batch):
        sb, sl = s.get("sign_boxes"), s.get("sign_labels")
        if sb is None or sl is None:
            continue
        keep = sl >= 0
        if keep.any():
            kb = sb[keep]
            bidx = torch.full((kb.shape[0], 1), float(bi))
            sign_rois.append(torch.cat([bidx, kb], 1))
            sign_labels.append(sl[keep])
    sign_rois = torch.cat(sign_rois, 0) if sign_rois else torch.zeros(0, 5)
    sign_labels = torch.cat(sign_labels, 0) if sign_labels else torch.zeros(0, dtype=torch.long)

    targets = {
        "detection": detection,
        "drivable": drivable,
        "lane": lane,
        "sign_labels": sign_labels,
    }
    if all("future" in s for s in batch):
        targets["future"] = torch.cat([s["future"] for s in batch], 0)
        targets["future_mask"] = torch.cat([s["future_mask"] for s in batch], 0)
    if all("track_ids" in s for s in batch):
        targets["track_ids"] = [s["track_ids"] for s in batch]

    extras = {
        "sign_rois": sign_rois,
        "dataset": batch[0].get("dataset", "?"),
        "image_ids": [s.get("image_id") for s in batch],
        "orig_sizes": [s.get("orig_size") for s in batch],
    }
    return {"images": images, "targets": targets, "extras": extras}
