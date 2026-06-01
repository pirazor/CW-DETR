"""Validation loop and metrics for the training-ready CW-DETR baseline."""
from __future__ import annotations

import argparse
import contextlib
import io
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cwdetr.config import load_config
from cwdetr.data import (BDD100KDataset, ConcatMultiTaskDataset, GTSRBSigns,
                         build_transforms, collate_fn, YoloDetectionDataset)
from cwdetr.engine.utils import targets_to_device
from cwdetr.models.cwdetr import build_cwdetr
from cwdetr.utils.progress import progress as iter_progress


METRIC_KEYS = (
    "detection/map",
    "detection/map50",
    "segmentation/drivable_miou",
    "segmentation/lane_iou",
    "segmentation/lane_f1",
    "sign/top1",
)


class SemanticMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear",
                                   align_corners=False)
        pred = logits.argmax(1).detach().cpu()
        target = target.detach().cpu()
        valid = (target >= 0) & (target < self.num_classes)
        encoded = self.num_classes * target[valid] + pred[valid]
        self.confusion += torch.bincount(
            encoded, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def iou(self) -> torch.Tensor:
        intersection = self.confusion.diag().float()
        union = (self.confusion.sum(0) + self.confusion.sum(1)).float() - intersection
        return torch.where(union > 0, intersection / union, torch.nan)

    def miou(self) -> float:
        value = torch.nanmean(self.iou())
        return float(value) if torch.isfinite(value) else 0.0


class BinaryLaneMetrics:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear",
                                   align_corners=False)
        pred = logits.argmax(1).detach().cpu() > 0
        truth = target.detach().cpu() > 0
        self.tp += int((pred & truth).sum())
        self.fp += int((pred & ~truth).sum())
        self.fn += int((~pred & truth).sum())

    def iou(self) -> float:
        return self.tp / max(1, self.tp + self.fp + self.fn)

    def f1(self) -> float:
        return 2 * self.tp / max(1, 2 * self.tp + self.fp + self.fn)


class SignTop1:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if logits.numel() == 0:
            return
        self.correct += int((logits.argmax(-1) == labels).sum())
        self.total += int(labels.numel())

    def compute(self) -> float:
        return self.correct / max(1, self.total)


class CocoDetectionMetrics:
    def __init__(self, num_classes: int, max_detections: int = 100):
        self.num_classes = num_classes
        self.max_detections = max_detections
        self.images = []
        self.annotations = []
        self.predictions = []
        self._next_image_id = 1
        self._next_annotation_id = 1

    @staticmethod
    def _to_xywh(boxes: torch.Tensor, hw):
        h, w = hw
        cx, cy, bw, bh = boxes.unbind(-1)
        return torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                            bw * w, bh * h], -1)

    def update(self, det_out, targets, orig_sizes, image_ids) -> None:
        logits = det_out["pred_logits"].detach().cpu().sigmoid()
        boxes = det_out["pred_boxes"].detach().cpu()
        for batch_index, target in enumerate(targets):
            if not target.get("train_detection", True):
                continue
            image_id = self._next_image_id
            self._next_image_id += 1
            orig_hw = orig_sizes[batch_index]
            if orig_hw is None:
                raise ValueError("evaluation samples must provide orig_size")
            self.images.append({"id": image_id, "source_id": image_ids[batch_index],
                                "height": orig_hw[0], "width": orig_hw[1]})

            gt_boxes = self._to_xywh(target["boxes"].detach().cpu(), orig_hw)
            for label, box in zip(target["labels"].detach().cpu(), gt_boxes):
                self.annotations.append({
                    "id": self._next_annotation_id,
                    "image_id": image_id,
                    "category_id": int(label) + 1,
                    "bbox": box.tolist(),
                    "area": float(box[2] * box[3]),
                    "iscrowd": 0,
                })
                self._next_annotation_id += 1

            scores, labels = logits[batch_index].flatten().topk(
                min(self.max_detections, logits[batch_index].numel()))
            query_indices = torch.div(labels, self.num_classes, rounding_mode="floor")
            class_indices = labels % self.num_classes
            pred_boxes = self._to_xywh(boxes[batch_index][query_indices], orig_hw)
            for score, label, box in zip(scores, class_indices, pred_boxes):
                self.predictions.append({
                    "image_id": image_id,
                    "category_id": int(label) + 1,
                    "bbox": box.tolist(),
                    "score": float(score),
                })

    def compute(self) -> Dict[str, float]:
        if not self.images or not self.annotations or not self.predictions:
            return {"detection/map": 0.0, "detection/map50": 0.0}
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError as exc:
            raise RuntimeError("pycocotools is required for detection evaluation") from exc

        gt = COCO()
        gt.dataset = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": [{"id": index + 1, "name": str(index)}
                           for index in range(self.num_classes)],
        }
        with contextlib.redirect_stdout(io.StringIO()):
            gt.createIndex()
            dt = gt.loadRes(self.predictions)
            evaluator = COCOeval(gt, dt, "bbox")
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        return {"detection/map": float(evaluator.stats[0]),
                "detection/map50": float(evaluator.stats[1])}


def empty_metrics() -> Dict[str, float]:
    return {key: 0.0 for key in METRIC_KEYS}


@torch.no_grad()
def evaluate_loader(model, loader, device, num_classes: int,
                    max_batches: Optional[int] = None,
                    progress: bool = False) -> Dict[str, float]:
    model.eval()
    detection = CocoDetectionMetrics(num_classes)
    drivable = SemanticMetrics(3)
    lane = BinaryLaneMetrics()
    signs = SignTop1()

    progress_bar = iter_progress(
        loader, desc="validate", dynamic_ncols=True, disable=not progress)
    for batch_index, batch in enumerate(progress_bar):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["images"].to(device, non_blocking=True)
        targets = targets_to_device(batch["targets"], device)
        sign_rois = batch["extras"]["sign_rois"].to(device)
        outputs = model(images, sign_rois=sign_rois if sign_rois.numel() else None)
        detection.update(outputs["detection"], batch["targets"]["detection"],
                         batch["extras"]["orig_sizes"], batch["extras"]["image_ids"])
        if "segmentation" in outputs and targets.get("drivable") is not None:
            drivable.update(outputs["segmentation"]["drivable_logits"], targets["drivable"])
        if "segmentation" in outputs and targets.get("lane") is not None:
            lane.update(outputs["segmentation"]["lane_logits"], targets["lane"])
        if "sign_logits" in outputs and "sign_labels" in targets:
            signs.update(outputs["sign_logits"], targets["sign_labels"])

    metrics = empty_metrics()
    metrics.update(detection.compute())
    metrics["segmentation/drivable_miou"] = drivable.miou()
    metrics["segmentation/lane_iou"] = lane.iou()
    metrics["segmentation/lane_f1"] = lane.f1()
    metrics["sign/top1"] = signs.compute()
    return metrics


def build_eval_dataset(cfg, bdd_root=None, gtsrb_root=None, yolo_data=None,
                       refresh_yolo_index=False):
    transforms = build_transforms(cfg, train=False)
    datasets = []
    if bdd_root:
        datasets.append(BDD100KDataset(bdd_root, "val", transforms, load_seg=True))
    if gtsrb_root:
        datasets.append(GTSRBSigns(gtsrb_root, "test", transforms))
    if yolo_data:
        datasets.append(YoloDetectionDataset(
            yolo_data, "val", transforms,
            expected_num_classes=cfg.model.heads.detection.num_classes,
            refresh_index=refresh_yolo_index))
    if not datasets:
        raise ValueError("provide at least one validation root")
    return ConcatMultiTaskDataset(datasets)


def print_metrics(metrics: Dict[str, float]) -> None:
    width = max(len(key) for key in METRIC_KEYS)
    print("CW-DETR validation metrics")
    for key in METRIC_KEYS:
        print(f"  {key:<{width}}  {metrics[key]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--bdd-root", default=None)
    parser.add_argument("--yolo-data", default=None,
                        help="YOLO data.yaml for detection-only evaluation")
    parser.add_argument("--refresh-yolo-index", action="store_true",
                        help="rescan YOLO images and rebuild parsed-label caches")
    parser.add_argument("--gtsrb-root", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cwdetr(cfg).to(device)
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint.get("ema", checkpoint.get("model", checkpoint)),
                              strict=False)
    dataset = build_eval_dataset(cfg, args.bdd_root, args.gtsrb_root, args.yolo_data,
                                 args.refresh_yolo_index)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_fn,
                        pin_memory=device.type == "cuda")
    metrics = evaluate_loader(model, loader, device,
                              cfg.model.heads.detection.num_classes,
                              args.max_batches, progress=True)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
