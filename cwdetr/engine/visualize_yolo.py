"""Draw CW-DETR detections on a small YOLO-format split."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

from cwdetr.config import load_config
from cwdetr.data import YoloDetectionDataset, build_transforms, collate_fn
from cwdetr.engine.utils import dataloader_worker_kwargs
from cwdetr.models.cwdetr import build_cwdetr
from cwdetr.utils.box_ops import box_cxcywh_to_xyxy


def _prediction_rows(det_out, image_hw, orig_hw, class_names, score_thresh, topk):
    logits = det_out["pred_logits"]
    boxes = det_out["pred_boxes"]
    if logits.ndim == 3:
        logits = logits[0]
        boxes = boxes[0]
    logits = logits.sigmoid()
    scores, labels = logits.max(-1)
    keep = scores >= score_thresh
    if keep.any():
        kept_scores = scores[keep]
        order = kept_scores.argsort(descending=True)[:topk]
        kept_boxes = boxes[keep][order]
        kept_labels = labels[keep][order]
        kept_scores = kept_scores[order]
    else:
        return []

    in_h, in_w = image_hw
    orig_h, orig_w = orig_hw
    xyxy = box_cxcywh_to_xyxy(kept_boxes)
    xyxy = xyxy * xyxy.new_tensor([in_w, in_h, in_w, in_h])
    xyxy = xyxy * xyxy.new_tensor([orig_w / in_w, orig_h / in_h,
                                   orig_w / in_w, orig_h / in_h])
    rows = []
    for box, label, score in zip(xyxy.cpu(), kept_labels.cpu(), kept_scores.cpu()):
        class_id = int(label)
        rows.append((box.tolist(), class_names[class_id], float(score), "pred"))
    return rows


def _ground_truth_rows(target, orig_hw, class_names):
    orig_h, orig_w = orig_hw
    boxes = box_cxcywh_to_xyxy(target["boxes"].cpu())
    boxes = boxes * boxes.new_tensor([orig_w, orig_h, orig_w, orig_h])
    return [(box.tolist(), class_names[int(label)], 1.0, "gt")
            for box, label in zip(boxes, target["labels"].cpu())]


def _draw_rows(image_path: Path, rows, output_path: Path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = {"pred": "red", "gt": "lime"}
    for box, name, score, kind in rows:
        x1, y1, x2, y2 = box
        color = colors[kind]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"{kind}:{name}" if kind == "gt" else f"{name} {score:.2f}"
        text_box = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((x1, y1), label, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--yolo-data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", default="visualizations/yolo_predictions")
    parser.add_argument("--num-images", type=int, default=15)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--score-thresh", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--draw-gt", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-persistent-workers", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cwdetr(cfg).to(device).eval()
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(checkpoint.get("ema", checkpoint.get("model", checkpoint)),
                          strict=False)

    dataset = YoloDetectionDataset(
        args.yolo_data, args.split, build_transforms(cfg, train=False),
        expected_num_classes=cfg.model.heads.detection.num_classes)
    indices = list(range(args.start_index, min(len(dataset), args.start_index + args.num_images)))
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False,
                        collate_fn=collate_fn,
                        **dataloader_worker_kwargs(args, device, seed=1337))

    output_dir = Path(args.out)
    for local_index, batch in enumerate(loader):
        image = batch["images"].to(device, non_blocking=True)
        outputs = model(image)
        global_index = indices[local_index]
        image_path = dataset.images[global_index]
        orig_hw = batch["extras"]["orig_sizes"][0]
        rows = _prediction_rows(
            outputs["detection"], image.shape[-2:], orig_hw, dataset.class_names,
            args.score_thresh, args.topk)
        pred_count = len(rows)
        gt_count = 0
        if args.draw_gt:
            gt_rows = _ground_truth_rows(
                batch["targets"]["detection"][0], orig_hw, dataset.class_names)
            gt_count = len(gt_rows)
            rows.extend(gt_rows)
        output_path = output_dir / f"{global_index:06d}_{image_path.stem}.jpg"
        _draw_rows(image_path, rows, output_path)
        print(f"wrote {output_path} preds={pred_count} gt={gt_count}", flush=True)


if __name__ == "__main__":
    main()
