"""Multi-task training loop for CW-DETR.

Scaffold trainer: param groups (the pretrained DINOv3 backbone gets a 10x lower
LR than the freshly-initialised heads/decoder), AMP, gradient clipping, the
Gram-anchored distillation hookup, and checkpointing. Distributed training is
left to ``torchrun`` + DDP wrapping (one line, marked below).

Run:
    python -m cwdetr.engine.train --config configs/cwdetr_nano_orin.yaml \
        --bdd-root /data/bdd100k --gtsrb-root /data/GTSRB --epochs 50
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from cwdetr.config import load_config
from cwdetr.models.cwdetr import build_cwdetr
from cwdetr.models.criterion import MultiTaskCriterion
from cwdetr.data import (BDD100KDataset, GTSRBSigns, ConcatMultiTaskDataset,
                         MixedBatchSampler, collate_fn, build_transforms)


def build_param_groups(model, base_lr, backbone_lr_mult=0.1, wd=1e-4):
    backbone, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if name.startswith("backbone.") else rest).append(p)
    return [
        {"params": rest, "lr": base_lr, "weight_decay": wd},
        {"params": backbone, "lr": base_lr * backbone_lr_mult, "weight_decay": wd},
    ]


def build_datasets(cfg, args):
    tf_train = build_transforms(cfg, train=True)
    datasets, weights = [], []
    if args.bdd_root:
        datasets.append(BDD100KDataset(args.bdd_root, "train", tf_train, load_seg=True))
        weights.append(2.0)                          # detection+seg backbone signal
    if args.gtsrb_root:
        datasets.append(GTSRBSigns(args.gtsrb_root, "train", tf_train))
        weights.append(1.0)
    # nuScenes (tracking+trajectory) plugs in identically once devkit data is present.
    assert datasets, "Provide at least one dataset root (e.g. --bdd-root)."
    return ConcatMultiTaskDataset(datasets), weights


def train_one_epoch(model, criterion, loader, optim, scaler, device, cfg, epoch,
                    clip=0.1, log_every=50):
    model.train()
    distill_on = cfg.model.backbone.gram_anchor_distill
    for it, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        targets = _targets_to_device(batch["targets"], device)
        sign_rois = batch["extras"]["sign_rois"].to(device)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out = model(images, sign_rois=sign_rois if sign_rois.numel() else None)
            teacher_feat = model.backbone.teacher_features(images) if distill_on else None
            student_feat = out["_srcs"][1] if distill_on else None      # stride-16 level
            losses = criterion(out, targets, student_feat, teacher_feat)
            loss = losses["total"]

        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optim)
        scaler.update()

        if it % log_every == 0:
            parts = {k: float(v) for k, v in losses.items()
                     if torch.is_tensor(v) and v.ndim == 0 and k != "total"}
            print(f"[ep{epoch} it{it}/{len(loader)}] total={float(loss):.3f} "
                  f"({batch['extras']['dataset']}) "
                  + " ".join(f"{k}={v:.3f}" for k, v in parts.items()
                             if k in ('detection', 'segmentation', 'sign', 'distill')))


def _targets_to_device(targets, device):
    out = {}
    for k, v in targets.items():
        if v is None:
            out[k] = None
        elif k == "detection":
            out[k] = [{kk: vv.to(device) for kk, vv in t.items()} for t in v]
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bdd-root", default=None)
    ap.add_argument("--gtsrb-root", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_cwdetr(cfg).to(device)
    teacher_dim = 768 if cfg.model.backbone.gram_anchor_distill else None
    criterion = MultiTaskCriterion(cfg.model.heads.detection.num_classes,
                                   teacher_dim=teacher_dim,
                                   student_dim=cfg.model.hidden_dim).to(device)

    dataset, weights = build_datasets(cfg, args)
    sampler = MixedBatchSampler(dataset, args.batch_size, weights)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                        collate_fn=collate_fn, pin_memory=True)

    optim = torch.optim.AdamW(build_param_groups(model, args.lr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(args.out, exist_ok=True)
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, loader, optim, scaler, device, cfg, epoch)
        sched.step()
        ckpt = {"model": model.state_dict(), "cfg": args.config, "epoch": epoch}
        torch.save(ckpt, os.path.join(args.out, f"{cfg.name}_ep{epoch:03d}.pth"))
        print(f"saved checkpoint for epoch {epoch}")


if __name__ == "__main__":
    main()
