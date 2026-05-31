"""Multi-task training loop for the measurable CW-DETR baseline."""
from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from cwdetr.config import config_to_dict, load_config
from cwdetr.data import (BDD100KDataset, ConcatMultiTaskDataset, GTSRBSigns,
                         MixedBatchSampler, NuScenesSequences, build_transforms,
                         collate_fn, YoloDetectionDataset)
from cwdetr.engine.evaluate import build_eval_dataset, evaluate_loader, print_metrics
from cwdetr.engine.utils import (ModelEMA, build_warmup_cosine_scheduler,
                                 make_worker_init_fn, seed_everything,
                                 targets_to_device, unwrap_module)
from cwdetr.models.criterion import MultiTaskCriterion
from cwdetr.models.cwdetr import build_cwdetr


def build_param_groups(model, base_lr, criterion=None, backbone_lr_mult=0.1, wd=1e-4):
    backbone, rest = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if name.startswith("backbone.") else rest).append(parameter)
    groups = [
        {"params": rest, "lr": base_lr, "weight_decay": wd},
        {"params": backbone, "lr": base_lr * backbone_lr_mult, "weight_decay": wd},
    ]
    if criterion is not None:
        criterion_params = [parameter for parameter in criterion.parameters()
                            if parameter.requires_grad]
        if criterion_params:
            groups.append({"params": criterion_params, "lr": base_lr, "weight_decay": wd})
    return [group for group in groups if group["params"]]


def build_datasets(cfg, args):
    transforms = build_transforms(cfg, train=True)
    datasets, weights = [], []
    if args.bdd_root:
        datasets.append(BDD100KDataset(args.bdd_root, "train", transforms, load_seg=True))
        weights.append(2.0)
    if args.yolo_data:
        datasets.append(YoloDetectionDataset(
            args.yolo_data, "train", transforms,
            expected_num_classes=cfg.model.heads.detection.num_classes,
            refresh_index=getattr(args, "refresh_yolo_index", False)))
        weights.append(2.0)
    if args.gtsrb_root:
        datasets.append(GTSRBSigns(args.gtsrb_root, "train", transforms))
        weights.append(1.0)
    if args.nuscenes_root:
        traj = cfg.model.heads.trajectory
        datasets.append(NuScenesSequences(
            args.nuscenes_root, version=args.nuscenes_version, split=args.nuscenes_split,
            future_len=traj.future_len, step_dt=traj.step_dt, space=traj.space,
            transforms=transforms))
        weights.append(1.0)
    if not datasets:
        raise ValueError("provide at least one dataset root, for example --bdd-root or --yolo-data")
    return ConcatMultiTaskDataset(datasets), weights


def _summary_writer(log_dir, rank):
    if rank != 0:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir)
    except ImportError:
        print("[train] tensorboard is unavailable; scalar event logging disabled")
        return None


def _distributed_context():
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def _checkpoint_state(model, criterion, optimizer, scheduler, scaler, ema,
                      cfg, epoch, global_step, best_map):
    model_state = {
        name: value for name, value in unwrap_module(model).state_dict().items()
        if not name.startswith("backbone.teacher.")
    }
    ema_state = {
        name: value for name, value in ema.state_dict().items()
        if not name.startswith("backbone.teacher.")
    }
    return {
        "model": model_state,
        "ema": ema_state,
        "criterion": unwrap_module(criterion).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "cfg": config_to_dict(cfg),
        "epoch": epoch,
        "global_step": global_step,
        "best_detection_map": best_map,
    }


def _load_resume(path, model, criterion, optimizer, scheduler, scaler, ema):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    if "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
    if "criterion" in checkpoint:
        criterion.load_state_dict(checkpoint["criterion"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return (checkpoint.get("epoch", -1) + 1, checkpoint.get("global_step", 0),
            checkpoint.get("best_detection_map", float("-inf")))


def train_one_epoch(model, criterion, loader, optimizer, scaler, device, cfg, epoch,
                    scheduler=None, ema: Optional[ModelEMA] = None, writer=None,
                    global_step=0, clip=0.1, log_every=50, rank=0):
    model.train()
    criterion.train()
    raw_model = unwrap_module(model)
    distill_on = cfg.model.backbone.gram_anchor_distill
    for iteration, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        targets = targets_to_device(batch["targets"], device)
        sign_rois = batch["extras"]["sign_rois"].to(device)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(images, sign_rois=sign_rois if sign_rois.numel() else None,
                            detection_targets=targets["detection"])
            teacher_feat = raw_model.backbone.teacher_features(images) if distill_on else None
            student_feat = outputs["_srcs"][1] if distill_on else None
            losses = criterion(outputs, targets, student_feat, teacher_feat)
            loss = losses["total"]

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([*model.parameters(), *criterion.parameters()], clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)
        global_step += 1

        scalars = {key: float(value.detach()) for key, value in losses.items()
                   if torch.is_tensor(value) and value.ndim == 0}
        if writer is not None:
            for key, value in scalars.items():
                writer.add_scalar(f"train/{key}", value, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
        if rank == 0 and iteration % log_every == 0:
            active = " ".join(
                f"{key}={value:.3f}" for key, value in scalars.items()
                if key in ("detection", "segmentation", "sign", "distill"))
            print(f"[ep{epoch} it{iteration}/{len(loader)}] total={scalars['total']:.3f} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} "
                  f"({batch['extras']['dataset']}) {active}")
    return global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--bdd-root", default=None)
    parser.add_argument("--yolo-data", default=None,
                        help="YOLO data.yaml for detection-only training")
    parser.add_argument("--refresh-yolo-index", action="store_true",
                        help="rescan YOLO images and rebuild parsed-label caches")
    parser.add_argument("--gtsrb-root", default=None)
    parser.add_argument("--nuscenes-root", default=None)
    parser.add_argument("--nuscenes-version", default="v1.0-trainval")
    parser.add_argument("--nuscenes-split", default="train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--ema-decay", type=float, default=0.9998)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--out", default="checkpoints")
    args = parser.parse_args()

    rank, world_size, local_rank, device = _distributed_context()
    seed_everything(args.seed + rank)
    cfg = load_config(args.config)

    model = build_cwdetr(cfg).to(device)
    teacher_dim = (model.backbone.teacher_out_channels
                   if cfg.model.backbone.gram_anchor_distill else None)
    criterion = MultiTaskCriterion(cfg.model.heads.detection.num_classes,
                                   teacher_dim=teacher_dim,
                                   student_dim=cfg.model.hidden_dim,
                                   dn_loss_weight=cfg.model.decoder.dn_loss_weight).to(device)
    dataset, weights = build_datasets(cfg, args)
    sampler = MixedBatchSampler(dataset, args.batch_size, weights, seed=args.seed,
                                rank=rank, world_size=world_size)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                        collate_fn=collate_fn, pin_memory=device.type == "cuda",
                        worker_init_fn=make_worker_init_fn(args.seed, rank))

    optimizer = torch.optim.AdamW(build_param_groups(model, args.lr, criterion))
    scheduler = build_warmup_cosine_scheduler(
        optimizer, args.warmup_steps, args.epochs * max(1, len(loader)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = ModelEMA(model, args.ema_decay)
    start_epoch, global_step, best_map = 0, 0, float("-inf")
    if args.resume:
        start_epoch, global_step, best_map = _load_resume(
            args.resume, model, criterion, optimizer, scheduler, scaler, ema)

    if world_size > 1:
        ddp_kwargs = {"find_unused_parameters": True}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [local_rank]
        model = DDP(model, **ddp_kwargs)
        criterion = DDP(criterion, **ddp_kwargs)

    os.makedirs(args.out, exist_ok=True)
    writer = _summary_writer(os.path.join(args.out, "tensorboard"), rank)
    val_loader = None
    if rank == 0 and (args.bdd_root or args.gtsrb_root or args.yolo_data):
        val_dataset = build_eval_dataset(cfg, args.bdd_root, args.gtsrb_root,
                                         args.yolo_data, args.refresh_yolo_index)
        val_loader = DataLoader(
            val_dataset, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.workers, collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
            worker_init_fn=make_worker_init_fn(args.seed))

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        global_step = train_one_epoch(
            model, criterion, loader, optimizer, scaler, device, cfg, epoch,
            scheduler=scheduler, ema=ema, writer=writer, global_step=global_step,
            rank=rank)

        metrics = None
        if rank == 0 and val_loader is not None and (epoch + 1) % args.eval_every == 0:
            metrics = evaluate_loader(
                ema.module, val_loader, device, cfg.model.heads.detection.num_classes,
                args.max_eval_batches)
            print_metrics(metrics)
            if writer is not None:
                for key, value in metrics.items():
                    writer.add_scalar(f"eval/{key}", value, global_step)

        if rank == 0:
            is_best = False
            if metrics is not None:
                is_best = metrics["detection/map"] > best_map
                best_map = max(best_map, metrics["detection/map"])
            state = _checkpoint_state(model, criterion, optimizer, scheduler, scaler,
                                      ema, cfg, epoch, global_step, best_map)
            epoch_path = os.path.join(args.out, f"{cfg.name}_ep{epoch:03d}.pth")
            torch.save(state, epoch_path)
            if is_best:
                torch.save(state, os.path.join(args.out, "best_detection_map.pth"))
            print(f"saved checkpoint for epoch {epoch}")
        if world_size > 1:
            dist.barrier()

    if writer is not None:
        writer.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
