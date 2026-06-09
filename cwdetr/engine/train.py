"""Multi-task training loop for the measurable CW-DETR baseline."""
from __future__ import annotations

import argparse
import os
import time
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
                                 dataloader_worker_kwargs, seed_everything,
                                 targets_to_device, unwrap_module)
from cwdetr.models.criterion import MultiTaskCriterion
from cwdetr.models.cwdetr import build_cwdetr
from cwdetr.utils.progress import progress, progress_write


def _log(message, rank=0):
    if rank == 0:
        print(f"[train] {message}", flush=True)


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
            refresh_index=getattr(args, "refresh_yolo_index", False),
            image_cache=getattr(args, "yolo_image_cache", "none")))
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
        _log("tensorboard is unavailable; scalar event logging disabled", rank)
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
                      cfg, epoch, global_step, best_map, checkpoint_kind="epoch",
                      resume_epoch=None, resume_iteration=0):
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
        "resume_epoch": epoch + 1 if resume_epoch is None else resume_epoch,
        "resume_iteration": resume_iteration,
        "global_step": global_step,
        "best_detection_map": best_map,
        "checkpoint_kind": checkpoint_kind,
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
    return (checkpoint.get("resume_epoch", checkpoint.get("epoch", -1) + 1),
            checkpoint.get("global_step", 0),
            checkpoint.get("best_detection_map", float("-inf")),
            checkpoint.get("resume_iteration", 0))


def train_one_epoch(model, criterion, loader, optimizer, scaler, device, cfg, epoch,
                    scheduler=None, ema: Optional[ModelEMA] = None, writer=None,
                    global_step=0, clip=0.1, log_every=10, rank=0,
                    num_epochs=None, start_iteration=0, checkpoint_callback=None):
    model.train()
    criterion.train()
    raw_model = unwrap_module(model)
    distill_on = cfg.model.backbone.gram_anchor_distill
    if start_iteration >= len(loader):
        _log(f"resuming epoch {epoch + 1}: all {len(loader)} batches already completed", rank)
        return global_step, {}
    epoch_label = f"{epoch + 1}/{num_epochs}" if num_epochs is not None else str(epoch + 1)
    iterator = progress(loader, desc=f"train epoch {epoch_label}", dynamic_ncols=True,
                        disable=rank != 0)
    if start_iteration:
        _log(f"resuming epoch {epoch + 1}: skipping {start_iteration} completed batches", rank)
    accum, count = {}, 0
    log_start = time.perf_counter()
    log_images = 0
    if rank == 0 and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for iteration, batch in enumerate(iterator):
        if iteration < start_iteration:
            continue
        images = batch["images"].to(device, non_blocking=True)
        targets = targets_to_device(batch["targets"], device)
        sign_rois = batch["extras"]["sign_rois"].to(device)
        log_images += int(images.shape[0])

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
        count += 1
        for key, value in scalars.items():
            accum[key] = accum.get(key, 0.0) + value
        if writer is not None:
            for key, value in scalars.items():
                writer.add_scalar(f"train/{key}", value, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
        precision = scalars.get("det/precision", 0.0)
        iterator.set_postfix(
            loss=f"{scalars.get('total', 0.0):.3f}",
            det=f"{scalars.get('detection', 0.0):.3f}",
            prec=f"{precision:.3f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        if checkpoint_callback is not None:
            checkpoint_callback(epoch, iteration, global_step)
        if rank == 0 and iteration % log_every == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = max(time.perf_counter() - log_start, 1e-9)
            imgs_per_sec = log_images / elapsed
            gpu_memory = ""
            if device.type == "cuda":
                gib = 1024 ** 3
                allocated = torch.cuda.max_memory_allocated(device) / gib
                reserved = torch.cuda.max_memory_reserved(device) / gib
                gpu_memory = f" gpu_alloc={allocated:.2f}GiB gpu_reserved={reserved:.2f}GiB"
                torch.cuda.reset_peak_memory_stats(device)
            log_start = time.perf_counter()
            log_images = 0
            active = " ".join(
                f"{key}={value:.3f}" for key, value in scalars.items()
                if key in ("detection", "det/precision", "det/mean_score",
                           "segmentation", "sign", "distill"))
            message = (f"epoch={epoch + 1} batch={iteration + 1}/{len(loader)} "
                       f"step={global_step} total={scalars['total']:.3f} "
                       f"lr={optimizer.param_groups[0]['lr']:.2e} "
                       f"imgs/s={imgs_per_sec:.1f}{gpu_memory} "
                       f"dataset={batch['extras']['dataset']} {active}")
            progress_write(f"[train] {message}")
    summary = {key: value / max(1, count) for key, value in accum.items()}
    if rank == 0:
        progress_write(
            "[train] epoch_summary "
            f"epoch={epoch + 1} total={summary.get('total', 0.0):.4f} "
            f"detection={summary.get('detection', 0.0):.4f} "
            f"precision={summary.get('det/precision', 0.0):.4f} "
            f"mean_score={summary.get('det/mean_score', 0.0):.4f}")
    return global_step, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--bdd-root", default=None)
    parser.add_argument("--yolo-data", default=None,
                        help="YOLO data.yaml for detection-only training")
    parser.add_argument("--refresh-yolo-index", action="store_true",
                        help="rescan YOLO images and rebuild parsed-label caches")
    parser.add_argument("--yolo-image-cache", default="none",
                        choices=("none", "ram-float32"),
                        help="experimental YOLO image cache mode")
    parser.add_argument("--gtsrb-root", default=None)
    parser.add_argument("--nuscenes-root", default=None)
    parser.add_argument("--nuscenes-version", default="v1.0-trainval")
    parser.add_argument("--nuscenes-split", default="train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1,
                        help="backbone LR multiplier; 0.1 gives 2e-5 for lr=2e-4")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad-norm", type=float, default=0.1)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="train projector/decoder/heads only; keep DINOv3 encoder frozen")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--ema-decay", type=float, default=0.9998)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="batches prefetched per worker when --workers > 0")
    parser.add_argument("--no-persistent-workers", action="store_true",
                        help="disable persistent DataLoader workers between epochs")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10,
                        help="emit a flushed training summary every N batches")
    parser.add_argument("--step-checkpoint-every", type=int, default=1,
                        help="overwrite last_step.pth every N optimizer steps; 0 disables")
    parser.add_argument("--keep-step-checkpoints", action="store_true",
                        help="also keep numbered step_XXXXXXXX.pth checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--out", default="checkpoints")
    args = parser.parse_args()
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    if args.workers < 0 or args.prefetch_factor <= 0:
        parser.error("--workers must be non-negative and --prefetch-factor must be positive")
    if args.step_checkpoint_every < 0:
        parser.error("--step-checkpoint-every must be non-negative")
    if args.backbone_lr_mult < 0 or args.weight_decay < 0 or args.clip_grad_norm <= 0:
        parser.error("--backbone-lr-mult and --weight-decay must be non-negative; "
                     "--clip-grad-norm must be positive")

    rank, world_size, local_rank, device = _distributed_context()
    _log(f"starting rank={rank}/{world_size} device={device}", rank)
    if rank == 0 and device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        _log(f"gpu={props.name} memory={props.total_memory / 1024 ** 3:.1f} GiB", rank)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    seed_everything(args.seed + rank)
    _log(f"loading config {args.config}", rank)
    cfg = load_config(args.config)
    if args.freeze_backbone:
        cfg.model.backbone.train_backbone = False

    _log("building model and loading backbone weights", rank)
    model = build_cwdetr(cfg).to(device)
    if not cfg.model.backbone.train_backbone:
        _log("DINOv3 encoder is frozen; training projector/decoder/heads only", rank)
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    _log(f"model ready: {trainable / 1e6:.2f}M trainable parameters", rank)
    teacher_dim = (model.backbone.teacher_out_channels
                   if cfg.model.backbone.gram_anchor_distill else None)
    criterion = MultiTaskCriterion(cfg.model.heads.detection.num_classes,
                                   teacher_dim=teacher_dim,
                                   student_dim=cfg.model.hidden_dim,
                                   dn_loss_weight=cfg.model.decoder.dn_loss_weight).to(device)
    _log("building training datasets", rank)
    dataset, weights = build_datasets(cfg, args)
    _log(f"training dataset ready: {len(dataset)} samples", rank)
    sampler = MixedBatchSampler(dataset, args.batch_size, weights, seed=args.seed,
                                rank=rank, world_size=world_size)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn,
                        **dataloader_worker_kwargs(args, device, args.seed, rank))
    _log(f"training loader ready: {len(loader)} batches/epoch batch_size={args.batch_size} "
         f"workers={args.workers} prefetch={args.prefetch_factor if args.workers else 0} "
         f"persistent_workers={args.workers > 0 and not args.no_persistent_workers}", rank)

    _log("building optimizer, scheduler, scaler, and EMA", rank)
    optimizer = torch.optim.AdamW(build_param_groups(
        model, args.lr, criterion, args.backbone_lr_mult, args.weight_decay))
    scheduler = build_warmup_cosine_scheduler(
        optimizer, args.warmup_steps, args.epochs * max(1, len(loader)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = ModelEMA(model, args.ema_decay)
    start_epoch, resume_iteration, global_step, best_map = 0, 0, 0, float("-inf")
    if args.resume:
        _log(f"resuming checkpoint {args.resume}", rank)
        start_epoch, global_step, best_map, resume_iteration = _load_resume(
            args.resume, model, criterion, optimizer, scheduler, scaler, ema)
        _log(f"resume ready: start_epoch={start_epoch + 1} "
             f"resume_iteration={resume_iteration} global_step={global_step}", rank)

    if world_size > 1:
        ddp_kwargs = {"find_unused_parameters": True}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [local_rank]
        model = DDP(model, **ddp_kwargs)
        criterion = DDP(criterion, **ddp_kwargs)

    os.makedirs(args.out, exist_ok=True)
    _log(f"writing logs and checkpoints to {args.out}", rank)
    if rank == 0 and args.step_checkpoint_every:
        _log(f"overwriting {os.path.join(args.out, 'last_step.pth')} every "
             f"{args.step_checkpoint_every} optimizer step(s)", rank)
        if "/content/drive" in args.out.replace("\\", "/").lower() and args.step_checkpoint_every == 1:
            _log("WARNING: full per-step checkpoints on Google Drive can dominate runtime. "
                 "Use --step-checkpoint-every 50 or 100 for higher GPU utilization.", rank)
    writer = _summary_writer(os.path.join(args.out, "tensorboard"), rank)
    val_loader = None
    if rank == 0 and (args.bdd_root or args.gtsrb_root or args.yolo_data):
        _log("building validation datasets", rank)
        val_dataset = build_eval_dataset(cfg, args.bdd_root, args.gtsrb_root,
                                         args.yolo_data, args.refresh_yolo_index,
                                         args.yolo_image_cache)
        val_loader = DataLoader(
            val_dataset, batch_size=args.eval_batch_size, shuffle=False,
            collate_fn=collate_fn,
            **dataloader_worker_kwargs(args, device, args.seed))
        _log(f"validation loader ready: {len(val_dataset)} samples, "
             f"{len(val_loader)} batches", rank)

    for epoch in range(start_epoch, args.epochs):
        _log(f"starting epoch {epoch + 1}/{args.epochs}", rank)
        sampler.set_epoch(epoch)
        def save_step_checkpoint(current_epoch, iteration, step):
            if rank != 0 or not args.step_checkpoint_every:
                return
            if step % args.step_checkpoint_every:
                return
            next_iteration = iteration + 1
            state = _checkpoint_state(
                model, criterion, optimizer, scheduler, scaler, ema, cfg,
                current_epoch, step, best_map, checkpoint_kind="step",
                resume_epoch=current_epoch, resume_iteration=next_iteration)
            last_path = os.path.join(args.out, "last_step.pth")
            torch.save(state, last_path)
            if args.keep_step_checkpoints:
                torch.save(state, os.path.join(args.out, f"step_{step:08d}.pth"))

        global_step, train_summary = train_one_epoch(
            model, criterion, loader, optimizer, scaler, device, cfg, epoch,
            scheduler=scheduler, ema=ema, writer=writer, global_step=global_step,
            clip=args.clip_grad_norm, log_every=args.log_every, rank=rank,
            num_epochs=args.epochs,
            start_iteration=resume_iteration if epoch == start_epoch else 0,
            checkpoint_callback=save_step_checkpoint)
        resume_iteration = 0
        if writer is not None:
            for key, value in train_summary.items():
                writer.add_scalar(f"epoch_train/{key}", value, epoch + 1)

        metrics = None
        if rank == 0 and val_loader is not None and (epoch + 1) % args.eval_every == 0:
            _log(f"evaluating EMA model after epoch {epoch + 1}", rank)
            metrics = evaluate_loader(
                ema.module, val_loader, device, cfg.model.heads.detection.num_classes,
                args.max_eval_batches, progress=True)
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
                                      ema, cfg, epoch, global_step, best_map,
                                      checkpoint_kind="epoch",
                                      resume_epoch=epoch + 1, resume_iteration=0)
            epoch_path = os.path.join(args.out, f"{cfg.name}_ep{epoch:03d}.pth")
            torch.save(state, epoch_path)
            torch.save(state, os.path.join(args.out, "last_epoch.pth"))
            if is_best:
                torch.save(state, os.path.join(args.out, "best_detection_map.pth"))
            _log(f"saved checkpoint {epoch_path}", rank)
        if world_size > 1:
            dist.barrier()

    if writer is not None:
        writer.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
