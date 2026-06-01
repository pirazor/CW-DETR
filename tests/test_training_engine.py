from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cwdetr.data.multitask_dataset import (ConcatMultiTaskDataset, MixedBatchSampler,
                                           collate_fn)
from cwdetr.config import CWDETRConfig
from cwdetr.data.traffic_signs import GTSRBSigns
from cwdetr.data.transforms import RandomScaleCrop
from cwdetr.engine.evaluate import (METRIC_KEYS, BinaryLaneMetrics, SemanticMetrics,
                                    SignTop1, evaluate_loader)
from cwdetr.engine.train import _checkpoint_state, train_one_epoch
from cwdetr.engine.utils import ModelEMA, build_warmup_cosine_scheduler


class _SizedDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


class _EvalDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, _):
        return {
            "image": torch.zeros(3, 4, 4),
            "labels": torch.zeros(0, dtype=torch.long),
            "boxes": torch.zeros(0, 4),
            "train_detection": False,
            "drivable": torch.tensor([[0, 0, 1, 1]] * 4),
            "lane": torch.tensor([[0, 1, 0, 1]] * 4),
            "sign_boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
            "sign_labels": torch.tensor([2]),
            "image_id": "synthetic",
            "orig_size": (4, 4),
            "dataset": "synthetic",
        }


class _EvalModel(nn.Module):
    def forward(self, images, sign_rois=None):
        batch = images.shape[0]
        drivable = torch.full((batch, 2, 4, 4), -10.0)
        drivable[:, 0, :, :2] = 10.0
        drivable[:, 1, :, 2:] = 10.0
        lane = torch.full((batch, 2, 4, 4), -10.0)
        lane[:, 0, :, 0::2] = 10.0
        lane[:, 1, :, 1::2] = 10.0
        sign_logits = torch.full((1, 4), -10.0)
        sign_logits[0, 2] = 10.0
        return {
            "detection": {
                "pred_logits": torch.zeros(batch, 2, 3),
                "pred_boxes": torch.zeros(batch, 2, 4),
            },
            "segmentation": {"drivable_logits": drivable, "lane_logits": lane},
            "sign_logits": sign_logits,
        }


def test_metric_accumulators():
    logits = torch.tensor([[[[10.0, -10.0]], [[-10.0, 10.0]]]])
    target = torch.tensor([[[0, 1]]])
    semantic = SemanticMetrics(2)
    semantic.update(logits, target)
    assert semantic.miou() == 1.0

    lane = BinaryLaneMetrics()
    lane.update(logits, target)
    assert lane.iou() == 1.0
    assert lane.f1() == 1.0

    signs = SignTop1()
    signs.update(torch.tensor([[0.0, 3.0], [4.0, 0.0]]), torch.tensor([1, 0]))
    assert signs.compute() == 1.0


def test_synthetic_evaluate_loader_reports_all_keys():
    loader = DataLoader(_EvalDataset(), batch_size=1, collate_fn=collate_fn)
    metrics = evaluate_loader(_EvalModel(), loader, torch.device("cpu"), num_classes=3)
    assert tuple(metrics) == METRIC_KEYS
    assert metrics["segmentation/drivable_miou"] == 1.0
    assert metrics["segmentation/lane_iou"] == 1.0
    assert metrics["segmentation/lane_f1"] == 1.0
    assert metrics["sign/top1"] == 1.0


def test_random_scale_crop_updates_boxes_masks_and_sign_rois():
    sample = {
        "image": Image.new("RGB", (4, 4)),
        "labels": torch.tensor([0]),
        "boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        "drivable": Image.new("L", (4, 4), color=1),
        "lane": Image.new("L", (4, 4), color=1),
        "sign_boxes": torch.tensor([[1.0, 1.0, 3.0, 3.0]]),
        "sign_labels": torch.tensor([2]),
    }
    transformed = RandomScaleCrop((4, 4), scale_min=2.0, scale_max=2.0,
                                  crop_prob=0.0)(sample)
    assert transformed["image"].size == (4, 4)
    assert transformed["drivable"].size == (4, 4)
    assert torch.allclose(transformed["boxes"], torch.tensor([[0.5, 0.5, 1.0, 1.0]]))
    assert torch.allclose(transformed["sign_boxes"], torch.tensor([[0.0, 0.0, 4.0, 4.0]]))


def test_gtsrb_flat_test_csv_layout(tmp_path):
    test_dir = tmp_path / "Test"
    test_dir.mkdir()
    Image.new("RGB", (4, 4)).save(test_dir / "00001.png")
    (tmp_path / "Test.csv").write_text("Path,ClassId\nTest/00001.png,3\n", encoding="utf-8")
    dataset = GTSRBSigns(str(tmp_path), "test")
    sample = dataset[0]
    assert sample["sign_labels"].tolist() == [3]
    assert sample["labels"].numel() == 0
    assert not sample["train_detection"]


def test_mixed_batch_sampler_is_seeded_and_rank_sharded():
    concat = ConcatMultiTaskDataset([_SizedDataset(8), _SizedDataset(8)])
    first = MixedBatchSampler(concat, 2, seed=7)
    second = MixedBatchSampler(concat, 2, seed=7)
    assert list(first) == list(second)
    first.set_epoch(1)
    assert list(first) != list(second)

    rank0 = list(MixedBatchSampler(concat, 2, seed=7, rank=0, world_size=2))
    rank1 = list(MixedBatchSampler(concat, 2, seed=7, rank=1, world_size=2))
    assert not {tuple(batch) for batch in rank0} & {tuple(batch) for batch in rank1}
    assert len(rank0) + len(rank1) == 8


def test_warmup_cosine_scheduler_and_ema_checkpoint_state():
    model = nn.Linear(2, 1)
    criterion = nn.Linear(1, 1)
    optimizer = torch.optim.SGD([*model.parameters(), *criterion.parameters()], lr=1.0)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=2, total_steps=6)
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    assert lrs[0] < lrs[1] <= 1.0
    assert lrs[-1] < lrs[2]

    ema = ModelEMA(model, decay=0.5)
    before = ema.module.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(2.0)
    ema.update(model)
    assert torch.allclose(ema.module.weight, before + 1.0)

    scaler = torch.amp.GradScaler("cuda", enabled=False)
    cfg = CWDETRConfig()
    state = _checkpoint_state(model, criterion, optimizer, scheduler, scaler, ema,
                              cfg, epoch=2, global_step=9, best_map=0.4)
    assert state["epoch"] == 2
    assert state["global_step"] == 9
    assert state["best_detection_map"] == 0.4


def test_train_one_epoch_emits_flushed_batch_summary(capsys):
    class TrainModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, images, sign_rois=None, detection_targets=None):
            return {"score": self.weight * images.sum()}

    class TrainCriterion(nn.Module):
        def forward(self, outputs, targets, student_feat=None, teacher_feat=None):
            loss = outputs["score"] ** 2
            return {"total": loss, "detection": loss}

    cfg = CWDETRConfig()
    model = TrainModel()
    criterion = TrainCriterion()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loader = [{
        "images": torch.ones(1, 3, 2, 2),
        "targets": {
            "detection": [{"labels": torch.zeros(0, dtype=torch.long),
                           "boxes": torch.zeros(0, 4)}],
            "drivable": None,
            "lane": None,
            "sign_labels": torch.zeros(0, dtype=torch.long),
        },
        "extras": {"sign_rois": torch.zeros(0, 5), "dataset": "synthetic"},
    }]
    step = train_one_epoch(
        model, criterion, loader, optimizer, scaler, torch.device("cpu"), cfg,
        epoch=0, log_every=1, num_epochs=1)
    output = capsys.readouterr().out
    assert step == 1
    assert "[train] epoch=1 batch=1/1 step=1" in output
