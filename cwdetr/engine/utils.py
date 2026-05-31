"""Shared training/evaluation utilities."""
from __future__ import annotations

import copy
import math
import random
from functools import partial

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


def unwrap_module(module):
    return module.module if hasattr(module, "module") else module


def targets_to_device(targets, device):
    out = {}
    for key, value in targets.items():
        if value is None:
            out[key] = None
        elif key == "detection":
            out[key] = [
                {name: item.to(device) if torch.is_tensor(item) else item
                 for name, item in target.items()}
                for target in value
            ]
        elif torch.is_tensor(value):
            out[key] = value.to(device)
        elif isinstance(value, list) and all(torch.is_tensor(item) for item in value):
            out[key] = [item.to(device) for item in value]
        else:
            out[key] = value
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int, seed: int, rank: int) -> None:
    worker_seed = seed + rank * 100_000 + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_worker_init_fn(seed: int, rank: int = 0):
    return partial(_seed_worker, seed=seed, rank=rank)


def build_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    total_steps = max(1, total_steps)
    warmup_steps = min(max(0, warmup_steps), total_steps - 1)

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, scale)


class ModelEMA:
    def __init__(self, model, decay: float = 0.9998):
        self.decay = decay
        self.module = copy.deepcopy(unwrap_module(model)).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        source = unwrap_module(model).state_dict()
        for name, value in self.module.state_dict().items():
            incoming = source[name].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(incoming, alpha=1.0 - self.decay)
            else:
                value.copy_(incoming)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict, strict=False)
