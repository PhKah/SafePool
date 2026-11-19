"""Vòng lặp train/eval chuẩn PyTorch cho SafePool (WP-2.1).

Hợp đồng dữ liệu (Data Contract): mỗi batch của DataLoader là cặp `(inputs, targets)` đều là Tensor.
- Stage 2: inputs (B,3,H,W) float, targets (B,) long ∈ {0,1,2}, model(inputs) -> logits (B,3).
- Stage 1: dùng lại `train_one_epoch` bằng cách truyền `criterion` bọc loss detection của bạn; còn đánh giá
  dùng `metrics.detection_metrics` trên box đã hậu xử lý (NMS).
"""
from __future__ import annotations

import math
import os
import random
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics import NUM_CLASSES, confusion_matrix, metrics_from_confusion_matrix


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Cố định mọi nguồn ngẫu nhiên để lượt chạy tái lập được."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """'auto' ưu tiên CUDA > MPS > CPU; hoặc chỉ định 'cpu' / 'cuda' / 'mps'."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(preference)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Yêu cầu CUDA nhưng không khả dụng")
    if device.type == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("Yêu cầu MPS nhưng không khả dụng")
    return device


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: Optional[float] = None,
    max_batches: Optional[int] = None,
) -> dict:
    """Train 1 epoch. Trả {'loss': trung bình theo mẫu, 'num_samples': int}. Ném FloatingPointError nếu loss NaN/Inf."""
    model.train()
    loss_sum, n_samples = 0.0, 0
    for step, (inputs, targets) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss_value = loss.item()  # float Python -> không giữ đồ thị autograd (tránh rò rỉ VRAM)
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Loss không hữu hạn ({loss_value}) tại step {step}")
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = inputs.size(0)
        loss_sum += loss_value * bs
        n_samples += bs
    if n_samples == 0:
        raise ValueError("DataLoader rỗng — không có batch nào để train")
    return {"loss": loss_sum / n_samples, "num_samples": n_samples}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[Callable],
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    max_batches: Optional[int] = None,
) -> dict:
    """Đánh giá Stage 2: cộng dồn confusion matrix qua batch rồi tính metric một lần (đúng cả khi batch thiếu lớp)."""
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    loss_sum, n_samples = 0.0, 0
    for step, (inputs, targets) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        if criterion is not None:
            loss_value = criterion(logits, targets).item()
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"Loss không hữu hạn ({loss_value}) tại step {step}")
            loss_sum += loss_value * inputs.size(0)
        cm += confusion_matrix(targets.cpu(), logits.argmax(dim=1).cpu(), num_classes)
        n_samples += inputs.size(0)
    if n_samples == 0:
        raise ValueError("DataLoader rỗng — không có batch nào để đánh giá")
    result = metrics_from_confusion_matrix(cm)
    result["loss"] = loss_sum / n_samples if criterion is not None else None
    return result
