#!/usr/bin/env python3
"""
scripts/sanity_check.py
=======================
Script Sanity Check (Overfit 1 batch — Gate 2, WP-2.2) cho SafePool.

Tái sử dụng trực tiếp:
  - `src/engine/trainer.py`: set_seed, get_device, train_one_epoch, evaluate (WP-2.1)
  - `src/metrics.py`: detection_metrics, box_iou_xyxy (WP-2.1)
  - `src/datasets/stage1_dataset.py`: Stage1Dataset, stage1_collate_fn (WP-1.2)
  - `src/datasets/stage2_dataset.py`: Stage2ROIDataset (WP-1.3)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.datasets.stage1_dataset import DEFAULT_MANIFEST_PATH, Stage1Dataset, stage1_collate_fn
from src.datasets.stage2_dataset import Stage2ROIDataset
from src.engine.trainer import evaluate, get_device, set_seed, train_one_epoch
from src.metrics import box_iou_xyxy, detection_metrics


# =============================================================================
# Mạng mồi tối giản From Scratch (chỉ dùng cho Sanity Check WP-2.2)
# =============================================================================
class TinyStage2Probe(nn.Module):
    """CNN 3 tầng tối giản From Scratch cho Stage 2 (3-class ROI 128x128)."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class TinyStage1Probe(nn.Module):
    """Detector lưới 13x13 tối giản From Scratch cho Stage 1 (1-class person 416x416)."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),   # 208x208
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 104x104
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 52x52
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 26x26
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 13x13
            nn.ReLU(inplace=True),
        )
        # 5 kênh: [obj_logit, tx, ty, tw, th]
        self.head = nn.Conv2d(128, 5, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# =============================================================================
# Detection Loss & Giải mã + NMS tối giản cho Stage 1 Sanity Check
# =============================================================================
class Stage1SimpleDetectionLoss(nn.Module):
    """Detection Loss cơ sở (Objectness BCE + Box SmoothL1) cho lưới SxS."""

    def __init__(self, pos_weight: float = 5.0, box_weight: float = 5.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.box_weight = box_weight

    def forward(self, preds: torch.Tensor, targets: Sequence[torch.Tensor]) -> torch.Tensor:
        bs, _, grid_h, grid_w = preds.shape
        device = preds.device

        obj_target = torch.zeros((bs, grid_h, grid_w), dtype=preds.dtype, device=device)
        box_target = torch.zeros((bs, grid_h, grid_w, 4), dtype=preds.dtype, device=device)
        pos_mask = torch.zeros((bs, grid_h, grid_w), dtype=torch.bool, device=device)

        for b_idx, t in enumerate(targets):
            if t.numel() == 0:
                continue
            for row in t:
                cx, cy, bw, bh = row[1], row[2], row[3], row[4]
                gx = int(torch.clamp(cx * grid_w, 0, grid_w - 1).item())
                gy = int(torch.clamp(cy * grid_h, 0, grid_h - 1).item())
                obj_target[b_idx, gy, gx] = 1.0
                box_target[b_idx, gy, gx, 0] = cx * grid_w - gx
                box_target[b_idx, gy, gx, 1] = cy * grid_h - gy
                box_target[b_idx, gy, gx, 2] = bw
                box_target[b_idx, gy, gx, 3] = bh
                pos_mask[b_idx, gy, gx] = True

        pos_w = torch.tensor(self.pos_weight, dtype=preds.dtype, device=device)
        obj_loss = F.binary_cross_entropy_with_logits(preds[:, 0], obj_target, pos_weight=pos_w)

        if pos_mask.any():
            box_pred = torch.sigmoid(preds[:, 1:5]).permute(0, 2, 3, 1)
            box_loss = F.smooth_l1_loss(box_pred[pos_mask], box_target[pos_mask])
        else:
            box_loss = preds.sum() * 0.0

        return obj_loss + self.box_weight * box_loss


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy NMS thuần NumPy trên định dạng [x1, y1, x2, y2]."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    order = np.argsort(-scores)
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou_xyxy(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < iou_thr]
    return boxes[keep], scores[keep]


def decode_stage1_predictions(
    preds: torch.Tensor,
    score_thr: float = 0.25,
    nms_iou_thr: float = 0.5,
) -> List[Dict[str, np.ndarray]]:
    """Giải mã tensor [B, 5, S, S] thành danh sách dict {'boxes', 'scores'} chuẩn xyxy [0, 1]."""
    preds_cpu = preds.detach().cpu()
    bs, _, grid_h, grid_w = preds_cpu.shape
    obj_probs = torch.sigmoid(preds_cpu[:, 0]).numpy()
    box_vals = torch.sigmoid(preds_cpu[:, 1:5]).numpy()

    results: List[Dict[str, np.ndarray]] = []
    for b_idx in range(bs):
        boxes_list = []
        scores_list = []
        for gy in range(grid_h):
            for gx in range(grid_w):
                score = float(obj_probs[b_idx, gy, gx])
                if score < score_thr:
                    continue
                tx, ty, tw, th = box_vals[b_idx, :, gy, gx]
                cx = (gx + float(tx)) / grid_w
                cy = (gy + float(ty)) / grid_h
                x1 = max(0.0, cx - float(tw) / 2.0)
                y1 = max(0.0, cy - float(th) / 2.0)
                x2 = min(1.0, cx + float(tw) / 2.0)
                y2 = min(1.0, cy + float(th) / 2.0)
                boxes_list.append([x1, y1, x2, y2])
                scores_list.append(score)

        if boxes_list:
            boxes_np = np.asarray(boxes_list, dtype=np.float32)
            scores_np = np.asarray(scores_list, dtype=np.float32)
            boxes_np, scores_np = _nms_xyxy(boxes_np, scores_np, iou_thr=nms_iou_thr)
        else:
            boxes_np = np.zeros((0, 4), dtype=np.float32)
            scores_np = np.zeros((0,), dtype=np.float32)
        results.append({"boxes": boxes_np, "scores": scores_np})
    return results


def _targets_yolo_to_xyxy(targets: Sequence[torch.Tensor]) -> List[Dict[str, np.ndarray]]:
    """Chuyển tuple targets [N_i, 5] (cls, cx, cy, w, h) sang định dạng {'boxes': xyxy}."""
    gt_list: List[Dict[str, np.ndarray]] = []
    for t in targets:
        t_np = t.detach().cpu().numpy()
        if t_np.size == 0:
            gt_list.append({"boxes": np.zeros((0, 4), dtype=np.float32)})
            continue
        cx, cy, bw, bh = t_np[:, 1], t_np[:, 2], t_np[:, 3], t_np[:, 4]
        x1 = np.clip(cx - bw / 2.0, 0.0, 1.0)
        y1 = np.clip(cy - bh / 2.0, 0.0, 1.0)
        x2 = np.clip(cx + bw / 2.0, 0.0, 1.0)
        y2 = np.clip(cy + bh / 2.0, 0.0, 1.0)
        gt_list.append({"boxes": np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)})
    return gt_list


# =============================================================================
# Quy trình Sanity Check (WP-2.2) cho Stage 2 và Stage 1
# =============================================================================
def _select_balanced_stage2_indices(dataset: Stage2ROIDataset, per_class: int = 4) -> List[int]:
    """Chọn đúng `per_class` mẫu cho mỗi lớp {0, 1, 2} để tạo 1 batch cân bằng."""
    picked: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    for idx, (_, label) in enumerate(dataset.samples):
        if len(picked[label]) < per_class:
            picked[label].append(idx)
        if all(len(v) == per_class for v in picked.values()):
            break
    indices = picked[0] + picked[1] + picked[2]
    if len(indices) != 3 * per_class:
        raise ValueError(f"Không đủ {per_class} mẫu/lớp trong {dataset.root_dir}")
    return indices


def run_stage2_sanity_check(
    processed_dir: str = "data/processed",
    per_class: int = 4,
    epochs: int = 50,
    lr: float = 3e-3,
    seed: int = 42,
    device_pref: str = "auto",
) -> Dict[str, object]:
    """Chạy S2-00: Overfit 1 batch cân bằng (12 ROI thật, tắt aug/reg)."""
    set_seed(seed)
    device = get_device(device_pref)

    dataset = Stage2ROIDataset(root_dir=str(Path(processed_dir) / "train"), transform=None)
    indices = _select_balanced_stage2_indices(dataset, per_class=per_class)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=len(indices), shuffle=False, num_workers=0)

    model = TinyStage2Probe(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    initial_loss = None
    final_loss = None
    for _ in range(epochs):
        out = train_one_epoch(model, loader, criterion, optimizer, device)
        if initial_loss is None:
            initial_loss = out["loss"]
        final_loss = out["loss"]

    eval_metrics = evaluate(model, loader, criterion, device)
    num_params = sum(p.numel() for p in model.parameters())
    return {
        "run_id": "S2-00",
        "device": str(device),
        "epochs": epochs,
        "batch_size": len(indices),
        "num_params": num_params,
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "metrics": eval_metrics,
    }


def _select_stage1_indices(dataset: Stage1Dataset, num_images: int = 2, grid_size: int = 13) -> List[int]:
    """Chọn `num_images` ảnh thật có 1-4 box nằm ở các ô lưới 13x13 tách biệt."""
    chosen: List[int] = []
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        if target.shape[0] == 0 or target.shape[0] > 4:
            continue
        cells = set()
        collision = False
        for row in target:
            gx = min(int(row[1].item() * grid_size), grid_size - 1)
            gy = min(int(row[2].item() * grid_size), grid_size - 1)
            if (gx, gy) in cells:
                collision = True
                break
            cells.add((gx, gy))
        if not collision:
            chosen.append(idx)
        if len(chosen) == num_images:
            break
    if len(chosen) < num_images:
        raise ValueError(f"Không tìm đủ {num_images} ảnh phù hợp cho Stage 1 sanity check")
    return chosen


def run_stage1_sanity_check(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    num_images: int = 2,
    epochs: int = 80,
    lr: float = 3e-3,
    seed: int = 42,
    device_pref: str = "auto",
) -> Dict[str, object]:
    """Chạy S1-00: Overfit 1 batch nhỏ (2 ảnh 416x416 thật, tắt aug/reg)."""
    set_seed(seed)
    device = get_device(device_pref)

    dataset = Stage1Dataset(manifest_path=manifest_path, split="train", transform=None)
    indices = _select_stage1_indices(dataset, num_images=num_images, grid_size=13)
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=len(indices),
        shuffle=False,
        num_workers=0,
        collate_fn=stage1_collate_fn,
    )

    model = TinyStage1Probe().to(device)
    criterion = Stage1SimpleDetectionLoss(pos_weight=5.0, box_weight=5.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    initial_loss = None
    final_loss = None
    for _ in range(epochs):
        out = train_one_epoch(model, loader, criterion, optimizer, device)
        if initial_loss is None:
            initial_loss = out["loss"]
        final_loss = out["loss"]

    model.eval()
    with torch.inference_mode():
        images, targets = next(iter(loader))
        preds = model(images.to(device))
        decoded = decode_stage1_predictions(preds, score_thr=0.25, nms_iou_thr=0.5)
        gt_list = _targets_yolo_to_xyxy(targets)
        det_metrics = detection_metrics(decoded, gt_list, iou_thr=0.5, score_thr=0.25)

    num_params = sum(p.numel() for p in model.parameters())
    return {
        "run_id": "S1-00",
        "device": str(device),
        "epochs": epochs,
        "batch_size": len(indices),
        "num_params": num_params,
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "metrics": det_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SafePool Training & Sanity Check Runner (WP-2.2)")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True, help="1: Person Localizer, 2: ROI Classifier")
    parser.add_argument("--sanity-check", action="store_true", help="Chạy kiểm tra overfit 1 batch (Gate 2)")
    parser.add_argument("--epochs", type=int, default=None, help="Số epoch (mặc định: 50 cho Stage 2, 80 cho Stage 1)")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    args = parser.parse_args()

    if not args.sanity_check:
        raise NotImplementedError("Chế độ huấn luyện đầy đủ sẽ được triển khai ở Giai đoạn 3 (WP-3.1, WP-3.2). Hãy truyền --sanity-check.")

    if args.stage == 2:
        epochs = args.epochs if args.epochs is not None else 50
        res = run_stage2_sanity_check(epochs=epochs, lr=args.lr, seed=args.seed, device_pref=args.device)
        m = res["metrics"]
        print("=" * 68)
        print(f"[{res['run_id']}] STAGE 2 SANITY CHECK (device={res['device']}, batch={res['batch_size']}, epochs={res['epochs']})")
        print("=" * 68)
        print(f"  Params         : {res['num_params']:,}")
        print(f"  Loss           : {res['initial_loss']:.4f} -> {res['final_loss']:.6f}")
        print(f"  Drowning Rec   : {m['drowning_recall'] * 100:.2f}%")
        print(f"  Per-class Rec  : {[round(r * 100, 2) for r in m['recall']]}")
        print(f"  Macro F1       : {m['macro_f1'] * 100:.2f}%")
        print(f"  FAR            : {m['far'] * 100:.2f}%")
        passed = res["final_loss"] < 0.05 and m["drowning_recall"] == 1.0 and m["macro_f1"] >= 0.999 and m["far"] == 0.0
        print(f"  GATE 2 STATUS  : {'PASSED' if passed else 'FAILED'}")
        print("=" * 68)
    else:
        epochs = args.epochs if args.epochs is not None else 80
        res = run_stage1_sanity_check(epochs=epochs, lr=args.lr, seed=args.seed, device_pref=args.device)
        m = res["metrics"]
        print("=" * 68)
        print(f"[{res['run_id']}] STAGE 1 SANITY CHECK (device={res['device']}, batch={res['batch_size']}, epochs={res['epochs']})")
        print("=" * 68)
        print(f"  Params         : {res['num_params']:,}")
        print(f"  Detection Loss : {res['initial_loss']:.4f} -> {res['final_loss']:.6f}")
        print(f"  Person Recall  : {m['person_recall'] * 100:.2f}% (GT={m['num_gt']}, Pred={m['num_pred']})")
        print(f"  mAP@50         : {m['map50'] * 100:.2f}%")
        passed = res["final_loss"] < 0.10 and m["person_recall"] >= 0.95 and m["map50"] >= 0.95
        print(f"  GATE 2 STATUS  : {'PASSED' if passed else 'FAILED'}")
        print("=" * 68)


if __name__ == "__main__":
    main()
