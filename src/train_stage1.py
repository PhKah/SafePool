#!/usr/bin/env python3
"""
src/train_stage1.py
===================
WP-3.2 — Huấn luyện & đánh giá mô hình gốc Stage 1 (`D1` — Person Localizer 1-lớp `416×416`).

Tái sử dụng trực tiếp:
  - `src/datasets/stage1_dataset.py`: create_stage1_dataloader, DEFAULT_MANIFEST_PATH (WP-1.2)
  - `src/engine/trainer.py`: set_seed, get_device, move_to_device, train_one_epoch (WP-2.1)
  - `src/metrics.py`: detection_metrics (WP-2.1)
  - `src/models/stage1_baseline.py`: Stage1BaselineD1, Stage1BaselineLoss, decode_stage1_outputs
  - `src/utils/edge_profiler.py`: STAGE1_BUDGET, profile_edge_model, compute_small_box_recall, targets_yolo_to_xyxy
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.stage1_dataset import (
    DEFAULT_MANIFEST_PATH,
    Stage1Dataset,
    stage1_collate_fn,
)
from src.engine.trainer import get_device, move_to_device, set_seed, train_one_epoch
from src.metrics import detection_metrics
from src.models.stage1_baseline import (
    Stage1BaselineD1,
    Stage1BaselineLoss,
    decode_stage1_outputs,
)
from src.utils.edge_profiler import (
    STAGE1_BUDGET,
    compute_small_box_recall,
    measure_model_size_mb,
    profile_edge_model,
    targets_yolo_to_xyxy,
)


class EarlyStopping:
    """
    Cơ chế dừng sớm có pha khởi động (`warmup_epochs` — Safeguard 2):
      - Luôn ghi nhận `best_score` ngay khi có kỷ lục mới ở bất kỳ epoch nào.
      - Trong `epoch <= warmup_epochs`, KHÔNG tăng bộ đếm `bad_epochs` để tránh ngắt non
        khi hàm mất mát đang giảm nhưng `mAP@50` rời rạc còn trễ pha (IoU chưa vượt 0.5).
    """

    def __init__(
        self,
        patience: int = 10,
        warmup_epochs: int = 12,
        min_delta: float = 1e-4,
        mode: str = "max",
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError("mode phải là 'max' hoặc 'min'")
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.bad_epochs: int = 0
        self.should_stop: bool = False

    def step(self, epoch: int, score: float) -> bool:
        """Cập nhật điểm của `epoch` (1-indexed). Trả về `True` nếu lập kỷ lục mới."""
        if self.best_score is None:
            improved = True
        elif self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            return True

        if epoch > self.warmup_epochs:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True
        return False


@torch.inference_mode()
def evaluate_stage1(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[nn.Module],
    device: torch.device,
    nms_type: str = "nms",
    pre_nms_score_thr: float = 0.05,
    eval_score_thr: float = 0.25,
    nms_iou_thr: float = 0.5,
    top_k: int = 100,
    max_batches: Optional[int] = None,
) -> Dict[str, float | int | None]:
    """
    Đánh giá Stage 1 trên tập `valid` / `test`:
      - Giải mã vector hóa trên GPU/MPS + lọc `top_k` + NMS / Soft-NMS.
      - Tính `person_recall`, `map50` (`src.metrics.detection_metrics`)
        và `small_box_recall` (`area < 32×32 px`).
    """
    model.eval()
    all_preds: List[dict] = []
    all_targets: List[dict] = []
    loss_sum = 0.0
    n_samples = 0

    for step, (images, targets) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        images_dev = move_to_device(images, device)
        targets_dev = move_to_device(targets, device)

        preds = model(images_dev)
        bs = images_dev.size(0)
        if criterion is not None:
            loss_val = float(criterion(preds, targets_dev).item())
            if not math.isfinite(loss_val):
                raise FloatingPointError(f"Validation loss không hữu hạn ({loss_val}) tại step {step}")
            loss_sum += loss_val * bs

        decoded = decode_stage1_outputs(
            preds,
            score_thr=pre_nms_score_thr,
            nms_iou_thr=nms_iou_thr,
            top_k=top_k,
            nms_type=nms_type,
        )
        gt_batch = targets_yolo_to_xyxy(targets)
        all_preds.extend(decoded)
        all_targets.extend(gt_batch)
        n_samples += bs

    if n_samples == 0:
        raise ValueError("DataLoader đánh giá rỗng")

    det_m = detection_metrics(all_preds, all_targets, iou_thr=0.5, score_thr=eval_score_thr)
    small_m = compute_small_box_recall(
        all_preds,
        all_targets,
        img_size=(416, 416),
        iou_thr=0.5,
        score_thr=eval_score_thr,
    )

    return {
        "loss": (loss_sum / n_samples) if criterion is not None else None,
        "person_recall": float(det_m["person_recall"]),
        "map50": float(det_m["map50"]),
        "small_box_recall": float(small_m["small_box_recall"]),
        "regular_box_recall": float(small_m["regular_box_recall"]),
        "num_gt": int(det_m["num_gt"]),
        "num_pred": int(det_m["num_pred"]),
        "num_small_gt": int(small_m["num_small_gt"]),
        "matched_small_gt": int(small_m["matched_small_gt"]),
    }


def _default_num_workers() -> int:
    return 0 if os.name == "nt" else 4


def _build_stage1_loader(
    manifest_path: Path,
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    """Tạo DataLoader từ `Stage1Dataset` với `persistent_workers` để tránh spawn lại mỗi epoch trên macOS."""
    dataset = Stage1Dataset(
        manifest_path=manifest_path,
        split=split,
        transform=None,  # Tắt augmentation ở mốc D1 Baseline (tuyệt đối không bật VerticalFlip)
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=stage1_collate_fn,
        generator=generator,
        persistent_workers=(num_workers > 0),
    )


def run_stage1_training(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
    patience: int = 10,
    warmup_epochs: int = 12,
    num_workers: Optional[int] = None,
    seed: int = 42,
    device_pref: str = "auto",
    checkpoint_path: Path = Path("checkpoints/stage1_d1_best.pt"),
    summary_path: Optional[Path] = Path("checkpoints/stage1_d1_summary.json"),
    max_batches: Optional[int] = None,
) -> Dict[str, object]:
    """Chạy huấn luyện đầy đủ Baseline `D1` (`S1-01`) + đo Edge AI + so sánh NMS vs Soft-NMS."""
    set_seed(seed)
    device = get_device(device_pref)
    workers = _default_num_workers() if num_workers is None else num_workers

    train_loader = _build_stage1_loader(
        manifest_path=manifest_path,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        seed=seed,
    )
    valid_loader = _build_stage1_loader(
        manifest_path=manifest_path,
        split="valid",
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        seed=seed,
    )

    model = Stage1BaselineD1().to(device)
    criterion = Stage1BaselineLoss(pos_weight=15.0, box_weight=5.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=1e-5
    )
    stopper = EarlyStopping(patience=patience, warmup_epochs=warmup_epochs, min_delta=1e-4, mode="max")

    # Đo ngân sách Edge AI (bao gồm cả hậu xử lý decode + NMS) ngay từ đầu
    edge_info = profile_edge_model(
        model,
        input_shape=(1, 3, 416, 416),
        device=device,
        postprocess_fn=lambda out: decode_stage1_outputs(out, score_thr=0.25, nms_type="nms"),
        warmup=10 if max_batches is None else 3,
        runs=40 if max_batches is None else 5,
        budget=STAGE1_BUDGET,
    )

    print("=" * 88)
    print(f"[S1-01] TRAIN STAGE 1 BASELINE D1 | device={device} | bs={batch_size} | epochs={epochs} | workers={workers}")
    print(
        f"  Edge Profile: Params={edge_info['total_params']:,} ({edge_info['params_m']}M) | "
        f"FLOPs={edge_info['gflops']} GFLOPs | Size={edge_info['size_mb']} MB | "
        f"E2E Stage1={edge_info['latency_mean_ms']} ms ({edge_info['fps']} FPS) | "
        f"Within Stage1 Budget={edge_info['within_budget']}"
    )
    print("=" * 88)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, object]] = []
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        cur_lr = float(optimizer.param_groups[0]["lr"])

        train_out = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
            max_batches=max_batches,
        )
        scheduler.step()

        val_out = evaluate_stage1(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            nms_type="nms",
            max_batches=max_batches,
        )

        # Trọng số tiêu chí tổng hợp: ưu tiên mAP@50, dùng Person Recall làm tie-breaker khi mAP@50 = 0 ở epoch đầu
        score_key = float(val_out["map50"]) + 1e-3 * float(val_out["person_recall"])
        is_best = stopper.step(epoch, score_key)

        if is_best:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_out,
                    "config": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "grad_clip": grad_clip,
                        "patience": patience,
                        "warmup_epochs": warmup_epochs,
                        "seed": seed,
                    },
                },
                checkpoint_path,
            )

        ep_sec = time.perf_counter() - ep_t0
        row = {
            "epoch": epoch,
            "lr": round(cur_lr, 6),
            "train_loss": round(float(train_out["loss"]), 5),
            "val_loss": round(float(val_out["loss"]), 5) if val_out["loss"] is not None else None,
            "person_recall": round(float(val_out["person_recall"]), 4),
            "map50": round(float(val_out["map50"]), 4),
            "small_box_recall": round(float(val_out["small_box_recall"]), 4),
            "regular_box_recall": round(float(val_out["regular_box_recall"]), 4),
            "sec": round(ep_sec, 2),
            "is_best": is_best,
        }
        history.append(row)

        mark = "*BEST*" if is_best else f"wait {stopper.bad_epochs}/{patience}"
        print(
            f"Epoch {epoch:02d}/{epochs:02d} ({ep_sec:5.1f}s) | lr={cur_lr:.5f} | "
            f"tr_loss={train_out['loss']:.4f} | val_loss={val_out['loss']:.4f} | "
            f"Recall={val_out['person_recall']*100:5.2f}% | mAP@50={val_out['map50']*100:5.2f}% | "
            f"SmallRec={val_out['small_box_recall']*100:5.2f}% | {mark}"
        )

        if stopper.should_stop:
            print(f"[EarlyStopping] Dừng sớm tại epoch {epoch} (best epoch = {stopper.best_epoch}).")
            break

    total_min = (time.perf_counter() - t_start) / 60.0

    # Nạp lại trọng số tốt nhất để đối chiếu công bằng Standard NMS vs Gaussian Soft-NMS trên valid sạch
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    best_epoch = int(ckpt["epoch"])

    best_nms = evaluate_stage1(
        model=model,
        loader=valid_loader,
        criterion=criterion,
        device=device,
        nms_type="nms",
        max_batches=max_batches,
    )
    best_soft_nms = evaluate_stage1(
        model=model,
        loader=valid_loader,
        criterion=criterion,
        device=device,
        nms_type="soft_nms",
        max_batches=max_batches,
    )

    # Cập nhật dung lượng file weights thuần (model_state_dict) chuẩn xác
    pure_weights_mb = measure_model_size_mb(model)
    ckpt_file_mb = measure_model_size_mb(model, checkpoint_path=checkpoint_path)
    edge_info["size_mb"] = round(pure_weights_mb, 4)
    edge_info["checkpoint_file_mb"] = round(ckpt_file_mb, 4)

    print("=" * 88)
    print(f"[KẾT QUẢ CUỐI S1-01] Best Epoch = {best_epoch} | Tổng thời gian = {total_min:.2f} phút")
    print(
        f"  [Standard NMS] Person Recall = {best_nms['person_recall']*100:.2f}% | "
        f"mAP@50 = {best_nms['map50']*100:.2f}% | Small-box Recall (<32²) = {best_nms['small_box_recall']*100:.2f}% "
        f"({best_nms['matched_small_gt']}/{best_nms['num_small_gt']})"
    )
    print(
        f"  [Soft-NMS]     Person Recall = {best_soft_nms['person_recall']*100:.2f}% | "
        f"mAP@50 = {best_soft_nms['map50']*100:.2f}% | Small-box Recall (<32²) = {best_soft_nms['small_box_recall']*100:.2f}% "
        f"({best_soft_nms['matched_small_gt']}/{best_soft_nms['num_small_gt']})"
    )
    print("=" * 88)

    summary = {
        "run_id": "S1-01",
        "device": str(device),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "total_minutes": round(total_min, 2),
        "edge_profile": edge_info,
        "valid_nms": best_nms,
        "valid_soft_nms": best_soft_nms,
        "history": history,
    }

    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="WP-3.2: Huấn luyện Stage 1 Baseline D1 (S1-01)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/stage1_d1_best.pt"))
    parser.add_argument("--summary", type=Path, default=Path("checkpoints/stage1_d1_summary.json"))
    parser.add_argument("--max-batches", type=int, default=None, help="Giới hạn số batch/epoch khi test nhanh")
    args = parser.parse_args()

    run_stage1_training(
        manifest_path=args.manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        num_workers=args.num_workers,
        seed=args.seed,
        device_pref=args.device,
        checkpoint_path=args.checkpoint,
        summary_path=args.summary,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
