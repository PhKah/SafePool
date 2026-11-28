"""
src/utils/edge_profiler.py
==========================
Bộ đo chỉ số Edge AI (Params, FLOPs, Model Size MB, Latency/FPS) dùng chung cho
cả Stage 1 (`WP-3.2`) và Stage 2 (`WP-3.1`), kèm hàm đo Small-box Recall (`< 32×32 px`).
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from src.metrics import box_iou_xyxy, safe_div


# Trần ngân sách chính thức theo `Specs/roadmap.md`
STAGE1_BUDGET = {
    "max_params": 2_200_000,
    "max_size_mb": 9.0,
    "max_gflops": 4.5,
    "max_latency_ms": 18.0,
}

STAGE2_BUDGET = {
    "max_params": 1_300_000,
    "max_size_mb": 5.0,
    "max_gflops": 0.18,
    "max_latency_ms": 10.0,
}


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Đếm tổng số tham số và số tham số huấn luyện được."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def compute_flops(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 416, 416),
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Đo chính xác MACs và FLOPs (`FLOPs = 2 * MACs` cho Conv2d/Linear, cộng chuẩn hóa BN)
    bằng forward hooks nhẹ mà không cần cài thêm thư viện ngoài.
    Hỗ trợ cả Depthwise / Grouped Convolution cho các vòng tối ưu ở Giai đoạn 4.
    """
    was_training = model.training
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    macs_total = 0
    extra_flops = 0
    handles = []

    def conv_hook(module: nn.Conv2d, _inp, out: torch.Tensor) -> None:
        nonlocal macs_total, extra_flops
        batch_size, out_c, out_h, out_w = out.shape
        k_h, k_w = module.kernel_size
        in_per_group = module.in_channels // module.groups
        conv_macs = batch_size * out_c * out_h * out_w * in_per_group * k_h * k_w
        macs_total += conv_macs
        if module.bias is not None:
            extra_flops += batch_size * out_c * out_h * out_w

    def linear_hook(module: nn.Linear, _inp, out: torch.Tensor) -> None:
        nonlocal macs_total, extra_flops
        out_elems = out.numel()
        macs_total += out_elems * module.in_features
        if module.bias is not None:
            extra_flops += out_elems

    def bn_hook(_module: nn.BatchNorm2d, _inp, out: torch.Tensor) -> None:
        nonlocal extra_flops
        extra_flops += 2 * out.numel()

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.BatchNorm2d):
            handles.append(m.register_forward_hook(bn_hook))

    try:
        model.eval()
        dummy = torch.zeros(input_shape, dtype=torch.float32, device=device)
        with torch.no_grad():
            model(dummy)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    total_flops = 2 * macs_total + extra_flops
    return {
        "macs": float(macs_total),
        "flops": float(total_flops),
        "gmacs": float(macs_total / 1e9),
        "gflops": float(total_flops / 1e9),
    }


def measure_model_size_mb(
    model: nn.Module,
    checkpoint_path: Optional[Path | str] = None,
) -> float:
    """Đo dung lượng lưu trữ thực tế (MB) của trọng số mô hình."""
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        return float(os.path.getsize(checkpoint_path) / (1024.0 * 1024.0))
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return float(buf.tell() / (1024.0 * 1024.0))


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def benchmark_latency_fps(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 416, 416),
    device: Optional[torch.device] = None,
    warmup: int = 15,
    runs: int = 60,
    postprocess_fn: Optional[Callable[[torch.Tensor], object]] = None,
) -> Dict[str, float]:
    """
    Đo độ trễ suy luận (ms/frame) và tốc độ khung hình (FPS) có đồng bộ phần cứng CUDA/MPS.
    Nếu truyền `postprocess_fn` (ví dụ decode + NMS), thời gian đo bao gồm trọn vẹn hậu xử lý.
    """
    was_training = model.training
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    dummy = torch.rand(input_shape, dtype=torch.float32, device=device)
    timings_ms: List[float] = []

    try:
        model.eval()
        with torch.inference_mode():
            for _ in range(warmup):
                out = model(dummy)
                if postprocess_fn is not None:
                    postprocess_fn(out)
            _sync_device(device)

            for _ in range(runs):
                _sync_device(device)
                t0 = time.perf_counter()
                out = model(dummy)
                if postprocess_fn is not None:
                    postprocess_fn(out)
                _sync_device(device)
                timings_ms.append((time.perf_counter() - t0) * 1000.0)
    finally:
        model.train(was_training)

    arr = np.asarray(timings_ms, dtype=np.float64)
    mean_ms = float(arr.mean())
    p50_ms = float(np.percentile(arr, 50))
    p95_ms = float(np.percentile(arr, 95))
    fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0
    return {
        "latency_mean_ms": mean_ms,
        "latency_p50_ms": p50_ms,
        "latency_p95_ms": p95_ms,
        "fps": fps,
    }


def profile_edge_model(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 416, 416),
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[Path | str] = None,
    postprocess_fn: Optional[Callable[[torch.Tensor], object]] = None,
    warmup: int = 15,
    runs: int = 60,
    budget: Optional[Dict[str, float]] = STAGE1_BUDGET,
) -> Dict[str, object]:
    """Đo trọn gói 4 chỉ số Edge AI và kiểm tra đối chiếu với trần ngân sách."""
    p_info = count_parameters(model)
    f_info = compute_flops(model, input_shape=input_shape, device=device)
    size_mb = measure_model_size_mb(model, checkpoint_path=checkpoint_path)
    lat_info = benchmark_latency_fps(
        model,
        input_shape=input_shape,
        device=device,
        warmup=warmup,
        runs=runs,
        postprocess_fn=postprocess_fn,
    )

    within_budget = True
    if budget is not None:
        within_budget = (
            p_info["total_params"] <= budget["max_params"]
            and size_mb <= budget["max_size_mb"]
            and f_info["gflops"] <= budget["max_gflops"]
            and lat_info["latency_mean_ms"] <= budget["max_latency_ms"]
        )

    return {
        "total_params": p_info["total_params"],
        "trainable_params": p_info["trainable_params"],
        "params_m": round(p_info["total_params"] / 1e6, 4),
        "gflops": round(f_info["gflops"], 4),
        "size_mb": round(size_mb, 4),
        "latency_mean_ms": round(lat_info["latency_mean_ms"], 3),
        "latency_p50_ms": round(lat_info["latency_p50_ms"], 3),
        "latency_p95_ms": round(lat_info["latency_p95_ms"], 3),
        "fps": round(lat_info["fps"], 2),
        "within_budget": within_budget,
    }


# =============================================================================
# Thước đo chuyên biệt cho Stage 1: Chuyển đổi nhãn & Small-box Recall (<32×32 px)
# =============================================================================
def targets_yolo_to_xyxy(targets: Sequence[torch.Tensor]) -> List[Dict[str, np.ndarray]]:
    """Chuyển tuple targets `[N_i, 5]` (cls, cx, cy, w, h) sang định dạng `{'boxes': xyxy}`."""
    gt_list: List[Dict[str, np.ndarray]] = []
    for t in targets:
        t_np = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        if t_np.size == 0:
            gt_list.append({"boxes": np.zeros((0, 4), dtype=np.float32)})
            continue
        cx, cy, bw, bh = t_np[:, 1], t_np[:, 2], t_np[:, 3], t_np[:, 4]
        x1 = np.clip(cx - 0.5 * bw, 0.0, 1.0)
        y1 = np.clip(cy - 0.5 * bh, 0.0, 1.0)
        x2 = np.clip(cx + 0.5 * bw, 0.0, 1.0)
        y2 = np.clip(cy + 0.5 * bh, 0.0, 1.0)
        gt_list.append({"boxes": np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)})
    return gt_list


def compute_small_box_recall(
    preds: Sequence[dict],
    targets: Sequence[dict],
    img_size: Tuple[int, int] = (416, 416),
    small_area_px: float = 32.0 * 32.0,
    iou_thr: float = 0.5,
    score_thr: float = 0.25,
) -> Dict[str, float | int]:
    """
    Đo riêng Recall cho nhóm mục tiêu nhỏ (`area < 32×32 px` theo chuẩn COCO, chiếm ~14.5% dataset)
    và nhóm mục tiêu vừa/lớn (`area >= 32×32 px`), khớp quy tắc ghép cặp greedy của `src.metrics.detection_metrics`.
    """
    if len(preds) != len(targets):
        raise ValueError("preds và targets phải cùng số lượng ảnh")

    img_w, img_h = img_size
    num_small_gt = 0
    matched_small_gt = 0
    num_regular_gt = 0
    matched_regular_gt = 0

    for p, t in zip(preds, targets):
        gt = np.asarray(t["boxes"], dtype=np.float64).reshape(-1, 4)
        if len(gt) == 0:
            continue

        widths_px = np.clip(gt[:, 2] - gt[:, 0], 0.0, None) * img_w
        heights_px = np.clip(gt[:, 3] - gt[:, 1], 0.0, None) * img_h
        areas_px = widths_px * heights_px
        is_small = areas_px < small_area_px

        pb = np.asarray(p["boxes"], dtype=np.float64).reshape(-1, 4)
        ps = np.asarray(p["scores"], dtype=np.float64).reshape(-1)
        valid_pred = ps >= score_thr
        pb = pb[valid_pred]
        ps = ps[valid_pred]

        gt_matched = np.zeros(len(gt), dtype=bool)
        if len(pb) > 0:
            order = np.argsort(-ps, kind="stable")
            pb = pb[order]
            ious = box_iou_xyxy(pb, gt)
            for i in range(len(pb)):
                cand = np.where(gt_matched, -1.0, ious[i])
                j = int(cand.argmax())
                if cand[j] >= iou_thr:
                    gt_matched[j] = True

        num_small_gt += int(is_small.sum())
        matched_small_gt += int(gt_matched[is_small].sum())
        num_regular_gt += int((~is_small).sum())
        matched_regular_gt += int(gt_matched[~is_small].sum())

    return {
        "small_box_recall": float(safe_div(matched_small_gt, num_small_gt)),
        "regular_box_recall": float(safe_div(matched_regular_gt, num_regular_gt)),
        "num_small_gt": num_small_gt,
        "matched_small_gt": matched_small_gt,
        "num_regular_gt": num_regular_gt,
        "matched_regular_gt": matched_regular_gt,
    }


def compute_stage1_slicing_metrics(
    preds: Sequence[dict],
    targets: Sequence[dict],
    raw_classes_per_image: Optional[Sequence[Sequence[int] | np.ndarray]] = None,
    img_size: Tuple[int, int] = (416, 416),
    small_area_px: float = 32.0 * 32.0,
    crowded_min_gt: int = 3,
    iou_thr: float = 0.5,
    score_thr: float = 0.25,
) -> Dict[str, object]:
    """
    Bộ kiểm thử đa chiều (Slicing Metrics & Error Taxonomy) cho Stage 1 theo `Specs/experiment_log.md`:
      1. Tổng thể tại `score_thr`: Person Recall, Precision, F1, số hộp dự đoán TB/ảnh.
      2. Phân khúc kích thước: Small-box Recall (`< 32×32 px`) vs Regular-box Recall (`>= 32×32 px`).
      3. Phân khúc mật độ người: Crowded Recall (`>= 3` người/ảnh) vs Sparse Recall (`1–2` người/ảnh).
      4. Phân khúc theo 3 nhãn hành vi gốc (`0: Drowning`, `1: Out-of-water`, `2: Swimming`) nếu cung cấp `raw_classes_per_image`.
      5. Giải phẫu lỗi FN (`poor_loc`, `low_score`, `complete_miss`) và FP (`duplicate_cluster`, `background_noise`).
    """
    if len(preds) != len(targets):
        raise ValueError("preds và targets phải cùng số lượng ảnh")
    if raw_classes_per_image is not None and len(raw_classes_per_image) != len(targets):
        raise ValueError("raw_classes_per_image phải cùng số lượng ảnh với targets")

    img_w, img_h = img_size
    tp_total, fp_total, num_gt_total = 0, 0, 0
    num_small_gt, matched_small_gt = 0, 0
    num_regular_gt, matched_regular_gt = 0, 0
    num_crowded_gt, matched_crowded_gt = 0, 0
    num_sparse_gt, matched_sparse_gt = 0, 0

    fn_poor_loc = 0        # Có hộp dự đoán score >= score_thr đạt 0.25 <= IoU < 0.50
    fn_low_score = 0       # Có hộp dự đoán 0.05 <= score < score_thr đạt IoU >= 0.50
    fn_complete_miss = 0   # Không có hộp nào khớp vùng (IoU < 0.25)

    fp_duplicate_cluster = 0  # Hộp thừa có IoU >= 0.10 với ít nhất 1 GT (dư quanh người thật)
    fp_background_noise = 0   # Hộp thừa hoàn toàn ở vùng nền nước/thành bể (IoU < 0.10 với mọi GT)

    cls_support = {0: 0, 1: 0, 2: 0}
    cls_matched = {0: 0, 1: 0, 2: 0}

    for img_idx, (p, t) in enumerate(zip(preds, targets)):
        gt = np.asarray(t["boxes"], dtype=np.float64).reshape(-1, 4)
        pb_all = np.asarray(p["boxes"], dtype=np.float64).reshape(-1, 4)
        ps_all = np.asarray(p["scores"], dtype=np.float64).reshape(-1)

        keep_thr = ps_all >= score_thr
        pb = pb_all[keep_thr]
        ps = ps_all[keep_thr]

        if len(pb) > 0:
            order = np.argsort(-ps, kind="stable")
            pb = pb[order]
            ps = ps[order]

        n_gt = len(gt)
        num_gt_total += n_gt
        gt_matched = np.zeros(n_gt, dtype=bool)
        pred_is_tp = np.zeros(len(pb), dtype=bool)

        ious_thr = box_iou_xyxy(pb, gt) if (len(pb) > 0 and n_gt > 0) else np.zeros((len(pb), n_gt))
        for i in range(len(pb)):
            if n_gt == 0:
                break
            cand = np.where(gt_matched, -1.0, ious_thr[i])
            j = int(cand.argmax())
            if cand[j] >= iou_thr:
                gt_matched[j] = True
                pred_is_tp[i] = True

        tp_img = int(pred_is_tp.sum())
        fp_img = int(len(pb) - tp_img)
        tp_total += tp_img
        fp_total += fp_img

        # Phân loại FP: Hộp dư quanh người thật vs Báo nhầm vùng nền thuần túy
        for i in range(len(pb)):
            if pred_is_tp[i]:
                continue
            max_iou_gt = float(ious_thr[i].max()) if n_gt > 0 else 0.0
            if max_iou_gt >= 0.10:
                fp_duplicate_cluster += 1
            else:
                fp_background_noise += 1

        if n_gt == 0:
            continue

        # Kích thước & Mật độ
        widths_px = np.clip(gt[:, 2] - gt[:, 0], 0.0, None) * img_w
        heights_px = np.clip(gt[:, 3] - gt[:, 1], 0.0, None) * img_h
        is_small = (widths_px * heights_px) < small_area_px

        num_small_gt += int(is_small.sum())
        matched_small_gt += int(gt_matched[is_small].sum())
        num_regular_gt += int((~is_small).sum())
        matched_regular_gt += int(gt_matched[~is_small].sum())

        if n_gt >= crowded_min_gt:
            num_crowded_gt += n_gt
            matched_crowded_gt += int(gt_matched.sum())
        else:
            num_sparse_gt += n_gt
            matched_sparse_gt += int(gt_matched.sum())

        # Nhãn hành vi gốc 3 lớp (nếu có)
        if raw_classes_per_image is not None:
            raw_cls = np.asarray(raw_classes_per_image[img_idx], dtype=np.int64).reshape(-1)
            if len(raw_cls) != n_gt:
                raise ValueError(f"Ảnh #{img_idx}: số nhãn hành vi ({len(raw_cls)}) != số GT ({n_gt})")
            for c_id, hit in zip(raw_cls.tolist(), gt_matched.tolist()):
                if c_id in cls_support:
                    cls_support[c_id] += 1
                    if hit:
                        cls_matched[c_id] += 1

        # Phân loại FN cho từng GT bị bỏ sót
        ious_all = box_iou_xyxy(pb_all, gt) if len(pb_all) > 0 else np.zeros((0, n_gt))
        for j in range(n_gt):
            if gt_matched[j]:
                continue
            max_iou_at_thr = float(ious_thr[:, j].max()) if len(pb) > 0 else 0.0
            max_iou_any = float(ious_all[:, j].max()) if len(pb_all) > 0 else 0.0
            if max_iou_at_thr >= 0.25:
                fn_poor_loc += 1
            elif max_iou_any >= iou_thr:
                fn_low_score += 1
            else:
                fn_complete_miss += 1

    num_pred_thr = tp_total + fp_total
    fn_total = num_gt_total - tp_total
    recall = float(safe_div(tp_total, num_gt_total))
    precision = float(safe_div(tp_total, num_pred_thr))
    f1 = float(safe_div(2.0 * precision * recall, precision + recall))

    behavior_slices = None
    if raw_classes_per_image is not None:
        behavior_slices = {
            "drowning_recall": float(safe_div(cls_matched[0], cls_support[0])),
            "out_of_water_recall": float(safe_div(cls_matched[1], cls_support[1])),
            "swimming_recall": float(safe_div(cls_matched[2], cls_support[2])),
            "support": cls_support,
            "matched": cls_matched,
        }

    return {
        "person_recall": recall,
        "precision": precision,
        "f1": f1,
        "num_gt": num_gt_total,
        "num_pred_at_thr": num_pred_thr,
        "avg_pred_per_image": float(safe_div(num_pred_thr, len(preds))),
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "small_box_recall": float(safe_div(matched_small_gt, num_small_gt)),
        "regular_box_recall": float(safe_div(matched_regular_gt, num_regular_gt)),
        "num_small_gt": num_small_gt,
        "matched_small_gt": matched_small_gt,
        "crowded_recall": float(safe_div(matched_crowded_gt, num_crowded_gt)),
        "sparse_recall": float(safe_div(matched_sparse_gt, num_sparse_gt)),
        "num_crowded_gt": num_crowded_gt,
        "matched_crowded_gt": matched_crowded_gt,
        "num_sparse_gt": num_sparse_gt,
        "matched_sparse_gt": matched_sparse_gt,
        "fn_taxonomy": {
            "poor_loc_025_050": fn_poor_loc,
            "low_score_below_thr": fn_low_score,
            "complete_miss_under_025": fn_complete_miss,
        },
        "fp_taxonomy": {
            "duplicate_cluster_near_gt": fp_duplicate_cluster,
            "background_noise": fp_background_noise,
        },
        "behavior_class_recall": behavior_slices,
    }
