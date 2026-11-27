"""
src/models/stage1_baseline.py
=============================
WP-3.2 — Kiến trúc Person Localizer 1-lớp (`0: person`) tối giản From Scratch (`D1`).

Thiết kế bám sát đặc tả `PM/plan.md` & `Specs/roadmap.md`:
  1. Single-scale CNN (Conv-BN-SiLU, 0% pretrained) xuất lưới `26×26` (stride 16 trên ảnh `416×416`),
     vùng tiếp nhận (Receptive Field) ~207×207 px bao phủ >90% kích thước người bơi nhưng chỉ ~1.07M params,
     ~2.79 GFLOPs (nằm gọn dưới trần Stage 1: ≤ 2.2M params, ≤ 4.5 GFLOPs, ≤ 9.0 MB).
  2. Safeguard 1 (Chống nổ gradient): Tham số hóa hộp bao bị chặn bằng Sigmoid `S(t) ∈ (0, 1)` thay vì `exp(t)`,
     khởi tạo prior bias cho nhánh objectness `b_obj = -ln((1 - pi) / pi)` với `pi = 0.01`.
  3. Hàm mất mát `Stage1BaselineLoss`: `BCEWithLogitsLoss(pos_weight=15.0)` + `5.0 * (SmoothL1 + (1 - GIoU))`.
  4. Safeguard 3 (Chống nghẽn cổ chai Eval O(M^2)): Giải mã lưới 100% vector hóa bằng broadcasting trên GPU/MPS,
     lọc ngưỡng tiên nghiệm + giới hạn `top_k = 100` ứng viên/ảnh trước khi chạy Standard NMS (`torchvision.ops.nms`)
     hoặc Gaussian Soft-NMS.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import nms as tv_nms


class ConvBNSiLU(nn.Module):
    """Khối tích chập cơ sở From Scratch: Conv2d (no bias) -> BatchNorm2d -> SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Stage1BaselineD1(nn.Module):
    """
    Baseline Person Localizer `D1` (1-class `person`, single-scale `26×26` tại input `416×416`).
    Đầu ra: Tensor `[B, 5, H/16, W/16]` tương ứng `[obj_logit, raw_tx, raw_ty, raw_tw, raw_th]`.
    """

    stride: int = 16

    def __init__(self, in_channels: int = 3, prior_prob: float = 0.01) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBNSiLU(in_channels, 32, kernel_size=3, stride=2, padding=1),   # 416 -> 208
            ConvBNSiLU(32, 64, kernel_size=3, stride=2, padding=1),            # 208 -> 104
            ConvBNSiLU(64, 128, kernel_size=3, stride=2, padding=1),           # 104 -> 52
            ConvBNSiLU(128, 128, kernel_size=3, stride=1, padding=1),          # 52 -> 52
            ConvBNSiLU(128, 160, kernel_size=3, stride=2, padding=1),          # 52 -> 26
            ConvBNSiLU(160, 160, kernel_size=3, stride=1, padding=1),          # 26 -> 26
            ConvBNSiLU(160, 160, kernel_size=3, stride=1, padding=2, dilation=2),  # mở rộng RF
        )
        self.head_refine = ConvBNSiLU(160, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.head_out = nn.Conv2d(128, 5, kernel_size=1, stride=1, padding=0, bias=True)

        self._init_weights(prior_prob=prior_prob)

    def _init_weights(self, prior_prob: float = 0.01) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Khởi tạo head đầu ra an toàn chống shock loss & nổ gradient ở epoch đầu
        nn.init.normal_(self.head_out.weight, mean=0.0, std=0.01)
        assert self.head_out.bias is not None
        with torch.no_grad():
            bias_obj = -math.log((1.0 - prior_prob) / prior_prob)
            self.head_out.bias[0].fill_(bias_obj)   # sigmoid ~ 0.01 ở đầu quá trình học
            self.head_out.bias[1:3].fill_(0.0)      # tx, ty ~ 0.5 (tâm ô lưới)
            self.head_out.bias[3:5].fill_(-1.4)     # tw, th ~ 0.20 (~83 px trên ảnh 416x416)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.head_refine(feat)
        return self.head_out(feat)


def generalized_box_iou_paired(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Tính GIoU từng cặp 1-1 giữa `boxes1` [N, 4] và `boxes2` [N, 4] theo chuẩn [x1, y1, x2, y2].
    Độ phức tạp O(N), hoàn toàn khả vi và bị chặn trong (-1, 1].
    """
    ix1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    iy1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    ix2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    iy2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])

    inter_w = (ix2 - ix1).clamp(min=0.0)
    inter_h = (iy2 - iy1).clamp(min=0.0)
    inter = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0.0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0.0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0.0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0.0)
    union = area1 + area2 - inter + eps
    iou = inter / union

    cx1 = torch.minimum(boxes1[:, 0], boxes2[:, 0])
    cy1 = torch.minimum(boxes1[:, 1], boxes2[:, 1])
    cx2 = torch.maximum(boxes1[:, 2], boxes2[:, 2])
    cy2 = torch.maximum(boxes1[:, 3], boxes2[:, 3])
    area_c = (cx2 - cx1).clamp(min=0.0) * (cy2 - cy1).clamp(min=0.0) + eps

    return iou - (area_c - union) / area_c


class Stage1BaselineLoss(nn.Module):
    """
    Hàm mất mát cho `Stage1BaselineD1`:
      Loss = BCEWithLogitsLoss(pos_weight=15.0) + box_weight * (SmoothL1(tx,ty,tw,th) + (1 - GIoU))
    Sử dụng Bounded Sigmoid Mapping S(t) ∈ (0, 1) để đảm bảo ổn định số học tuyệt đối.
    """

    def __init__(
        self,
        pos_weight: float = 15.0,
        box_weight: float = 5.0,
        min_wh: float = 1e-4,
    ) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.box_weight = box_weight
        self.min_wh = min_wh

    def forward(self, preds: torch.Tensor, targets: Sequence[torch.Tensor]) -> torch.Tensor:
        bs, _, grid_h, grid_w = preds.shape
        device = preds.device
        dtype = preds.dtype

        obj_target_cpu = torch.zeros((bs, grid_h, grid_w), dtype=torch.float32)
        box_target_cpu = torch.zeros((bs, grid_h, grid_w, 4), dtype=torch.float32)
        gt_xyxy_cpu = torch.zeros((bs, grid_h, grid_w, 4), dtype=torch.float32)
        pos_mask_cpu = torch.zeros((bs, grid_h, grid_w), dtype=torch.bool)
        best_dist_sq = torch.full((bs, grid_h, grid_w), float("inf"), dtype=torch.float32)

        for b_idx, t in enumerate(targets):
            if t.numel() == 0:
                continue
            t_cpu = t.detach().to(device="cpu", dtype=torch.float32)
            for _, raw_cx, raw_cy, raw_bw, raw_bh in t_cpu.tolist():
                cx = min(max(raw_cx, 0.0), 1.0 - 1e-6)
                cy = min(max(raw_cy, 0.0), 1.0 - 1e-6)
                bw = min(max(raw_bw, self.min_wh), 1.0)
                bh = min(max(raw_bh, self.min_wh), 1.0)

                gx_f = cx * grid_w
                gy_f = cy * grid_h
                gx = int(gx_f)
                gy = int(gy_f)
                tx = gx_f - gx
                ty = gy_f - gy

                # Nếu 2 tâm người cùng rơi vào 1 ô 16x16, giữ mục tiêu có tâm gần tâm ô hơn
                dist_sq = (tx - 0.5) ** 2 + (ty - 0.5) ** 2
                if pos_mask_cpu[b_idx, gy, gx] and dist_sq >= float(best_dist_sq[b_idx, gy, gx]):
                    continue

                best_dist_sq[b_idx, gy, gx] = dist_sq
                pos_mask_cpu[b_idx, gy, gx] = True
                obj_target_cpu[b_idx, gy, gx] = 1.0
                box_target_cpu[b_idx, gy, gx, 0] = tx
                box_target_cpu[b_idx, gy, gx, 1] = ty
                box_target_cpu[b_idx, gy, gx, 2] = bw
                box_target_cpu[b_idx, gy, gx, 3] = bh

                gt_xyxy_cpu[b_idx, gy, gx, 0] = cx - 0.5 * bw
                gt_xyxy_cpu[b_idx, gy, gx, 1] = cy - 0.5 * bh
                gt_xyxy_cpu[b_idx, gy, gx, 2] = cx + 0.5 * bw
                gt_xyxy_cpu[b_idx, gy, gx, 3] = cy + 0.5 * bh

        obj_target = obj_target_cpu.to(device=device, dtype=dtype)
        pos_w = torch.tensor(self.pos_weight, dtype=dtype, device=device)
        obj_loss = F.binary_cross_entropy_with_logits(preds[:, 0], obj_target, pos_weight=pos_w)

        b_idx_cpu, gy_cpu, gx_cpu = torch.where(pos_mask_cpu)
        if b_idx_cpu.numel() > 0:
            # Bounded Sigmoid Mapping: đạo hàm bị chặn <= 0.25, không bao giờ tràn số như exp(t)
            box_sig = torch.sigmoid(preds[:, 1:5]).permute(0, 2, 3, 1)  # [B, H, W, 4]
            b_idx_dev = b_idx_cpu.to(device=device)
            gy_idx = gy_cpu.to(device=device)
            gx_idx = gx_cpu.to(device=device)

            pred_pos = box_sig[b_idx_dev, gy_idx, gx_idx]
            target_pos = box_target_cpu[b_idx_cpu, gy_cpu, gx_cpu].to(device=device, dtype=dtype)

            smooth_l1 = F.smooth_l1_loss(pred_pos, target_pos, beta=0.1, reduction="mean")

            # Dựng tọa độ xyxy dự đoán tại đúng các ô dương để tính GIoU Loss
            pred_cx = (gx_idx.to(dtype) + pred_pos[:, 0]) / grid_w
            pred_cy = (gy_idx.to(dtype) + pred_pos[:, 1]) / grid_h
            pred_bw = pred_pos[:, 2].clamp(min=self.min_wh)
            pred_bh = pred_pos[:, 3].clamp(min=self.min_wh)

            pred_xyxy = torch.stack(
                [
                    pred_cx - 0.5 * pred_bw,
                    pred_cy - 0.5 * pred_bh,
                    pred_cx + 0.5 * pred_bw,
                    pred_cy + 0.5 * pred_bh,
                ],
                dim=1,
            )
            gt_xyxy = gt_xyxy_cpu[b_idx_cpu, gy_cpu, gx_cpu].to(device=device, dtype=dtype)
            giou = generalized_box_iou_paired(pred_xyxy, gt_xyxy, eps=1e-6)
            giou_loss = (1.0 - giou).mean()

            reg_loss = smooth_l1 + giou_loss
        else:
            reg_loss = preds.sum() * 0.0

        return obj_loss + self.box_weight * reg_loss


# =============================================================================
# Hậu xử lý Vector hóa: Pre-filtering + Top-K + Standard NMS / Gaussian Soft-NMS
# =============================================================================
def nms_xyxy(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thr: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard NMS dùng `torchvision.ops.nms` trên tensor CPU đã lọc top-K."""
    if boxes.numel() == 0:
        return boxes, scores
    keep = tv_nms(boxes, scores, iou_thr)
    return boxes[keep], scores[keep]


def soft_nms_xyxy(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    sigma: float = 0.5,
    score_thr: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gaussian Soft-NMS vector hóa trên tập ứng viên đã lọc `top_k`:
      s_j <- s_j * exp(-IoU(b_i, b_j)^2 / sigma)
    Giúp giữ lại người bơi cạnh nhau trong các khung hình đông người.
    """
    if boxes.numel() == 0:
        return boxes, scores

    boxes = boxes.clone()
    scores = scores.clone()
    n = boxes.shape[0]
    kept_boxes: List[torch.Tensor] = []
    kept_scores: List[torch.Tensor] = []

    active = torch.ones(n, dtype=torch.bool, device=boxes.device)
    while active.any():
        active_indices = torch.where(active)[0]
        rel_max = int(torch.argmax(scores[active_indices]).item())
        best_idx = int(active_indices[rel_max].item())
        active[best_idx] = False

        best_score = scores[best_idx]
        if float(best_score.item()) < score_thr:
            break

        best_box = boxes[best_idx]
        kept_boxes.append(best_box)
        kept_scores.append(best_score)

        rem_indices = torch.where(active)[0]
        if rem_indices.numel() == 0:
            break

        rem_boxes = boxes[rem_indices]
        ix1 = torch.maximum(best_box[0], rem_boxes[:, 0])
        iy1 = torch.maximum(best_box[1], rem_boxes[:, 1])
        ix2 = torch.minimum(best_box[2], rem_boxes[:, 2])
        iy2 = torch.minimum(best_box[3], rem_boxes[:, 3])
        inter = (ix2 - ix1).clamp(min=0.0) * (iy2 - iy1).clamp(min=0.0)

        area_best = (best_box[2] - best_box[0]).clamp(min=0.0) * (best_box[3] - best_box[1]).clamp(min=0.0)
        area_rem = (rem_boxes[:, 2] - rem_boxes[:, 0]).clamp(min=0.0) * (rem_boxes[:, 3] - rem_boxes[:, 1]).clamp(min=0.0)
        iou = inter / (area_best + area_rem - inter + 1e-6)

        decay = torch.exp(-(iou * iou) / sigma)
        scores[rem_indices] = scores[rem_indices] * decay
        drop = scores[rem_indices] < score_thr
        active[rem_indices[drop]] = False

    if not kept_boxes:
        return torch.zeros((0, 4), dtype=boxes.dtype), torch.zeros((0,), dtype=scores.dtype)
    out_boxes = torch.stack(kept_boxes, dim=0)
    out_scores = torch.stack(kept_scores, dim=0)
    order = torch.argsort(out_scores, descending=True)
    return out_boxes[order], out_scores[order]


@torch.no_grad()
def decode_stage1_outputs(
    preds: torch.Tensor,
    score_thr: float = 0.05,
    nms_iou_thr: float = 0.5,
    top_k: int = 100,
    max_detections: int = 30,
    nms_type: str = "nms",
    soft_nms_sigma: float = 0.5,
) -> List[Dict[str, np.ndarray]]:
    """
    Giải mã 100% vector hóa tensor `[B, 5, H, W]` trên thiết bị song song (GPU/MPS),
    lọc ngưỡng tiên nghiệm + giới hạn `top_k` rồi chạy NMS / Soft-NMS.

    Trả về danh sách `B` phần tử `{"boxes": np.ndarray (K, 4), "scores": np.ndarray (K,)}`
    theo chuẩn tọa độ chuẩn hóa `[x1, y1, x2, y2] ∈ [0, 1]` khớp trực tiếp `src.metrics.detection_metrics`.
    """
    if nms_type not in {"nms", "soft_nms"}:
        raise ValueError(f"nms_type không hợp lệ: {nms_type}. Chọn 'nms' hoặc 'soft_nms'.")

    bs, _, grid_h, grid_w = preds.shape
    device = preds.device
    dtype = preds.dtype

    # 1. Giải mã toàn bộ lưới bằng broadcasting song song trên GPU/MPS (0 vòng lặp theo ô lưới)
    scores_all = torch.sigmoid(preds[:, 0])        # [B, H, W]
    box_sig = torch.sigmoid(preds[:, 1:5])         # [B, 4, H, W]

    gy, gx = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=dtype),
        torch.arange(grid_w, device=device, dtype=dtype),
        indexing="ij",
    )
    cx = (gx.unsqueeze(0) + box_sig[:, 0]) / grid_w
    cy = (gy.unsqueeze(0) + box_sig[:, 1]) / grid_h
    bw = box_sig[:, 2]
    bh = box_sig[:, 3]

    x1 = (cx - 0.5 * bw).clamp(0.0, 1.0)
    y1 = (cy - 0.5 * bh).clamp(0.0, 1.0)
    x2 = (cx + 0.5 * bw).clamp(0.0, 1.0)
    y2 = (cy + 0.5 * bh).clamp(0.0, 1.0)

    boxes_flat = torch.stack([x1, y1, x2, y2], dim=-1).reshape(bs, -1, 4)  # [B, H*W, 4]
    scores_flat = scores_all.reshape(bs, -1)                               # [B, H*W]

    # 2. Batched Top-K ngay trên GPU/MPS (1 phép toán SIMD cho cả batch, chỉ truyền <= B*top_k phần tử về CPU)
    num_cells = scores_flat.shape[1]
    k_cap = min(top_k, num_cells)
    scores_top, top_idx = torch.topk(scores_flat, k=k_cap, dim=1, largest=True, sorted=True)  # [B, k_cap]
    gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, 4)
    boxes_top = torch.gather(boxes_flat, dim=1, index=gather_idx)                             # [B, k_cap, 4]

    scores_top_cpu = scores_top.detach().to(device="cpu", dtype=torch.float32)
    boxes_top_cpu = boxes_top.detach().to(device="cpu", dtype=torch.float32)

    results: List[Dict[str, np.ndarray]] = []
    for b_idx in range(bs):
        sc_cpu = scores_top_cpu[b_idx]
        bx_cpu = boxes_top_cpu[b_idx]

        # 3. Heuristic Pre-filtering sau batched Top-K (tương đương toán học 100%, 0 đồng bộ GPU/MPS lẻ)
        keep_mask = sc_cpu >= score_thr
        if not keep_mask.any():
            results.append(
                {
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "scores": np.zeros((0,), dtype=np.float32),
                }
            )
            continue

        sc_cpu = sc_cpu[keep_mask]
        bx_cpu = bx_cpu[keep_mask]

        if nms_type == "nms":
            bx_out, sc_out = nms_xyxy(bx_cpu, sc_cpu, iou_thr=nms_iou_thr)
        else:
            bx_out, sc_out = soft_nms_xyxy(bx_cpu, sc_cpu, sigma=soft_nms_sigma, score_thr=score_thr)

        if bx_out.shape[0] > max_detections:
            bx_out = bx_out[:max_detections]
            sc_out = sc_out[:max_detections]

        results.append(
            {
                "boxes": bx_out.numpy(),
                "scores": sc_out.numpy(),
            }
        )
    return results
