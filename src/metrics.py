"""Metric an toàn sinh mạng cho SafePool (WP-2.1).

Stage 2 (ROI classifier, 3 lớp): Per-class Recall/Precision/F1, Macro F1, FAR, Confusion Matrix.
Stage 1 (Person localizer, 1 lớp): Person Recall, mAP@50.

Module này chỉ phụ thuộc NumPy; tensor PyTorch được chấp nhận qua duck-typing (.detach().cpu().numpy()).
Quy ước: confusion matrix `cm[i, j]` = số mẫu có nhãn thật i bị dự đoán là j (hàng = thật, cột = dự đoán).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

CLASS_NAMES: Dict[int, str] = {0: "Drowning", 1: "Person out of water", 2: "Swimming"}
NUM_CLASSES = 3
DROWNING_CLASS = 0


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def safe_div(num, den, default: float = 0.0):
    """num/den, trả `default` (mặc định 0.0) ở những vị trí den == 0 — không sinh NaN/Inf."""
    num, den = np.broadcast_arrays(np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64))
    out = np.full(num.shape, default, dtype=np.float64)
    np.divide(num, den, out=out, where=den != 0)
    return float(out) if out.ndim == 0 else out


# ----------------------------------------------------------------------------- Stage 2
def confusion_matrix(y_true, y_pred, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Ma trận nhầm lẫn (num_classes x num_classes), hàng = nhãn thật, cột = dự đoán."""
    t = _to_numpy(y_true).astype(np.int64).ravel()
    p = _to_numpy(y_pred).astype(np.int64).ravel()
    if t.shape != p.shape:
        raise ValueError(f"y_true và y_pred khác kích thước: {t.shape} vs {p.shape}")
    if t.size and (t.min() < 0 or p.min() < 0 or t.max() >= num_classes or p.max() >= num_classes):
        raise ValueError(f"Nhãn nằm ngoài khoảng [0, {num_classes - 1}]")
    return np.bincount(t * num_classes + p, minlength=num_classes**2).reshape(num_classes, num_classes)
# .

def metrics_from_confusion_matrix(cm: np.ndarray, alarm_class: int = DROWNING_CLASS) -> dict:
    """Tính toàn bộ metric Stage 2 từ confusion matrix (cộng dồn được qua nhiều batch).

    - Lớp không có mẫu thật (support = 0): recall = 0.0, KHÔNG tính vào các macro (chỉ macro trên lớp có mặt).
    - FAR = FP(alarm_class) / (số mẫu thật KHÔNG thuộc alarm_class) = tỉ lệ báo động giả của lớp Drowning.
    """
    cm = np.asarray(cm, dtype=np.int64)
    n = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    total = float(cm.sum())

    recall = safe_div(tp, support)
    precision = safe_div(tp, predicted)
    f1 = safe_div(2 * precision * recall, precision + recall)

    present = support > 0
    macro = lambda v: float(v[present].mean()) if present.any() else 0.0  # noqa: E731

    fp_alarm = predicted[alarm_class] - tp[alarm_class]
    far = safe_div(fp_alarm, total - support[alarm_class])

    names = [CLASS_NAMES.get(i, str(i)) for i in range(n)]
    return {
        "per_class": {
            names[i]: {
                "recall": float(recall[i]),
                "precision": float(precision[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(n)
        },
        "recall": recall.tolist(),
        "precision": precision.tolist(),
        "f1": f1.tolist(),
        "support": support.astype(int).tolist(),
        "macro_recall": macro(recall),
        "macro_precision": macro(precision),
        "macro_f1": macro(f1),
        "accuracy": safe_div(tp.sum(), total),
        "drowning_recall": float(recall[DROWNING_CLASS]),
        "far": float(far),
        "confusion_matrix": cm.tolist(),
    }


def compute_classification_metrics(y_true, y_pred, num_classes: int = NUM_CLASSES) -> dict:
    return metrics_from_confusion_matrix(confusion_matrix(y_true, y_pred, num_classes))


# ----------------------------------------------------------------------------- Stage 1
def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU cặp (N,4) x (M,4), định dạng [x1, y1, x2, y2]."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(axis=1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(axis=1)
    return safe_div(inter, area_a[:, None] + area_b[None, :] - inter)


def _match_image(pred_boxes, pred_scores, gt_boxes, iou_thr):
    """Ghép greedy theo score giảm dần; trả (scores đã sắp xếp, cờ TP tương ứng)."""
    order = np.argsort(-pred_scores, kind="stable")
    pred_boxes, pred_scores = pred_boxes[order], pred_scores[order]
    is_tp = np.zeros(len(pred_scores), dtype=bool)
    if len(gt_boxes) and len(pred_boxes):
        ious = box_iou_xyxy(pred_boxes, gt_boxes)
        used = np.zeros(len(gt_boxes), dtype=bool)
        for i in range(len(pred_boxes)):
            cand = np.where(used, -1.0, ious[i])
            j = int(cand.argmax())
            if cand[j] >= iou_thr:
                used[j] = True
                is_tp[i] = True
    return pred_scores, is_tp


def detection_metrics(preds: Sequence[dict], targets: Sequence[dict], iou_thr: float = 0.5, score_thr: float = 0.25) -> dict:
    """Stage 1 (1 lớp `person`). preds[i] = {"boxes": (N,4), "scores": (N,)}; targets[i] = {"boxes": (M,4)}.

    - person_recall: tỉ lệ GT được tìm thấy (IoU >= iou_thr) trong số dự đoán có score >= score_thr.
    - map50: AP tại IoU 0.5 (nội suy all-point kiểu VOC) trên mọi dự đoán — 1 lớp nên mAP = AP.
    """
    if len(preds) != len(targets):
        raise ValueError("preds và targets phải cùng số ảnh")
    all_scores: List[np.ndarray] = []
    all_tp: List[np.ndarray] = []
    num_gt = 0
    for p, t in zip(preds, targets):
        gt = _to_numpy(t["boxes"]).reshape(-1, 4)
        pb = _to_numpy(p["boxes"]).reshape(-1, 4)
        ps = _to_numpy(p["scores"]).reshape(-1)
        num_gt += len(gt)
        s, tp = _match_image(pb, ps, gt, iou_thr)
        all_scores.append(s)
        all_tp.append(tp)

    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    tps = np.concatenate(all_tp) if all_tp else np.zeros(0, dtype=bool)
    num_pred = int(scores.size)

    person_recall = safe_div(float(tps[scores >= score_thr].sum()), num_gt)

    if num_gt == 0 or num_pred == 0:
        ap = 0.0
    else:
        order = np.argsort(-scores, kind="stable")
        tp_c = np.cumsum(tps[order].astype(np.float64))
        fp_c = np.cumsum(~tps[order])
        rec = tp_c / num_gt
        prec = safe_div(tp_c, tp_c + fp_c)
        mrec = np.concatenate([[0.0], rec, [1.0]])
        mpre = np.concatenate([[0.0], prec, [0.0]])
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())

    return {"person_recall": float(person_recall), "map50": ap, "num_gt": int(num_gt), "num_pred": num_pred}
