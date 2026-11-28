#!/usr/bin/env python3
"""
scripts/audit_stage1_baseline.py
================================
Kiểm thử toàn diện & Phân tích lỗi đa chiều (Slicing Metrics + Visual Error Audit + Hard Edge Gate)
cho mô hình Baseline Stage 1 (`D1` — Run `S1-01`) tuân thủ nghiêm ngặt `Specs/experiment_log.md`:

  1. Khóa cứng đánh giá trên tập `valid` của `clean Group Split` (`seed=42`, tuyệt đối không đụng `test`).
  2. Đo đầy đủ Tổng thể + Phân khúc hiểm (Slicing Metrics):
     - Person Recall (@0.25), Precision (@0.25), F1 (@0.25), mAP@50
     - Small-box Recall (< 32×32 px) vs Regular-box Recall (>= 32×32 px)
     - Crowded Recall (>= 3 người/ảnh) vs Sparse Recall (1–2 người/ảnh)
     - Recall phân tách theo 3 lớp hành vi gốc (`0: Drowning`, `1: Person out of water`, `2: Swimming`)
     - Giải phẫu toán học FN (`poor_loc`, `low_score`, `complete_miss`) và FP (`duplicate_cluster`, `background_noise`)
  3. Kiểm tra Cổng Loại Trừ Phần Cứng (`STAGE1_BUDGET`).
  4. Trích xuất & vẽ lưới trực quan 30 ca lỗi điển hình nhất (15 ca bỏ sót `0: Drowning` tệ nhất +
     15 ca báo giả nền nước/bọt sóng điểm cao nhất) vào thư mục git-ignored `checkpoints/audit_s1_01/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.stage1_dataset import DEFAULT_MANIFEST_PATH, Stage1Dataset, stage1_collate_fn
from src.engine.trainer import get_device, move_to_device, set_seed
from src.metrics import CLASS_NAMES, box_iou_xyxy, detection_metrics
from src.models.stage1_baseline import Stage1BaselineD1, decode_stage1_outputs
from src.utils.edge_profiler import (
    STAGE1_BUDGET,
    compute_stage1_slicing_metrics,
    measure_model_size_mb,
    profile_edge_model,
    targets_yolo_to_xyxy,
)


def load_raw_behavior_classes_for_manifest(dataset: Stage1Dataset) -> List[List[int]]:
    """
    Đối chiếu từng ảnh trong `Stage1Dataset` ngược về file nhãn gốc `data/raw/*/labels/<stem>.txt`
    để lấy nhãn hành vi thật (`0: Drowning`, `1: Person out of water`, `2: Swimming`) của từng GT box.
    """
    all_raw_classes: List[List[int]] = []
    for entry in dataset.entries:
        img_path = dataset.resolve_project_path(entry["image_path"])
        raw_label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if not raw_label_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy nhãn gốc 3 lớp tại: {raw_label_path}")

        cls_ids: List[int] = []
        with raw_label_path.open("r", encoding="utf-8") as f:
            for line in f:
                toks = line.strip().split()
                if toks:
                    cls_ids.append(int(toks[0]))
        all_raw_classes.append(cls_ids)
    return all_raw_classes


def collect_worst_error_samples(
    dataset: Stage1Dataset,
    preds: List[dict],
    targets: List[dict],
    raw_classes: List[List[int]],
    score_thr: float = 0.25,
    top_n: int = 15,
) -> Tuple[List[dict], List[dict]]:
    """
    Lọc 2 nhóm mẫu lỗi nghiêm trọng nhất phục vụ Bước 1 (Error Analysis 20–30 ca sai) trong `experiment_log.md`:
      - `worst_fn_drowning`: Top `top_n` ca bỏ sót nạn nhân `0: Drowning` (ưu tiên hoàn toàn không thấy IoU thấp nhất).
      - `worst_fp_background`: Top `top_n` ca báo giả vùng nền (`IoU < 0.10` với mọi GT) nhưng mạng tự tin cao nhất.
    """
    drowning_misses: List[dict] = []
    bg_false_positives: List[dict] = []

    for img_idx, (p, t, cls_list) in enumerate(zip(preds, targets, raw_classes)):
        gt = np.asarray(t["boxes"], dtype=np.float64).reshape(-1, 4)
        pb_all = np.asarray(p["boxes"], dtype=np.float64).reshape(-1, 4)
        ps_all = np.asarray(p["scores"], dtype=np.float64).reshape(-1)

        keep = ps_all >= score_thr
        pb = pb_all[keep]
        ps = ps_all[keep]
        if len(pb) > 0:
            order = np.argsort(-ps, kind="stable")
            pb, ps = pb[order], ps[order]

        gt_matched = np.zeros(len(gt), dtype=bool)
        pred_is_tp = np.zeros(len(pb), dtype=bool)
        ious = box_iou_xyxy(pb, gt) if (len(pb) > 0 and len(gt) > 0) else np.zeros((len(pb), len(gt)))

        for i in range(len(pb)):
            if len(gt) == 0:
                break
            cand = np.where(gt_matched, -1.0, ious[i])
            j = int(cand.argmax())
            if cand[j] >= 0.50:
                gt_matched[j] = True
                pred_is_tp[i] = True

        # 1. Thu thập ca bỏ sót nhãn thật 0: Drowning tệ nhất của mỗi bức ảnh (đảm bảo 15 mẫu thuộc 15 khung cảnh độc lập)
        ious_all = box_iou_xyxy(pb_all, gt) if (len(pb_all) > 0 and len(gt) > 0) else np.zeros((len(pb_all), len(gt)))
        worst_fn_in_img = None
        for j, (is_hit, c_id) in enumerate(zip(gt_matched.tolist(), cls_list)):
            if is_hit or c_id != 0:
                continue
            max_iou_thr = float(ious[:, j].max()) if len(pb) > 0 else 0.0
            max_iou_any = float(ious_all[:, j].max()) if len(pb_all) > 0 else 0.0
            best_sc_any = float(ps_all[ious_all[:, j].argmax()]) if len(pb_all) > 0 else 0.0
            w_px = float((gt[j, 2] - gt[j, 0]) * 416.0)
            h_px = float((gt[j, 3] - gt[j, 1]) * 416.0)

            if max_iou_thr >= 0.25:
                reason = "poor_loc (0.25<=IoU<0.5)"
            elif max_iou_any >= 0.50:
                reason = "low_score (<0.25)"
            else:
                reason = "complete_miss (IoU<0.25)"

            cand_fn = {
                "valid_idx": img_idx,
                "image_name": Path(dataset.entries[img_idx]["image_path"]).name,
                "gt_box": gt[j].tolist(),
                "box_wh_px": [round(w_px, 1), round(h_px, 1)],
                "max_iou_at_thr": round(max_iou_thr, 3),
                "max_iou_any": round(max_iou_any, 3),
                "best_overlap_score": round(best_sc_any, 3),
                "reason": reason,
                "num_gt_in_image": len(gt),
            }
            if worst_fn_in_img is None or (cand_fn["max_iou_at_thr"], cand_fn["max_iou_any"]) < (
                worst_fn_in_img["max_iou_at_thr"],
                worst_fn_in_img["max_iou_any"],
            ):
                worst_fn_in_img = cand_fn
        if worst_fn_in_img is not None:
            drowning_misses.append(worst_fn_in_img)

        # 2. Thu thập các ca báo giả vùng nền nước / thành bể có score cao nhất (lấy tối đa 1 hộp tệ nhất/ảnh để đa dạng góc quay)
        best_bg_fp = None
        for i in range(len(pb)):
            if pred_is_tp[i]:
                continue
            max_iou_gt = float(ious[i].max()) if len(gt) > 0 else 0.0
            if max_iou_gt < 0.10:
                cand_fp = {
                    "valid_idx": img_idx,
                    "image_name": Path(dataset.entries[img_idx]["image_path"]).name,
                    "fp_box": pb[i].tolist(),
                    "fp_score": round(float(ps[i]), 4),
                    "max_iou_with_any_gt": round(max_iou_gt, 3),
                    "num_gt_in_image": len(gt),
                }
                if best_bg_fp is None or cand_fp["fp_score"] > best_bg_fp["fp_score"]:
                    best_bg_fp = cand_fp
        if best_bg_fp is not None:
            bg_false_positives.append(best_bg_fp)

    # Sắp xếp: FN ưu tiên complete_miss (max_iou_at_thr nhỏ nhất), FP ưu tiên fp_score lớn nhất
    drowning_misses.sort(key=lambda x: (x["max_iou_at_thr"], x["max_iou_any"]))
    bg_false_positives.sort(key=lambda x: -x["fp_score"])
    return drowning_misses[:top_n], bg_false_positives[:top_n]


def render_error_grid(
    dataset: Stage1Dataset,
    preds: List[dict],
    targets: List[dict],
    cases: List[dict],
    mode: str,
    out_path: Path,
    score_thr: float = 0.25,
) -> None:
    """Vẽ bảng 3×5 (15 mẫu) cho nhóm lỗi FN hoặc FP để kiểm tra trực quan bằng mắt."""
    if not cases:
        return
    cols = 5
    rows = int(np.ceil(len(cases) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.3 * rows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax_idx, ax in enumerate(axes_flat):
        if ax_idx >= len(cases):
            ax.axis("off")
            continue
        item = cases[ax_idx]
        v_idx = item["valid_idx"]
        img_t, _ = dataset[v_idx]
        img_np = img_t.permute(1, 2, 0).numpy()
        gt_all = targets[v_idx]["boxes"]
        p_boxes = preds[v_idx]["boxes"]
        p_scores = preds[v_idx]["scores"]
        keep = p_scores >= score_thr
        p_boxes, p_scores = p_boxes[keep], p_scores[keep]

        ax.imshow(img_np)
        # Vẽ tất cả GT bằng nét xanh lá
        for x1, y1, x2, y2 in gt_all:
            ax.add_patch(
                patches.Rectangle(
                    (x1 * 416, y1 * 416), (x2 - x1) * 416, (y2 - y1) * 416,
                    lw=1.8, edgecolor="lime", facecolor="none"
                )
            )
        # Vẽ dự đoán bằng nét đứt đỏ
        for (x1, y1, x2, y2), sc in zip(p_boxes, p_scores):
            ax.add_patch(
                patches.Rectangle(
                    (x1 * 416, y1 * 416), (x2 - x1) * 416, (y2 - y1) * 416,
                    lw=1.5, edgecolor="red", linestyle="--", facecolor="none"
                )
            )

        if mode == "fn":
            gx1, gy1, gx2, gy2 = item["gt_box"]
            ax.add_patch(
                patches.Rectangle(
                    (gx1 * 416, gy1 * 416), (gx2 - gx1) * 416, (gy2 - gy1) * 416,
                    lw=2.8, edgecolor="yellow", facecolor="none"
                )
            )
            w_p, h_p = item["box_wh_px"]
            ax.set_title(
                f"valid[{v_idx}] | {w_p:.0f}×{h_p:.0f}px\nIoU={item['max_iou_at_thr']:.2f} ({item['reason'].split()[0]})",
                fontsize=9, fontweight="bold",
            )
        else:
            fx1, fy1, fx2, fy2 = item["fp_box"]
            ax.add_patch(
                patches.Rectangle(
                    (fx1 * 416, fy1 * 416), (fx2 - fx1) * 416, (fy2 - fy1) * 416,
                    lw=2.8, edgecolor="cyan", facecolor="none"
                )
            )
            ax.set_title(
                f"valid[{v_idx}] | FP score={item['fp_score']:.2f}\nIoU_GT={item['max_iou_with_any_gt']:.2f} (GT={item['num_gt_in_image']})",
                fontsize=9, fontweight="bold",
            )

        ax.set_xticks([])
        ax.set_yticks([])

    title = (
        "15 Ca Bỏ Sót Nạn Nhân 0: Drowning Nghiêm Trọng Nhất (Vàng = Drowning Bị Sót, Xanh = GT, Đỏ = Pred)"
        if mode == "fn"
        else "15 Ca Báo Giả Vùng Nền Có Điểm Tự Tin Cao Nhất (Xanh Lam = Hộp Báo Giả Nền, Xanh Lá = GT)"
    )
    plt.suptitle(title, fontsize=13, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_stage1_audit(
    checkpoint_path: Path = Path("checkpoints/stage1_d1_best.pt"),
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    out_dir: Path = Path("checkpoints/audit_s1_01"),
    score_thr: float = 0.25,
    seed: int = 42,
    device_pref: str = "auto",
) -> Dict[str, object]:
    set_seed(seed)
    device = get_device(device_pref)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Stage1BaselineD1().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    valid_ds = Stage1Dataset(manifest_path=manifest_path, split="valid")
    raw_classes = load_raw_behavior_classes_for_manifest(valid_ds)
    loader = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=stage1_collate_fn)

    raw_outputs_list: List[torch.Tensor] = []
    gt_xyxy_list: List[dict] = []
    with torch.inference_mode():
        for imgs, tgts in loader:
            preds_raw = model(move_to_device(imgs, device))
            raw_outputs_list.append(preds_raw.cpu())
            gt_xyxy_list.extend(targets_yolo_to_xyxy(tgts))
    raw_all = torch.cat(raw_outputs_list, dim=0)

    # Giải mã với Standard NMS và Gaussian Soft-NMS
    preds_nms = decode_stage1_outputs(raw_all, score_thr=0.05, nms_iou_thr=0.5, top_k=100, max_detections=30, nms_type="nms")
    preds_soft = decode_stage1_outputs(raw_all, score_thr=0.05, soft_nms_sigma=0.5, top_k=100, max_detections=30, nms_type="soft_nms")

    det_nms = detection_metrics(preds_nms, gt_xyxy_list, iou_thr=0.5, score_thr=score_thr)
    det_soft = detection_metrics(preds_soft, gt_xyxy_list, iou_thr=0.5, score_thr=score_thr)

    slicing_nms = compute_stage1_slicing_metrics(preds_nms, gt_xyxy_list, raw_classes_per_image=raw_classes, score_thr=score_thr)
    slicing_soft = compute_stage1_slicing_metrics(preds_soft, gt_xyxy_list, raw_classes_per_image=raw_classes, score_thr=score_thr)

    edge_prof = profile_edge_model(
        model,
        input_shape=(1, 3, 416, 416),
        device=device,
        postprocess_fn=lambda out: decode_stage1_outputs(out, score_thr=score_thr, nms_type="nms"),
        warmup=10,
        runs=40,
        budget=STAGE1_BUDGET,
    )
    edge_prof["size_mb"] = round(measure_model_size_mb(model), 4)

    worst_fn, worst_fp = collect_worst_error_samples(
        valid_ds, preds_nms, gt_xyxy_list, raw_classes, score_thr=score_thr, top_n=15
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fn_grid_path = out_dir / "worst_15_drowning_fn.png"
    fp_grid_path = out_dir / "worst_15_background_fp.png"
    render_error_grid(valid_ds, preds_nms, gt_xyxy_list, worst_fn, mode="fn", out_path=fn_grid_path, score_thr=score_thr)
    render_error_grid(valid_ds, preds_nms, gt_xyxy_list, worst_fp, mode="fp", out_path=fp_grid_path, score_thr=score_thr)

    report = {
        "run_id": "S1-01",
        "split": "valid (clean Group Split — 0% test touched)",
        "seed": seed,
        "best_epoch": int(ckpt["epoch"]),
        "score_thr": score_thr,
        "edge_profile": edge_prof,
        "stage1_budget_ceiling": STAGE1_BUDGET,
        "standard_nms": {
            "map50": round(float(det_nms["map50"]), 4),
            **slicing_nms,
        },
        "gaussian_soft_nms": {
            "map50": round(float(det_soft["map50"]), 4),
            **slicing_soft,
        },
        "worst_15_drowning_fn": worst_fn,
        "worst_15_background_fp": worst_fp,
        "artifacts": {
            "fn_grid": str(fn_grid_path),
            "fp_grid": str(fp_grid_path),
        },
    }

    report_path = out_dir / "s1_01_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    beh = slicing_nms["behavior_class_recall"]
    fn_tax = slicing_nms["fn_taxonomy"]
    fp_tax = slicing_nms["fp_taxonomy"]

    print("=" * 90)
    print(f"[AUDIT S1-01 | Stage1BaselineD1] split=valid ({len(valid_ds):,} imgs, {slicing_nms['num_gt']:,} GT) | Best Epoch={ckpt['epoch']}")
    print("=" * 90)
    print(f"  1. TỔNG THỂ (@score_thr={score_thr}):")
    print(f"     - Person Recall         : {slicing_nms['person_recall']*100:6.2f}% ({slicing_nms['tp']}/{slicing_nms['num_gt']})")
    print(f"     - Precision             : {slicing_nms['precision']*100:6.2f}% ({slicing_nms['tp']}/{slicing_nms['num_pred_at_thr']} boxes, TB {slicing_nms['avg_pred_per_image']:.2f} box/ảnh)")
    print(f"     - F1-Score (@0.25)      : {slicing_nms['f1']*100:6.2f}% | mAP@50 (Full PR): {det_nms['map50']*100:6.2f}%")
    print(f"  2. SLICING THEO KÍCH THƯỚC & MẬT ĐỘ ĐÁM ĐÔNG:")
    print(f"     - Small-box (<32×32 px) : {slicing_nms['small_box_recall']*100:6.2f}% ({slicing_nms['matched_small_gt']}/{slicing_nms['num_small_gt']}) | Soft-NMS: {slicing_soft['small_box_recall']*100:6.2f}%")
    print(f"     - Regular-box (>=32²)   : {slicing_nms['regular_box_recall']*100:6.2f}%")
    print(f"     - Sparse (1–2 người)    : {slicing_nms['sparse_recall']*100:6.2f}% ({slicing_nms['matched_sparse_gt']}/{slicing_nms['num_sparse_gt']})")
    print(f"     - Crowded (>=3 người)   : {slicing_nms['crowded_recall']*100:6.2f}% ({slicing_nms['matched_crowded_gt']}/{slicing_nms['num_crowded_gt']})")
    print(f"  3. SLICING THEO NHÃN HÀNH VI GỐC (TÁC ĐỘNG LÊN STAGE 2):")
    for cid in (0, 1, 2):
        rec_c = beh["matched"][cid] / max(1, beh["support"][cid]) * 100.0
        print(f"     - Class {cid} ({CLASS_NAMES[cid]:<19}): Recall = {rec_c:6.2f}% ({beh['matched'][cid]}/{beh['support'][cid]})")
    print(f"  4. GIẢI PHẪU LỖI FN ({slicing_nms['fn']} ca) & FP ({slicing_nms['fp']} hộp thừa):")
    print(f"     - FN: Poor-Loc (0.25<=IoU<0.5) = {fn_tax['poor_loc_025_050']} ({fn_tax['poor_loc_025_050']/max(1,slicing_nms['fn'])*100:.1f}%) | Low-Score = {fn_tax['low_score_below_thr']} ({fn_tax['low_score_below_thr']/max(1,slicing_nms['fn'])*100:.1f}%) | Complete-Miss = {fn_tax['complete_miss_under_025']} ({fn_tax['complete_miss_under_025']/max(1,slicing_nms['fn'])*100:.1f}%)")
    print(f"     - FP: Duplicate quanh GT (IoU>=0.1) = {fp_tax['duplicate_cluster_near_gt']} ({fp_tax['duplicate_cluster_near_gt']/max(1,slicing_nms['fp'])*100:.1f}%) | Nhiễu nền thuần túy (IoU<0.1) = {fp_tax['background_noise']} ({fp_tax['background_noise']/max(1,slicing_nms['fp'])*100:.1f}%)")
    print(f"  5. CỔNG NGÂN SÁCH EDGE AI (HARD EDGE GATE): Within Budget = {edge_prof['within_budget']}")
    print(f"     - Params={edge_prof['params_m']}M (<=2.2M) | FLOPs={edge_prof['gflops']}G (<=4.5G) | Size={edge_prof['size_mb']}MB (<=9.0MB) | Latency={edge_prof['latency_mean_ms']}ms ({edge_prof['fps']} FPS, <=18ms)")
    print(f"  6. ĐÃ XUẤT BÁO CÁO & 30 MẪU LỖI TRỰC QUAN TẠI: {out_dir}")
    print("=" * 90)
    return report


def main() -> None:
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description="Kiểm thử đa chiều & soi lỗi Baseline Stage 1 D1 (S1-01)")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/stage1_d1_best.pt"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/audit_s1_01"))
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_stage1_audit(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        score_thr=args.score_thr,
        seed=args.seed,
        device_pref=args.device,
    )


if __name__ == "__main__":
    main()
