"""
tests/test_stage1_baseline.py
=============================
WP-3.2 — Bộ kiểm thử tự động cho `Stage1BaselineD1`, `Stage1BaselineLoss` (Bounded Sigmoid + GIoU),
giải mã vector hóa + NMS / Soft-NMS, `EarlyStopping` có pha warmup, `edge_profiler` và `Small-box Recall`.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

torch = pytest.importorskip("torch")

from src.models.stage1_baseline import (
    Stage1BaselineD1,
    Stage1BaselineLoss,
    decode_stage1_outputs,
    soft_nms_xyxy,
    nms_xyxy,
)
from src.train_stage1 import EarlyStopping, run_stage1_training
from src.utils.edge_profiler import (
    STAGE1_BUDGET,
    compute_small_box_recall,
    compute_stage1_slicing_metrics,
    profile_edge_model,
)


def test_d1_forward_shape_and_prior_bias():
    """Kiểm tra kích thước lưới 26x26 (stride 16) và xác suất khởi tạo ~0.01."""
    model = Stage1BaselineD1()
    model.eval()
    x = torch.zeros((2, 3, 416, 416), dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    assert tuple(out.shape) == (2, 5, 26, 26)
    init_obj_prob = float(torch.sigmoid(out[:, 0]).mean().item())
    assert 0.002 <= init_obj_prob <= 0.05


def test_d1_complies_with_stage1_edge_budget():
    """Đảm bảo D1 nằm gọn dưới trần phân bổ của Team Stage 1 (Params, Size, FLOPs)."""
    model = Stage1BaselineD1()
    prof = profile_edge_model(
        model,
        input_shape=(1, 3, 416, 416),
        device=torch.device("cpu"),
        warmup=2,
        runs=5,
        budget=STAGE1_BUDGET,
    )
    assert prof["total_params"] <= STAGE1_BUDGET["max_params"]
    assert prof["size_mb"] <= STAGE1_BUDGET["max_size_mb"]
    assert prof["gflops"] <= STAGE1_BUDGET["max_gflops"]
    # Riêng Params & FLOPs của D1 phải gọn nhẹ quanh ~1.07M và ~2.8 GFLOPs
    assert 800_000 <= prof["total_params"] <= 1_500_000
    assert 1.5 <= prof["gflops"] <= 3.6


def test_bounded_sigmoid_loss_prevents_gradient_explosion():
    """Safeguard 1: Ngay cả khi logit cực đoan ±50 (nơi exp(50) tràn số), loss & grad vẫn hữu hạn."""
    criterion = Stage1BaselineLoss(pos_weight=15.0, box_weight=5.0)
    preds = torch.full((2, 5, 26, 26), 50.0, dtype=torch.float32, requires_grad=True)
    targets = (
        torch.tensor([[0.0, 0.5, 0.5, 0.2, 0.3]], dtype=torch.float32),
        torch.zeros((0, 5), dtype=torch.float32),
    )
    loss = criterion(preds, targets)
    assert math.isfinite(float(loss.item()))
    loss.backward()
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()


def test_vectorized_decode_and_nms_vs_soft_nms():
    """Safeguard 3 & Đặc thù đông người: Soft-NMS giữ lại box người bơi cạnh nhau mà Hard NMS xóa."""
    boxes = torch.tensor(
        [
            [0.10, 0.10, 0.30, 0.30],
            [0.14, 0.10, 0.34, 0.30],  # chồng lấn IoU = 0.16*0.20 / (0.04+0.04-0.032) = 0.667 > 0.5
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.90, 0.85], dtype=torch.float32)

    hard_b, _ = nms_xyxy(boxes, scores, iou_thr=0.5)
    soft_b, soft_s = soft_nms_xyxy(boxes, scores, sigma=0.5, score_thr=0.25)

    assert len(hard_b) == 1, "Hard NMS phải triệt tiêu box chồng lấn IoU=0.67"
    assert len(soft_b) == 2, "Soft-NMS phải giữ lại người thứ 2 với điểm tin cậy suy giảm mượt"
    assert float(soft_s[1].item()) < 0.85

    # Kiểm tra decode_stage1_outputs và giới hạn top_k
    raw = torch.zeros((1, 5, 26, 26), dtype=torch.float32)
    raw[0, 0] = 2.0  # toàn bộ 676 ô đều vượt score_thr
    decoded = decode_stage1_outputs(raw, score_thr=0.25, top_k=50, max_detections=20, nms_type="nms")
    assert len(decoded) == 1
    assert decoded[0]["boxes"].shape[1] == 4
    assert len(decoded[0]["boxes"]) <= 20


def test_early_stopping_warmup_prevents_premature_termination():
    """Safeguard 2: Không ngắt non trong 12 epoch warmup dù mAP@50 phẳng ở 0.0."""
    stopper = EarlyStopping(patience=3, warmup_epochs=12, min_delta=1e-4, mode="max")
    for ep in range(1, 13):
        stopper.step(ep, 0.0)
        assert not stopper.should_stop, f"Không được dừng sớm trong pha warmup (epoch {ep})"
        assert stopper.bad_epochs == 0

    # Sau warmup (epoch 13, 14, 15 không cải thiện) -> dừng đúng tại epoch 15
    stopper.step(13, 0.0)
    assert not stopper.should_stop
    stopper.step(14, 0.0)
    assert not stopper.should_stop
    stopper.step(15, 0.0)
    assert stopper.should_stop


def test_small_box_recall_separates_small_and_regular_targets():
    """Kiểm tra đo riêng Small-box Recall (< 32x32 px trên ảnh 416x416)."""
    # Box nhỏ: 20x20 px -> chuẩn hóa 20/416 = 0.048
    # Box lớn: 100x100 px -> chuẩn hóa 100/416 = 0.240
    gt = [
        {
            "boxes": np.array(
                [
                    [0.10, 0.10, 0.10 + 20 / 416, 0.10 + 20 / 416],
                    [0.50, 0.50, 0.50 + 100 / 416, 0.50 + 100 / 416],
                ],
                dtype=np.float32,
            )
        }
    ]
    # Chỉ phát hiện được box lớn, bỏ sót box nhỏ
    pred = [
        {
            "boxes": np.array([[0.50, 0.50, 0.50 + 100 / 416, 0.50 + 100 / 416]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
        }
    ]
    res = compute_small_box_recall(pred, gt, img_size=(416, 416), iou_thr=0.5, score_thr=0.25)
    assert res["num_small_gt"] == 1
    assert res["num_regular_gt"] == 1
    assert res["small_box_recall"] == 0.0
    assert res["regular_box_recall"] == 1.0


def test_compute_stage1_slicing_metrics_taxonomy_and_behavior():
    """Kiểm tra đo đa chiều Slicing Metrics (Crowded, Behavior 3 lớp, FN/FP taxonomy) theo experiment_log.md."""
    # Ảnh 1 (Sparse - 2 GT: lớp 0 Drowning nhỏ & lớp 2 Swimming lớn):
    #   - GT0 (Drowning, nhỏ): bị lệch biên (pred IoU ~ 0.36 -> poor_loc FN + duplicate_cluster FP)
    #   - GT1 (Swimming, lớn): bắt trúng hoàn hảo (TP)
    #   - Thêm 1 hộp báo giả ở góc trống xa GT (IoU = 0 -> background_noise FP)
    gt = [
        {
            "boxes": np.array(
                [
                    [0.10, 0.10, 0.15, 0.15],  # 20.8x20.8 px (<32x32)
                    [0.50, 0.50, 0.75, 0.75],  # 104x104 px (>=32x32)
                ],
                dtype=np.float32,
            )
        },
        # Ảnh 2 (Crowded - 3 GT: lớp 0, 1, 2): bắt trúng 2, sót 1 do điểm thấp (0.12 < 0.25)
        {
            "boxes": np.array(
                [
                    [0.10, 0.10, 0.25, 0.25],
                    [0.35, 0.35, 0.50, 0.50],
                    [0.65, 0.65, 0.80, 0.80],
                ],
                dtype=np.float32,
            )
        },
    ]
    preds = [
        {
            "boxes": np.array(
                [
                    [0.50, 0.50, 0.75, 0.75],  # TP cho GT1
                    [0.10, 0.10, 0.18, 0.16],  # IoU với GT0 = (0.05*0.05)/(0.0025+0.0048-0.0025)=0.52? Chọn 0.19x0.17 -> IoU=0.0025/0.0063=0.397 (poor_loc)
                    [0.85, 0.05, 0.95, 0.15],  # IoU = 0 (background_noise FP)
                ],
                dtype=np.float32,
            ),
            "scores": np.array([0.92, 0.60, 0.70], dtype=np.float32),
        },
        {
            "boxes": np.array(
                [
                    [0.10, 0.10, 0.25, 0.25],  # TP (score 0.85)
                    [0.35, 0.35, 0.50, 0.50],  # TP (score 0.80)
                    [0.65, 0.65, 0.80, 0.80],  # Đúng vị trí nhưng score=0.12 < 0.25 -> low_score FN
                ],
                dtype=np.float32,
            ),
            "scores": np.array([0.85, 0.80, 0.12], dtype=np.float32),
        },
    ]
    preds[0]["boxes"][1] = np.array([0.10, 0.10, 0.19, 0.17], dtype=np.float32)  # IoU = 0.397 với GT0
    raw_cls = [[0, 2], [0, 1, 2]]

    s = compute_stage1_slicing_metrics(preds, gt, raw_classes_per_image=raw_cls, score_thr=0.25)
    assert s["tp"] == 3
    assert s["fp"] == 2
    assert s["fn"] == 2
    assert abs(s["person_recall"] - 3 / 5) < 1e-6
    assert abs(s["precision"] - 3 / 5) < 1e-6
    assert abs(s["sparse_recall"] - 1 / 2) < 1e-6
    assert abs(s["crowded_recall"] - 2 / 3) < 1e-6
    assert s["fn_taxonomy"]["poor_loc_025_050"] == 1
    assert s["fn_taxonomy"]["low_score_below_thr"] == 1
    assert s["fn_taxonomy"]["complete_miss_under_025"] == 0
    assert s["fp_taxonomy"]["duplicate_cluster_near_gt"] == 1
    assert s["fp_taxonomy"]["background_noise"] == 1
    beh = s["behavior_class_recall"]
    assert beh["support"] == {0: 2, 1: 1, 2: 2}
    assert beh["matched"] == {0: 1, 1: 1, 2: 1}


def test_train_stage1_smoke_pipeline(tmp_path):
    """Chạy thử 2 epochs x 2 batches với dữ liệu thật để xác nhận toàn bộ pipeline khớp nối."""
    manifest = Path("data/stage1/manifest.json")
    if not manifest.is_file():
        pytest.skip("Chưa sinh data/stage1/manifest.json")

    ckpt_p = tmp_path / "d1_smoke.pt"
    sum_p = tmp_path / "d1_smoke.json"
    summary = run_stage1_training(
        manifest_path=manifest,
        epochs=2,
        batch_size=4,
        num_workers=0,
        seed=42,
        device_pref="auto",
        checkpoint_path=ckpt_p,
        summary_path=sum_p,
        max_batches=2,
    )
    assert ckpt_p.is_file()
    assert sum_p.is_file()
    assert len(summary["history"]) == 2
    assert summary["edge_profile"]["within_budget"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
