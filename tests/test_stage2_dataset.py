"""
tests/test_stage2_dataset.py
============================
WP-1.3 — Bộ kiểm thử tự động (Unit Test & Acceptance Verification)
cho Pipeline xử lý dữ liệu Stage 2 (3-Class ROI Classification).

Tiêu chí nghiệm thu (Acceptance Criteria):
  1. Tồn tại đầy đủ dữ liệu ROI cho 3 tập train, valid, test và không có tập tin hỏng/rỗng.
  2. Kích thước Tensor và kênh màu chính xác: [B, 3, 128, 128] kiểu float32 chuẩn RGB trong khoảng [0.0, 1.0].
  3. Ánh xạ nhãn hợp lệ: nhãn thuộc tập {0, 1, 2}.
  4. Trọng số làm mịn (Smoothed Class Weights): w_c = sqrt(N_max / N_c), lớp 1 chiếm trọng số cao nhất.
  5. 0% Rò rỉ dữ liệu (Cross-split Leakage): Không trùng frame prefix giữa các tập train, valid, test.
"""

import os
import sys
import json
import math
import pytest
import torch
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.datasets.stage2_dataset import (
    Stage2ROIDataset,
    get_stage2_dataloaders,
    compute_smoothed_class_weights,
    class_weights_to_tensor,
    NUM_CLASSES,
    DEFAULT_IMG_SIZE,
)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
STATS_PATH = os.path.join(PROCESSED_DIR, "roi_stats.json")
BATCH_SIZE = 8


@pytest.fixture(scope="module")
def roi_stats():
    assert os.path.exists(STATS_PATH), f"Không tìm thấy file thống kê {STATS_PATH}"
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def train_dataset():
    train_dir = os.path.join(PROCESSED_DIR, "train")
    assert os.path.isdir(train_dir), f"Không tìm thấy thư mục {train_dir}"
    return Stage2ROIDataset(root_dir=train_dir)


@pytest.fixture(scope="module")
def dataloaders():
    return get_stage2_dataloaders(
        processed_dir=PROCESSED_DIR,
        batch_size=BATCH_SIZE,
        num_workers=0,
        seed=42,
    )


class TestROIIntegrityAndDistribution:
    """Kiểm tra tính toàn vẹn thư mục, tệp ảnh và số lượng mẫu sau lọc."""

    @pytest.mark.parametrize("split", ["train", "valid", "test"])
    @pytest.mark.parametrize("cls_id", [0, 1, 2])
    def test_roi_files_exist_and_non_empty(self, split, cls_id):
        cls_dir = os.path.join(PROCESSED_DIR, split, str(cls_id))
        assert os.path.isdir(cls_dir), f"Thư mục không tồn tại: {cls_dir}"
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(".jpg")]
        assert len(files) > 0, f"Lớp {cls_id} trong tập {split} không có ảnh nào!"

        # Kiểm tra file mẫu đầu tiên
        sample_path = os.path.join(cls_dir, files[0])
        with Image.open(sample_path) as img:
            assert img.size == (DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE), (
                f"Kích thước file {sample_path} ({img.size}) != ({DEFAULT_IMG_SIZE}, {DEFAULT_IMG_SIZE})"
            )
            assert img.mode in ("RGB", "L"), f"Hệ màu không hợp lệ: {img.mode}"

    def test_train_filtering_behavior(self, roi_stats):
        """Xác nhận tập train đã lọc bớt các box nhỏ < 16px."""
        train_counts = roi_stats["roi_counts"]["train"]
        total_train = sum(train_counts.values())
        assert total_train < 16321, f"Tập train chưa được lọc box nhỏ: {total_train} >= 16,321"
        assert train_counts["1"] > 0, "Lớp 1 trên tập train bị lọc hết!"


class TestDatasetAndDataLoaderContract:
    """Kiểm tra định dạng Tensor, Batch, Dtype và nhãn trả về."""

    def test_single_sample_properties(self, train_dataset):
        img_tensor, label = train_dataset[0]
        assert isinstance(img_tensor, torch.Tensor), "Đầu ra ảnh không phải là torch.Tensor"
        assert img_tensor.shape == (3, DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE), (
            f"Shape tensor ({img_tensor.shape}) != (3, {DEFAULT_IMG_SIZE}, {DEFAULT_IMG_SIZE})"
        )
        assert img_tensor.dtype == torch.float32, f"Dtype ({img_tensor.dtype}) != float32"
        assert 0.0 <= img_tensor.min() and img_tensor.max() <= 1.0, "Giá trị pixel nằm ngoài [0.0, 1.0]"
        assert label in {0, 1, 2}, f"Nhãn {label} không thuộc {0, 1, 2}"

    def test_batch_dataloader(self, dataloaders):
        train_loader = dataloaders["train"]
        batch_imgs, batch_labels = next(iter(train_loader))

        assert batch_imgs.shape == (BATCH_SIZE, 3, DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE)
        assert batch_labels.shape == (BATCH_SIZE,)
        assert batch_labels.dtype == torch.int64 or batch_labels.dtype == torch.long
        assert ((batch_labels >= 0) & (batch_labels <= 2)).all()


class TestSmoothedClassWeights:
    """Kiểm tra logic tính trọng số làm mịn chống mất cân bằng lớp."""

    def test_weights_calculation_formula(self):
        mock_counts = {0: 1000, 1: 100, 2: 500}
        weights = compute_smoothed_class_weights(mock_counts)
        assert weights[0] == 1.0
        assert math.isclose(weights[1], math.sqrt(1000 / 100), rel_tol=1e-3)
        assert math.isclose(weights[2], math.sqrt(1000 / 500), rel_tol=1e-3)

    def test_real_dataset_weights(self, train_dataset):
        counts = train_dataset.get_class_counts()
        weights = compute_smoothed_class_weights(counts)
        assert weights[1] > weights[0], "Trọng số lớp thiểu số phải lớn hơn lớp đa số"
        assert weights[1] > weights[2], "Trọng số lớp thiểu số phải lớn nhất"

        tensor_w = class_weights_to_tensor(counts)
        assert tensor_w.shape == (NUM_CLASSES,)
        assert tensor_w[1] > tensor_w[0]


class TestZeroDataLeakage:
    """Xác thực bảo toàn ranh giới Clean Group Split ở mức ảnh ROI."""

    def _extract_frame_prefixes(self, split_name: str) -> set:
        prefixes = set()
        split_dir = os.path.join(PROCESSED_DIR, split_name)
        for cls_id in range(NUM_CLASSES):
            cls_dir = os.path.join(split_dir, str(cls_id))
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if ".rf." in fname:
                    prefix = fname.split(".rf.")[0].lower()
                    prefixes.add(prefix)
        return prefixes

    def test_no_leakage_between_splits(self):
        train_pref = self._extract_frame_prefixes("train")
        valid_pref = self._extract_frame_prefixes("valid")
        test_pref = self._extract_frame_prefixes("test")

        assert len(train_pref & valid_pref) == 0, f"Rò rỉ train và valid: {train_pref & valid_pref}"
        assert len(train_pref & test_pref) == 0, f"Rò rỉ train và test: {train_pref & test_pref}"
        assert len(valid_pref & test_pref) == 0, f"Rò rỉ valid và test: {valid_pref & test_pref}"

