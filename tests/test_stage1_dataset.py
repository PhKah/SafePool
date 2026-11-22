"""Kiểm thử hợp đồng dữ liệu, độ đầy đủ nhãn và đầu vào PyTorch Stage 1.

Các kiểm tra completeness so sánh artifact đã sinh với annotation raw hiện có:
chúng bắt được ảnh/box bị bỏ trong quá trình chuyển đổi, nhưng không thể chứng
minh annotation raw phản ánh đầy đủ mọi người trong ảnh.
"""

import json
from pathlib import Path

import pytest


MANIFEST_PATH = Path("data/stage1/manifest.json")


@pytest.fixture(scope="module")
def stage1_manifest():
    """Đọc manifest dùng chung để các test xác minh cùng một lần sinh dữ liệu."""
    assert MANIFEST_PATH.is_file(), f"Khong tim thay manifest: {MANIFEST_PATH}"
    with MANIFEST_PATH.open("r", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def test_manifest_conservation_and_mapping(stage1_manifest):
    """Kiểm tra schema một lớp cùng số lượng tổng trên phiên bản dữ liệu hiện tại.

    Các tổng 12.365 ảnh và 23.021 box là cổng bảo toàn cho dataset này; nếu thay
    phiên bản hoặc sửa nguồn, cần cập nhật kỳ vọng sau khi audit split/labels.
    """
    assert stage1_manifest["num_classes"] == 1
    assert stage1_manifest["class_names"]["0"] == "person"
    assert stage1_manifest["class_mapping"] == {"0": 0, "1": 0, "2": 0}

    image_count = sum(
        stage1_manifest["split_summary"][split]["images_count"]
        for split in ("train", "valid", "test")
    )
    box_count = sum(
        stage1_manifest["split_summary"][split]["person_boxes_count"]
        for split in ("train", "valid", "test")
    )
    assert image_count == 12365
    assert box_count == 23021


def test_label_names_keep_image_stems(stage1_manifest):
    """Kiểm tra quy ước tên dễ truy vấn và không trùng stem trong từng split."""
    all_stems = set()
    for split_name in ("train", "valid", "test"):
        split_stems = set()
        for entry in stage1_manifest["splits"][split_name]:
            image_path = Path(entry["image_path"])
            label_path = Path(entry["label_path"])
            assert image_path.is_file()
            assert label_path.is_file()
            assert image_path.stem == label_path.stem
            assert label_path.parent.name == split_name

            stem = image_path.stem
            assert stem not in split_stems
            split_stems.add(stem)
            all_stems.add(stem)

    assert len(all_stems) == len(stage1_manifest["splits"]["train"]) + len(
        stage1_manifest["splits"]["valid"]
    ) + len(stage1_manifest["splits"]["test"])


def test_all_labels_are_person_and_coordinates_valid(stage1_manifest):
    """So sánh từng dòng nhãn Stage 1 với nguồn để bắt lỗi mất/sửa bbox.

    Test yêu cầu cùng số lượng và thứ tự bbox, bốn tọa độ phải giữ nguyên, class
    chỉ được map sang 0. Đây là kiểm tra chuyển đổi đầy đủ, không phải đánh giá
    độ chính xác nội dung annotation do con người gán trong raw.
    """
    for split_name in ("train", "valid", "test"):
        for entry in stage1_manifest["splits"][split_name]:
            label_path = Path(entry["label_path"])
            image_path = Path(entry["source_image_path"])
            source_label_path = (
                image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
            )
            assert source_label_path.is_file()

            expected_rows = []
            with source_label_path.open("r", encoding="utf-8") as source_file:
                for line in source_file:
                    tokens = line.strip().split()
                    if tokens:
                        assert len(tokens) == 5
                        assert tokens[0] in {"0", "1", "2"}
                        expected_rows.append(["0", *tokens[1:]])

            actual_rows = []
            with label_path.open("r", encoding="utf-8") as label_file:
                for line in label_file:
                    tokens = line.strip().split()
                    if not tokens:
                        continue
                    assert len(tokens) == 5
                    assert tokens[0] == "0"
                    values = [float(value) for value in tokens[1:]]
                    assert all(0.0 <= value <= 1.0 for value in values)
                    assert values[2] > 0
                    assert values[3] > 0
                    actual_rows.append(tokens)

            assert actual_rows == expected_rows
            assert len(actual_rows) == entry["boxes_count"]


def test_manifest_covers_all_clean_split_images(stage1_manifest):
    """Bắt ảnh thiếu/thừa hoặc bị chuyển nhầm train, valid hay test."""
    for split_name in ("train", "valid", "test"):
        split_file = Path(f"data/splits/clean_{split_name}.txt")
        with split_file.open("r", encoding="utf-8") as file:
            expected_images = {
                line.strip().replace("\\", "/")
                for line in file
                if line.strip()
            }

        actual_images = {
            entry["source_image_path"].replace("\\", "/")
            for entry in stage1_manifest["splits"][split_name]
        }
        assert actual_images == expected_images


def test_dataset_and_dataloader():
    """Xác nhận hợp đồng tensor mà bước huấn luyện Stage 1 sẽ nhận.

    Kiểm tra một mẫu và một batch nhỏ: ảnh RGB CHW float32, target class person,
    cùng số target trong batch; không huấn luyện mô hình ở test này.
    """
    pytest.importorskip("torch")
    import torch
    from src.dataset import (
        Stage1Dataset,
        create_stage1_dataloader,
    )

    dataset = Stage1Dataset(manifest_path=MANIFEST_PATH, split="train")
    assert len(dataset) == 8655
    image, target = dataset[0]
    assert tuple(image.shape) == (3, 416, 416)
    assert image.dtype == torch.float32
    assert target.shape[1] == 5
    assert torch.all(target[:, 0] == 0)

    dataloader = create_stage1_dataloader(
        manifest_path=MANIFEST_PATH,
        split="train",
        batch_size=4,
        shuffle=False,
    )
    images, targets = next(iter(dataloader))
    assert tuple(images.shape) == (4, 3, 416, 416)
    assert len(targets) == 4
