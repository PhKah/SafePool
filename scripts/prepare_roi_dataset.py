#!/usr/bin/env python3
"""
scripts/prepare_roi_dataset.py
==============================
WP-1.3 — Cắt ROI từ data/raw/ theo data/splits/clean_{train,valid,test}.txt
sang data/processed/{train,valid,test}/ phục vụ Stage 2 (3-Class ROI Classification).

Chức năng:
  - Đọc danh sách ảnh sạch từ data/splits/clean_{train,valid,test}.txt.
  - Phân tích nhãn YOLO tương ứng trong data/raw/{split}/labels/.
  - Thêm padding ngữ cảnh ~20% mỗi chiều (giữ bối cảnh mặt nước/thành bể).
  - Lọc bỏ các box ở tập TRAIN có cạnh ngắn < 16 px (trước padding) để tránh nhòe khi phóng to.
  - Crop và resize ROI về độ phân giải chuẩn 128×128 pixel.
  - Lưu vào thư mục phân lớp: data/processed/{split}/{class_id}/{image_base}_roi{idx}.jpg.
  - Tính và xuất Smoothed Class Weights sqrt(N_max / N_c) và file thống kê roi_stats.json.

Nguyên tắc:
  - Cố định seed, không hardcode đường dẫn cá nhân.
  - Bảo toàn 100% Clean Group Split (0% rò rỉ dữ liệu).
  - Không lọc box nhỏ ở valid/test nhằm bảo đảm tính khách quan khi đánh giá.
"""

import os
import sys
import json
import math
import argparse
from collections import Counter
from typing import Dict, List, Tuple

from PIL import Image


# =============================================================================
# Cấu hình mặc định
# =============================================================================
CLASS_NAMES = {
    0: "Drowning",
    1: "Person out of water",
    2: "Swimming",
}
DEFAULT_IMG_SIZE = 128
DEFAULT_PAD_RATIO = 0.20
MIN_BOX_SIDE_PX = 16


# =============================================================================
# Xử lý nhãn & Tọa độ Bounding Box
# =============================================================================
def parse_yolo_label(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Đọc tệp nhãn YOLOv8: mỗi dòng gồm `class_id cx cy w h` (chuẩn hóa [0.0, 1.0]).
    Trả về danh sách các tuple (class_id, cx, cy, w, h).
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) != 5:
                continue
            cls_id = int(tokens[0])
            cx, cy, w, h = float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])
            boxes.append((cls_id, cx, cy, w, h))
    return boxes


def get_label_path_for_image(image_path: str) -> str:
    """
    Xác định đường dẫn tệp nhãn từ đường dẫn tệp ảnh theo quy ước data/raw/{split}/images/ -> data/raw/{split}/labels/.
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    split_dir = os.path.dirname(os.path.dirname(image_path))
    return os.path.join(split_dir, "labels", image_name)


def crop_roi_with_context(
    img: Image.Image,
    cx: float, cy: float, bw: float, bh: float,
    pad_ratio: float = DEFAULT_PAD_RATIO,
    target_size: int = DEFAULT_IMG_SIZE,
) -> Image.Image:
    """
    Cắt ROI kèm padding ngữ cảnh và resize về target_size x target_size.
    """
    img_w, img_h = img.size

    box_w_px = bw * img_w
    box_h_px = bh * img_h
    cx_px = cx * img_w
    cy_px = cy * img_h

    pad_w = box_w_px * pad_ratio
    pad_h = box_h_px * pad_ratio

    x1 = max(0, int(cx_px - box_w_px / 2.0 - pad_w))
    y1 = max(0, int(cy_px - box_h_px / 2.0 - pad_h))
    x2 = min(img_w, int(cx_px + box_w_px / 2.0 + pad_w))
    y2 = min(img_h, int(cy_px + box_h_px / 2.0 + pad_h))

    # Xử lý trường hợp biên dị thường
    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)

    roi = img.crop((x1, y1, x2, y2))
    roi = roi.resize((target_size, target_size), Image.BILINEAR)
    return roi


# =============================================================================
# Xử lý từng Split
# =============================================================================
def process_split(
    split_name: str,
    image_paths: List[str],
    output_dir: str,
    filter_small_boxes: bool,
    target_size: int = DEFAULT_IMG_SIZE,
    pad_ratio: float = DEFAULT_PAD_RATIO,
    min_box_side: int = MIN_BOX_SIDE_PX,
) -> Dict[int, int]:
    """
    Cắt và lưu ROI cho toàn bộ ảnh trong một tập dữ liệu (train/valid/test).
    """
    roi_counts = Counter()
    filtered_boxes = 0
    error_count = 0

    for cls_id in CLASS_NAMES:
        os.makedirs(os.path.join(output_dir, str(cls_id)), exist_ok=True)

    for img_idx, img_path in enumerate(image_paths):
        if (img_idx + 1) % 1000 == 0:
            print(f"  [{split_name}] Đã xử lý {img_idx + 1}/{len(image_paths)} ảnh...")

        lbl_path = get_label_path_for_image(img_path)
        boxes = parse_yolo_label(lbl_path)
        if not boxes:
            continue

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img_w, img_h = img.size
                base_name = os.path.splitext(os.path.basename(img_path))[0]

                for roi_idx, (cls_id, cx, cy, bw, bh) in enumerate(boxes):
                    # Lọc box nhỏ chỉ áp dụng cho tập train
                    if filter_small_boxes:
                        min_side = min(bw * img_w, bh * img_h)
                        if min_side < min_box_side:
                            filtered_boxes += 1
                            continue

                    roi = crop_roi_with_context(img, cx, cy, bw, bh, pad_ratio, target_size)
                    out_path = os.path.join(output_dir, str(cls_id), f"{base_name}_roi{roi_idx}.jpg")
                    roi.save(out_path, quality=95)
                    roi_counts[cls_id] += 1
        except Exception as e:
            print(f"  [CẢNH BÁO] Lỗi khi xử lý ảnh {img_path}: {e}")
            error_count += 1
            continue

    if filtered_boxes > 0:
        print(f"  [{split_name}] Đã lọc {filtered_boxes} bounding boxes có cạnh ngắn < {min_box_side} px.")
    if error_count > 0:
        print(f"  [{split_name}] Gặp {error_count} ảnh lỗi.")

    return dict(roi_counts)


# =============================================================================
# Tính Smoothed Class Weights
# =============================================================================
def compute_smoothed_class_weights(train_counts: Dict[int, int]) -> Dict[int, float]:
    """
    Tính trọng số lớp làm mịn: w_c = sqrt(N_max / N_c).
    """
    if not train_counts:
        return {}
    n_max = max(train_counts.values())
    weights = {}
    for cls_id in sorted(train_counts.keys()):
        n_c = train_counts[cls_id]
        weights[cls_id] = round(math.sqrt(n_max / n_c), 4) if n_c > 0 else 1.0
    return weights


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="WP-1.3: Cắt ROI cho Stage 2 từ Clean Group Split.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Thư mục data thô")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Thư mục chứa clean_*.txt")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Thư mục lưu ảnh ROI")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Kích thước ảnh ROI vuông")
    parser.add_argument("--pad-ratio", type=float, default=DEFAULT_PAD_RATIO, help="Tỷ lệ padding ngữ cảnh")
    parser.add_argument("--min-box-side", type=int, default=MIN_BOX_SIDE_PX, help="Ngưỡng cạnh ngắn box train")
    args = parser.parse_args()

    print("=" * 72)
    print("WP-1.3: Chuẩn bị ROI Dataset cho Stage 2 (3-Class ROI Classification)")
    print("=" * 72)
    print(f"  Raw dir    : {args.raw_dir}")
    print(f"  Splits dir : {args.splits_dir}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  ROI size   : {args.img_size}×{args.img_size}")
    print(f"  Pad ratio  : {args.pad_ratio * 100:.0f}%")
    print(f"  Min box side (train only): {args.min_box_side} px")
    print()

    split_paths = {}
    for s_name in ["train", "valid", "test"]:
        txt_file = os.path.join(args.splits_dir, f"clean_{s_name}.txt")
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Không tìm thấy file danh sách: {txt_file}")
        with open(txt_file, "r", encoding="utf-8") as f:
            paths = [line.strip() for line in f if line.strip()]
        split_paths[s_name] = paths
        print(f"  [{s_name}] Đọc thành công {len(paths)} đường dẫn ảnh.")

    print()
    all_stats = {}
    for s_name in ["train", "valid", "test"]:
        print(f"[XỬ LÝ] Đang crop ROI cho tập '{s_name}'...")
        split_out_dir = os.path.join(args.output_dir, s_name)
        os.makedirs(split_out_dir, exist_ok=True)

        counts = process_split(
            split_name=s_name,
            image_paths=split_paths[s_name],
            output_dir=split_out_dir,
            filter_small_boxes=(s_name == "train"),
            target_size=args.img_size,
            pad_ratio=args.pad_ratio,
            min_box_side=args.min_box_side,
        )
        all_stats[s_name] = {str(k): v for k, v in sorted(counts.items())}
        total = sum(counts.values())
        print(f"  [{s_name}] Hoàn thành: tổng {total} ROI.")
        for cid in sorted(counts.keys()):
            pct = counts[cid] / total * 100.0 if total > 0 else 0
            print(f"    Lớp {cid} ({CLASS_NAMES.get(cid, '?')}): {counts[cid]} ({pct:.1f}%)")
        print()

    train_counts_int = {int(k): v for k, v in all_stats["train"].items()}
    weights = compute_smoothed_class_weights(train_counts_int)

    manifest_stats = {
        "config": {
            "img_size": args.img_size,
            "pad_ratio": args.pad_ratio,
            "min_box_side_train": args.min_box_side,
        },
        "roi_counts": all_stats,
        "smoothed_class_weights": {str(k): v for k, v in weights.items()},
    }

    os.makedirs(args.output_dir, exist_ok=True)
    stats_json_path = os.path.join(args.output_dir, "roi_stats.json")
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_stats, f, indent=2, ensure_ascii=False)
    print(f"[LƯU TRỮ] Đã xuất file thống kê: {stats_json_path}")

    print("\n" + "=" * 72)
    print(f"{'TẬP DỮ LIỆU':<12} | {'C0: DROWNING':<14} | {'C1: OUT WATER':<14} | {'C2: SWIMMING':<14} | {'TỔNG ROI':<10}")
    print("-" * 72)
    for s_name in ["train", "valid", "test"]:
        st = all_stats[s_name]
        c0, c1, c2 = st.get("0", 0), st.get("1", 0), st.get("2", 0)
        print(f"{s_name.upper():<12} | {c0:<14} | {c1:<14} | {c2:<14} | {c0 + c1 + c2:<10}")
    print("=" * 72)
    print("SMOOTHED CLASS WEIGHTS (sqrt(N_max / N_c)) trên tập train:")
    for cid, w in weights.items():
        print(f"  Class {cid} ({CLASS_NAMES.get(cid, '?')}): {w}")
    print("=" * 72)


if __name__ == "__main__":
    main()
