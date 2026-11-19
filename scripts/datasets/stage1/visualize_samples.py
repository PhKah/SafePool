#!/usr/bin/env python3
"""Tạo ảnh minh họa bbox Stage 1 để rà soát trực quan dữ liệu.

Script lấy mẫu có seed cố định ở từng split, đọc nhãn YOLO chuẩn hóa, vẽ bbox
person lên bản sao RGB rồi lưu riêng dưới visualizations/{train,valid,test}.
Việc visualize hỗ trợ phát hiện nhanh lỗi mapping/toạ độ nhưng chỉ một vài ảnh
mẫu không thay thế kiểm tra toàn bộ annotation.
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT_DIR / "data/stage1/manifest.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data/stage1/visualizations"


def parse_args() -> argparse.Namespace:
    """Đọc đường dẫn manifest/đầu ra cùng cấu hình lấy mẫu.

    --samples-per-split đặt số ảnh tối đa cho mỗi split; --seed giúp chọn lại
    đúng các mẫu khi manifest và thứ tự entry không đổi.
    """
    parser = argparse.ArgumentParser(description="Visualize Stage 1 samples")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-split", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_project_path(path: str) -> Path:
    """Chuyển đường dẫn manifest thành đường dẫn tuyệt đối trên máy hiện tại.

    Đường dẫn tương đối được tính từ ROOT_DIR giống với Stage1Dataset; đường dẫn
    tuyệt đối được giữ nguyên.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (ROOT_DIR / candidate).resolve()


def draw_person_boxes(image: Image.Image, label_path: Path) -> Image.Image:
    """Vẽ tất cả bbox YOLO person lên bản sao ảnh và trả ảnh đã annotate.

    Tọa độ tâm và kích thước chuẩn hóa được đổi sang pixel dựa trên kích thước
    ảnh đang vẽ. Ảnh đầu vào không bị sửa tại chỗ. Nhãn sai class hoặc sai số
    trường gây lỗi để không tạo hình minh họa có vẻ hợp lệ nhưng thiếu bbox.
    """
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    with label_path.open("r", encoding="utf-8") as label_file:
        for line in label_file:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] != "0" or len(tokens) != 5:
                raise ValueError(f"Nhan Stage 1 khong hop le: {label_path}")

            _, x_center, y_center, box_width, box_height = map(float, tokens)
            left = (x_center - box_width / 2) * width
            top = (y_center - box_height / 2) * height
            right = (x_center + box_width / 2) * width
            bottom = (y_center + box_height / 2) * height
            draw.rectangle((left, top, right, bottom), outline="red", width=2)
            draw.text((left, max(top - 10, 0)), "person", fill="red")

    return annotated


def main() -> None:
    """Đọc manifest, lấy mẫu từng split và lưu ảnh minh họa.

    Thư mục đầu ra được tạo nếu chưa có; nội dung đầu ra là artifact kiểm tra,
    không được dùng làm dữ liệu train. Tên file ảnh gốc được giữ để đối chiếu.
    """
    args = parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()

    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    random.seed(args.seed)
    saved_count = 0

    for split_name in ("train", "valid", "test"):
        split_output = output_dir / split_name
        split_output.mkdir(parents=True, exist_ok=True)
        entries = manifest["splits"][split_name]
        sample_entries = random.sample(
            entries,
            k=min(args.samples_per_split, len(entries)),
        )

        for entry in sample_entries:
            image_path = resolve_project_path(entry["image_path"])
            label_path = resolve_project_path(entry["label_path"])
            with Image.open(image_path) as image:
                annotated = draw_person_boxes(image, label_path)

            output_path = split_output / f"{image_path.stem}.jpg"
            annotated.save(output_path)
            saved_count += 1

    print(f"Da luu {saved_count} anh truc quan vao {output_dir}")


if __name__ == "__main__":
    main()
