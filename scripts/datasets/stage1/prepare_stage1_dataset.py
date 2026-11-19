#!/usr/bin/env python3
"""
Tạo nhãn person dùng cho Stage 1 từ clean Group Split.

Ảnh gốc không được sao chép. Tên nhãn Stage 1 giữ nguyên stem của ảnh và được
tách theo thư mục train/valid/test.

Luồng xử lý: đọc các danh sách clean_*.txt cạnh split manifest, tìm nhãn raw
theo cùng stem với mỗi ảnh, xác thực dữ liệu, chuyển class 0/1/2 thành 0 và
ghi manifest Stage 1. Danh sách clean_*.txt là nguồn liệt kê ảnh thực tế; JSON
split được mở để xác nhận tồn tại/đọc được nhưng không dùng để dựng lại danh sách.
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SPLIT_FILE = ROOT_DIR / "data/splits/clean_group_split.json"
DEFAULT_RAW_DIR = ROOT_DIR / "data/raw"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data/stage1"


def parse_args() -> argparse.Namespace:
    """Đọc cấu hình đường dẫn đầu vào và đầu ra từ dòng lệnh.

    Giá trị mặc định trỏ tới data/splits, data/raw và data/stage1 trong repo.
    --raw-dir hiện chỉ được dùng để kiểm tra thư mục raw tồn tại; các đường dẫn
    ảnh cụ thể được lấy từ file clean_*.txt và giải quyết tương đối theo ROOT_DIR.
    """
    parser = argparse.ArgumentParser(description="Prepare Stage 1 person labels")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def relative_path(path: Path) -> str:
    """Chuẩn hóa đường dẫn ghi vào manifest để dễ di chuyển giữa máy.

    Trả đường dẫn tương đối POSIX nếu file nằm trong repository; file bên ngoài
    repository được ghi bằng đường dẫn tuyệt đối vì không thể biểu diễn tương đối
    ổn định với gốc dự án.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return resolved.as_posix()


def validate_image_label_pair(image_path: Path, label_path: Path) -> None:
    """Kiểm tra cặp raw trước khi tạo nhãn mới, dừng ngay khi dữ liệu sai.

    Yêu cầu ảnh mở được và có kích thước 416x416; mỗi dòng nhãn không rỗng có
    class 0/1/2 cùng bốn số YOLO trong [0, 1], với width/height dương. Hàm này
    kiểm tra tính hợp lệ cú pháp/phạm vi, không thẩm định bbox có bao đúng người
    hay annotation raw có bỏ sót người hay không.

    Raises:
        FileNotFoundError: Nếu ảnh hoặc nhãn nguồn không tồn tại.
        ValueError: Nếu kích thước ảnh, cấu trúc dòng, class hoặc bbox không hợp lệ.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Khong tim thay anh: {image_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"Khong tim thay nhan: {label_path}")

    with Image.open(image_path) as image:
        if image.size != (416, 416):
            raise ValueError(f"Kich thuoc anh khong phai 416x416: {image_path}")

    with label_path.open("r", encoding="utf-8") as label_file:
        for line_number, line in enumerate(label_file, start=1):
            tokens = line.strip().split()
            if not tokens:
                continue
            if len(tokens) != 5:
                raise ValueError(
                    f"Dinh dang YOLO sai tai {label_path}:{line_number}"
                )

            class_id = tokens[0]
            if class_id not in {"0", "1", "2"}:
                raise ValueError(
                    f"Class ID khong hop le tai {label_path}:{line_number}: {class_id}"
                )

            values = [float(value) for value in tokens[1:]]
            if any(value < 0 or value > 1 for value in values):
                raise ValueError(
                    f"Toa do nhan ngoai khoang [0,1] tai {label_path}:{line_number}"
                )
            if values[2] <= 0 or values[3] <= 0:
                raise ValueError(
                    f"Kich thuoc bbox khong duong tai {label_path}:{line_number}"
                )


def source_label_path(image_path: Path) -> Path:
    """Suy ra vị trí nhãn raw từ quy ước thư mục YOLO.

    Với data/raw/<split>/images/name.jpg, kết quả là
    data/raw/<split>/labels/name.txt; giữ nguyên stem giúp đối chiếu trực tiếp.
    """
    label_name = f"{image_path.stem}.txt"
    return image_path.parent.parent / "labels" / label_name


def read_split_images(split_file: Path) -> Dict[str, List[Path]]:
    """Đọc danh sách ảnh canonical cho ba split sạch.

    Mỗi danh sách clean_{train,valid,test}.txt phải nằm cạnh JSON split manifest.
    Các đường dẫn tương đối trong danh sách được tính từ ROOT_DIR, không phải
    từ vị trí script hay --raw-dir.

    Raises:
        FileNotFoundError: Nếu không có JSON manifest hoặc một danh sách split.
        json.JSONDecodeError: Nếu nội dung JSON không hợp lệ.
    """
    with split_file.open("r", encoding="utf-8") as file:
        json.load(file)

    splits: Dict[str, List[Path]] = {}
    for split_name in ("train", "valid", "test"):
        text_path = split_file.parent / f"clean_{split_name}.txt"
        if not text_path.is_file():
            raise FileNotFoundError(f"Khong tim thay danh sach split: {text_path}")

        split_images: List[Path] = []
        with text_path.open("r", encoding="utf-8") as file:
            for line in file:
                entry = line.strip()
                if not entry:
                    continue
                split_images.append((ROOT_DIR / entry).resolve())
        splits[split_name] = split_images

    return splits


def convert_label(source_label_path: Path, target_label_path: Path) -> int:
    """Chuyển nội dung một nhãn raw sang định dạng một lớp của Stage 1.

    Mỗi bbox được giữ nguyên thứ tự và bốn trường tọa độ; chỉ class ID được đổi
    thành 0 (person). Tạo thư mục cha nếu cần và trả về số bbox đã ghi, kể cả
    trường hợp file rỗng thì số trả về là 0.
    """
    box_count = 0
    target_label_path.parent.mkdir(parents=True, exist_ok=True)

    with source_label_path.open("r", encoding="utf-8") as source_file:
        converted_lines: List[str] = []
        for line in source_file:
            tokens = line.strip().split()
            if not tokens:
                continue
            if len(tokens) != 5:
                raise ValueError(f"Dinh dang YOLO sai: {source_label_path}")
            if tokens[0] not in {"0", "1", "2"}:
                raise ValueError(f"Class ID khong hop le: {source_label_path}")
            converted_lines.append("0 " + " ".join(tokens[1:]))
            box_count += 1

    target_label_path.write_text(
        "\n".join(converted_lines) + ("\n" if converted_lines else ""),
        encoding="utf-8",
    )
    return box_count


def validate_unique_stems(split_name: str, image_paths: Iterable[Path]) -> None:
    """Ngăn hai ảnh cùng split ghi đè vào một file nhãn chung.

    Do tên nhãn chỉ dùng stem ảnh, trùng stem trong cùng split là mơ hồ và phải
    xử lý dữ liệu nguồn thay vì âm thầm thêm tên khác hoặc ghi đè.
    """
    stems: Dict[str, Path] = {}
    for image_path in image_paths:
        stem = image_path.stem
        if stem in stems:
            raise ValueError(
                f"Stem anh trung nhau trong split {split_name}: "
                f"{stems[stem]} va {image_path}"
            )
        stems[stem] = image_path


def prepare_split(
    split_name: str,
    image_paths: List[Path],
    output_dir: Path,
) -> tuple[Dict[str, object], Dict[str, int]]:
    """Tạo nhãn cho một split và thu thập thống kê từng mẫu.

    Mỗi entry manifest ánh xạ ảnh raw với nhãn Stage 1 cùng split, ghi số box
    của mẫu. Phần summary tổng hợp số ảnh, số person box và phân bố class trước
    mapping để có thể audit việc gộp lớp.
    """
    validate_unique_stems(split_name, image_paths)
    label_dir = output_dir / "labels" / split_name
    entries: List[Dict[str, object]] = []
    source_class_distribution: Counter[str] = Counter()
    person_boxes_count = 0

    for image_path in image_paths:
        source_label = source_label_path(image_path)
        validate_image_label_pair(image_path, source_label)

        with source_label.open("r", encoding="utf-8") as source_file:
            for line in source_file:
                tokens = line.strip().split()
                if tokens:
                    source_class_distribution[tokens[0]] += 1

        target_label = label_dir / f"{image_path.stem}.txt"
        box_count = convert_label(source_label, target_label)
        person_boxes_count += box_count
        entries.append(
            {
                "source_split": split_name,
                "source_image_path": relative_path(image_path),
                "image_path": relative_path(image_path),
                "label_path": relative_path(target_label),
                "boxes_count": box_count,
            }
        )

    summary = {
        "images_count": len(image_paths),
        "person_boxes_count": person_boxes_count,
        "source_class_distribution": dict(sorted(source_class_distribution.items())),
    }
    return {"summary": summary, "entries": entries}, dict(source_class_distribution)


def main() -> None:
    """Chạy pipeline đầy đủ và ghi mới artifact nhãn cùng manifest.

    LƯU Ý QUAN TRỌNG: trước khi sinh lại dữ liệu, hàm xóa đệ quy
    <output-dir>/labels rồi tạo lại ba thư mục nhãn. Chỉ chạy với output-dir đã
    xác định đúng; không đặt nó trỏ vào thư mục dữ liệu nguồn hoặc thư mục dùng
    chung khác. Ảnh raw không bị xóa hay sao chép.
    """
    args = parse_args()
    split_file = args.split_file.resolve()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not split_file.is_file():
        raise FileNotFoundError(f"Khong tim thay manifest split: {split_file}")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Khong tim thay thu muc du lieu raw: {raw_dir}")

    splits = read_split_images(split_file)
    label_root = output_dir / "labels"
    if label_root.exists():
        shutil.rmtree(label_root)
    label_root.mkdir(parents=True, exist_ok=True)

    manifest_splits: Dict[str, List[Dict[str, object]]] = {}
    split_summary: Dict[str, Dict[str, object]] = {}

    for split_name in ("train", "valid", "test"):
        prepared, _ = prepare_split(split_name, splits[split_name], output_dir)
        split_summary[split_name] = prepared["summary"]
        manifest_splits[split_name] = prepared["entries"]

    manifest = {
        "generated_by": "prepare_stage1_dataset.py",
        "format_version": 1,
        "source_manifest": relative_path(split_file),
        "image_size": [416, 416],
        "num_classes": 1,
        "class_names": {"0": "person"},
        "class_mapping": {"0": 0, "1": 0, "2": 0},
        "split_summary": split_summary,
        "splits": manifest_splits,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Da sinh manifest Stage 1: {manifest_path}")
    for split_name in ("train", "valid", "test"):
        summary = split_summary[split_name]
        print(
            f"{split_name}: {summary['images_count']} anh, "
            f"{summary['person_boxes_count']} person boxes"
        )


if __name__ == "__main__":
    main()
