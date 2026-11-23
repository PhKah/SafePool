"""Đọc dữ liệu định vị person Stage 1 và đóng gói thành batch PyTorch.

Dataset dùng manifest làm bảng ánh xạ ảnh–nhãn. Ảnh được đọc thành RGB, resize
về kích thước yêu cầu rồi đổi thành tensor float32 theo thứ tự kênh CHW, giá trị
trong [0, 1]. Mỗi target có dạng [N, 5], gồm class_id và bbox YOLO chuẩn hóa.
Collate function giữ target riêng từng ảnh vì số lượng người trong ảnh thay đổi.
"""

import json
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = ROOT_DIR / "data/stage1/manifest.json"


class Stage1Dataset(Dataset):
    """Cung cấp từng cặp ảnh–bbox person cho một split.

    Manifest lưu đường dẫn ảnh và nhãn tương ứng; class ID Stage 1 hợp lệ là 0.
    Lớp này không thực hiện augmentation cho bbox. Nếu truyền transform, transform
    phải chỉ đổi màu/giá trị ảnh hoặc tự bảo toàn hình học; phép biến đổi hình học
    đồng thời phải cập nhật bbox ở phía gọi.
    """

    def __init__(
        self,
        manifest_path: Path = DEFAULT_MANIFEST_PATH,
        split: str = "train",
        image_size: Tuple[int, int] = (416, 416),
        transform: Optional[Callable] = None,
    ) -> None:
        """Nạp danh sách mẫu và lưu cấu hình tiền xử lý.

        Args:
            manifest_path: Manifest do script chuẩn bị Stage 1 tạo ra.
            split: Một trong ba tập train, valid hoặc test.
            image_size: Kích thước Pillow resize theo thứ tự (width, height).
            transform: Phép biến đổi nhận ảnh Pillow và trả ảnh/tensor tương thích.

        Raises:
            ValueError: Nếu tên split không hợp lệ hoặc kích thước khác manifest.
            FileNotFoundError: Nếu không mở được manifest.
        """
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Split khong hop le: {split}")

        self.manifest_path = Path(manifest_path).resolve()
        self.split = split
        self.image_size = image_size
        self.transform = transform

        with self.manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)

        if manifest["image_size"] != list(image_size):
            raise ValueError(
                f"Manifest yeu cau {manifest['image_size']} nhung Dataset dung {image_size}"
            )
        self.entries = manifest["splits"][split]

    def __len__(self) -> int:
        """Trả số mẫu ảnh để PyTorch lập chỉ mục và tính số batch."""
        return len(self.entries)

    def resolve_project_path(self, path: str) -> Path:
        """Chuyển đường dẫn trong manifest thành đường dẫn tuyệt đối.

        Đường dẫn tuyệt đối được giữ nguyên; đường dẫn tương đối được tính từ
        thư mục gốc repository, nhờ đó manifest không phụ thuộc working directory.
        """
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate.resolve()
        return (ROOT_DIR / candidate).resolve()

    def __getitem__(self, index: int):
        """Đọc và tiền xử lý một ảnh, sau đó tải target cùng chỉ mục.

        Ảnh đầu ra mặc định có shape [3, height, width], dtype float32 và miền
        giá trị [0, 1]. Target có shape [N, 5] với [class, x_center, y_center,
        width, height] chuẩn hóa. Không tự sửa nhãn khi ảnh/nhãn bị thiếu hoặc sai.

        Raises:
            FileNotFoundError: Nếu ảnh hoặc nhãn trong manifest không tồn tại.
            ValueError: Nếu nhãn sai định dạng hoặc có class khác 0.
        """
        entry = self.entries[index]
        image_path = self.resolve_project_path(entry["image_path"])
        label_path = self.resolve_project_path(entry["label_path"])

        if not image_path.is_file():
            raise FileNotFoundError(f"Khong tim thay anh: {image_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"Khong tim thay nhan: {label_path}")

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB").resize(self.image_size)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(
                __import__("numpy").asarray(image, dtype="float32").copy()
            ).permute(2, 0, 1).div(255.0)

        target = self._load_target(label_path)
        return image, target

    def _load_target(self, label_path: Path) -> torch.Tensor:
        """Parse nhãn YOLO của một ảnh thành tensor float32.

        Một dòng cần có năm trường, class 0 và bốn giá trị bbox. Ảnh không có
        bbox được biểu diễn bằng tensor rỗng shape [0, 5], không phải None.
        Kiểm tra giá trị tọa độ trong [0, 1] được thực hiện lúc sinh dữ liệu/test.
        """
        rows = []
        with label_path.open("r", encoding="utf-8") as label_file:
            for line in label_file:
                tokens = line.strip().split()
                if not tokens:
                    continue
                if len(tokens) != 5:
                    raise ValueError(f"Dinh dang nhan sai: {label_path}")
                if tokens[0] != "0":
                    raise ValueError(f"Nhan Stage 1 phai la class 0: {label_path}")
                values = [float(value) for value in tokens]
                rows.append(values)

        if not rows:
            return torch.empty((0, 5), dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)


def stage1_collate_fn(batch: Iterable):
    """Ghép ảnh thành tensor batch mà không pad hoặc gộp các target.

    Returns:
        Tuple gồm images shape [B, C, H, W] và tuple B target shape [N_i, 5].
        Giữ target tách biệt để tránh thêm bbox giả khi N_i khác nhau.
    """
    batch = list(batch)
    images = torch.stack([item[0] for item in batch])
    targets = tuple(item[1] for item in batch)
    return images, targets


def create_stage1_dataloader(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    split: str = "train",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    transform: Optional[Callable] = None,
) -> DataLoader:
    """Tạo DataLoader Stage 1 với sampler có thứ tự tái lập.

    Args:
        manifest_path: Manifest đầu vào.
        split: Tập cần đọc.
        batch_size: Số ảnh trong một batch.
        shuffle: Có xáo trộn thứ tự mẫu hay không; thường bật cho train.
        num_workers: Số worker đọc dữ liệu; mặc định 0 thuận tiện trên Windows.
        seed: Seed cho bộ sinh số ngẫu nhiên của DataLoader/sampler.
        transform: Transform ảnh được chuyển tiếp cho Stage1Dataset.

    Lưu ý:
        Seed cố định thứ tự lấy mẫu và đặt seed cho module random toàn cục
        của Python. Transform dùng nguồn ngẫu nhiên khác hoặc worker riêng
        vẫn cần cấu hình seed tương ứng để tái lập toàn bộ augmentation.
    """
    random.seed(seed)
    dataset = Stage1Dataset(
        manifest_path=manifest_path,
        split=split,
        transform=transform,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=stage1_collate_fn,
        generator=generator,
    )
