"""
src/datasets/stage2_dataset.py
==============================
WP-1.3 — PyTorch Dataset & DataLoader cho Stage 2: 3-Class ROI Classification.

Ánh xạ nhãn 3 trạng thái:
    0: Drowning (Đang đuối nước)
    1: Person out of water (Người trên bờ / ngoài nước)
    2: Swimming (Đang bơi lội an toàn)

Cấu trúc thư mục dữ liệu đầu vào:
    data/processed/{train,valid,test}/{0,1,2}/*.jpg

Tính năng chính:
    - Stage2ROIDataset: PyTorch Dataset chuẩn, trả về (image_tensor [3, H, W], label_int).
    - get_stage2_dataloaders(): Factory function sinh DataLoader cho cả 3 tập train/valid/test.
    - compute_smoothed_class_weights(): Tính trọng số làm mịn w_c = sqrt(N_max / N_c).
    - class_weights_to_tensor(): Chuyển đổi trọng số sang PyTorch Tensor cho CrossEntropyLoss.
"""

import os
import math
import random
from typing import Optional, Dict, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# =============================================================================
# Hằng số định danh 3 lớp
# =============================================================================
CLASS_NAMES = {
    0: "Drowning",
    1: "Person out of water",
    2: "Swimming",
}
NUM_CLASSES = 3
DEFAULT_IMG_SIZE = 128


# =============================================================================
# PyTorch Dataset cho ROI
# =============================================================================
class Stage2ROIDataset(Dataset):
    """
    PyTorch Dataset nạp ảnh ROI từ cấu trúc thư mục phân lớp data/processed/{split}/{0,1,2}/.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        img_size: int = DEFAULT_IMG_SIZE,
    ):
        """
        Args:
            root_dir: Đường dẫn tới thư mục split (ví dụ: data/processed/train).
            transform: Chuỗi biến đổi ảnh torchvision transforms.
                       Nếu None: Resize(img_size, img_size) + ToTensor().
            img_size: Độ phân giải chuẩn hóa (mặc định 128x128).
        """
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # [0, 255] uint8 -> [0.0, 1.0] float32 shape [3, H, W]
            ])

        self.samples: List[Tuple[str, int]] = []
        self.class_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Thư mục không tồn tại: {root_dir}")

        for cls_id in range(NUM_CLASSES):
            cls_dir = os.path.join(root_dir, str(cls_id))
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    fpath = os.path.join(cls_dir, fname)
                    self.samples.append((fpath, cls_id))
                    self.class_counts[cls_id] += 1

        if len(self.samples) == 0:
            raise ValueError(f"Không tìm thấy ảnh ROI nào trong {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_tensor = self.transform(img)
        return img_tensor, label

    def get_class_counts(self) -> Dict[int, int]:
        """Trả về số lượng mẫu thực tế phân bố theo 3 lớp."""
        return dict(self.class_counts)

    def __repr__(self) -> str:
        return (
            f"Stage2ROIDataset(root='{self.root_dir}', "
            f"total_samples={len(self)}, "
            f"counts={self.class_counts})"
        )


# =============================================================================
# Hàm tính Smoothed Class Weights
# =============================================================================
def compute_smoothed_class_weights(
    class_counts: Dict[int, int],
) -> Dict[int, float]:
    """
    Tính trọng số lớp làm mịn: w_c = sqrt(N_max / N_c).
    
    Lý do kỹ thuật:
        Tập dữ liệu SafePool có lớp 1 (Person out of water) chỉ chiếm ~3.3% ở tập train.
        Nếu dùng Inverse Frequency chuẩn (N_total / (C * N_c)), trọng số lớp 1 sẽ bị thổi phồng
        gấp ~17.6 lần lớp 0, gây mất ổn định gradient (gradient explosion) và kéo sụt Precision.
        Căn bậc 2 (square root smoothing) giúp nén tỷ lệ chênh lệch xuống còn ~3.9x,
        vừa hỗ trợ lớp thiểu số học tốt vừa bảo vệ tính hội tụ của hàm Cross-Entropy Loss.
    """
    if not class_counts:
        return {}
    n_max = max(class_counts.values())
    weights = {}
    for cls_id in range(NUM_CLASSES):
        n_c = class_counts.get(cls_id, 0)
        if n_c > 0:
            weights[cls_id] = round(math.sqrt(n_max / n_c), 4)
        else:
            weights[cls_id] = 1.0
    return weights


def class_weights_to_tensor(
    class_counts: Dict[int, int],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Chuyển dictionary trọng số thành PyTorch Tensor 1D [NUM_CLASSES]
    để nạp trực tiếp vào `torch.nn.CrossEntropyLoss(weight=...)`.
    """
    w_dict = compute_smoothed_class_weights(class_counts)
    w_list = [w_dict.get(i, 1.0) for i in range(NUM_CLASSES)]
    tensor = torch.tensor(w_list, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


# =============================================================================
# DataLoader Generator & Worker Seed (Tái lập kết quả)
# =============================================================================
def _worker_init_fn(worker_id: int):
    """Cố định seed ngẫu nhiên độc lập cho từng worker tiến trình phụ."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)


def get_stage2_dataloaders(
    processed_dir: str = "data/processed",
    batch_size: int = 32,
    img_size: int = DEFAULT_IMG_SIZE,
    num_workers: int = 0,
    seed: int = 42,
    train_transform: Optional[transforms.Compose] = None,
    eval_transform: Optional[transforms.Compose] = None,
) -> Dict[str, DataLoader]:
    """
    Hàm tạo DataLoader cho cả 3 split: train, valid, test.

    Returns:
        Dict chứa 3 DataLoaders tương ứng với các khóa 'train', 'valid', 'test'.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    loaders = {}
    for split_name in ["train", "valid", "test"]:
        split_dir = os.path.join(processed_dir, split_name)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục: {split_dir}")

        transform = train_transform if split_name == "train" else eval_transform
        dataset = Stage2ROIDataset(
            root_dir=split_dir,
            transform=transform,
            img_size=img_size,
        )

        is_train = (split_name == "train")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=False,
            worker_init_fn=_worker_init_fn if is_train else None,
            generator=generator if is_train else None,
        )
        loaders[split_name] = loader

    return loaders
