# SafePool

**SafePool — Edge AI Drowning Detection & Early Warning System**: Hệ thống thị giác máy tính và học sâu phát hiện nguy cơ đuối nước trên thiết bị biên.

## Tài liệu đặc tả (Specs)

- [Danh mục tài liệu đặc tả](Specs/README.md)
- [Đặc tả yêu cầu tối thiểu (`bare_minimum.md`)](Specs/bare_minimum.md)
- [Đặc tả bộ dữ liệu & EDA (`dataset_specs.md`)](Specs/dataset_specs.md)
- [Lộ trình 5 giai đoạn 2-Stage From Scratch (`roadmap.md`)](Specs/roadmap.md)
- [Nhật ký thực nghiệm nhóm (`experiment_log.md`)](Specs/experiment_log.md)

---

## Cấu trúc thư mục dự án

Cây thư mục quy ước khi làm việc trên máy cá nhân / Google Colab (bao gồm cả các thư mục dữ liệu không đẩy lên Git):

```text
SafePool/
├── .github/          # [Tracked on Git] Mẫu GitHub Issue Forms cho nhóm
├── data/
│   ├── raw/          # [Local / Git-ignored] Ảnh gốc tải từ Roboflow (xem mục bên dưới)
│   ├── splits/       # [Tracked on Git] Bộ chia tập sạch (clean Group Split) dùng chung
│   ├── stage1/       # [Local / Git-ignored] Nhãn 1-lớp person & manifest Stage 1 (tự sinh bằng script)
│   └── processed/    # [Local / Git-ignored] Ảnh ROI cắt cho Stage 2 (tự sinh bằng script)
├── checkpoints/      # [Local / Git-ignored] Trọng số mô hình .pt / .onnx khi train & export
├── src/              # [Tracked on Git] Mã nguồn datasets, models, engine, inference
├── scripts/          # [Tracked on Git] Script chuẩn bị dữ liệu, sanity check & export ONNX
├── tests/            # [Tracked on Git] Unit test kiểm tra split (0% leakage), shape, metrics
└── Specs/            # [Tracked on Git] Đặc tả kỹ thuật, roadmap, experiment_log & run_logs
```

> [!NOTE]
> Thư mục `data/raw/`, `data/stage1/`, `data/processed/`, `checkpoints/` và các tệp trọng số/nén (`*.pt`, `*.onnx`, `*.zip`) nằm trong [`.gitignore`](.gitignore) nên không được đẩy lên Git. Thành viên mới sau khi `git clone` chỉ cần tải `data/raw/` theo hướng dẫn dưới đây và chạy các script tiền xử lý ở mục **Hướng dẫn chạy & Tái lập nhanh**.

---

## Dữ liệu (Dataset)

Dự án sử dụng bộ dữ liệu **Swimming and Drowning Detection** trên Roboflow Universe:

- **Liên kết duyệt dữ liệu:** https://universe.roboflow.com/university-g3h71/swimming-and-drowning-detection/browse?queryText=&pageSize=100&startingIndex=0&browseQuery=true
- **Trang dataset (v1):** https://universe.roboflow.com/university-g3h71/swimming-and-drowning-detection/dataset/1
- **Thông tin cơ bản:** `12,365` ảnh (`416×416`), **3 lớp** (`Drowning`, `Person out of water`, `Swimming`), giấy phép `CC BY 4.0`.
- **Định dạng yêu cầu:** **YOLOv8** (`yolov8`).

### Hướng dẫn tải dữ liệu (Định dạng YOLOv8)

#### Cách 1: Tải trực tiếp qua giao diện Web (Khuyên dùng)

1. Truy cập trang dataset trên Roboflow Universe:  
   [Swimming and Drowning Detection (Roboflow Universe)](https://universe.roboflow.com/university-g3h71/swimming-and-drowning-detection/browse?queryText=&pageSize=100&startingIndex=0&browseQuery=true)
2. Đăng nhập (hoặc đăng ký miễn phí) tài khoản **Roboflow**.
3. Chuyển sang mục **Dataset** (hoặc nhấn nút **Download Dataset**), chọn phiên bản **`v1`** (`2023-06-11 3:33pm`).
4. Tại hộp thoại tùy chọn tải xuống:
   - Mục **Format**: chọn **`YOLOv8`**.
   - Chọn **download zip to computer** rồi nhấn **Continue** để tải tệp `.zip` về máy.
5. Giải nén toàn bộ nội dung tệp `.zip` vào thư mục `data/raw/` trong repository.

#### Cách 2: Tải bằng Python (`roboflow` SDK)

1. Tại bước chọn định dạng **`YOLOv8`** trên Roboflow, chọn **show download code** để lấy `api_key` của tài khoản.
2. Cài đặt các thư viện trong `requirements.txt` từ thư mục gốc `SafePool/`:

```bash
pip install -r requirements.txt
```

`requirements.txt` bao gồm các thư viện dữ liệu, PyTorch, kiểm thử và tiện ích cho huấn luyện/demo. Backend CUDA không được cố định trong file này; nếu dùng GPU NVIDIA, hãy cài bản PyTorch tương thích với phiên bản CUDA/driver theo hướng dẫn cài đặt chính thức của PyTorch, rồi kiểm tra `torch.cuda.is_available()`.

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("university-g3h71").project("swimming-and-drowning-detection")
version = project.version(1)
dataset = version.download("yolov8", location="data/raw")
```

### Cấu trúc thư mục sau khi tải

Sau khi tải và giải nén đúng chuẩn **YOLOv8**, cấu trúc trong `data/raw/` sẽ như sau:

```text
SafePool/
└── data/
    └── raw/
        ├── data.yaml
        ├── README.dataset.txt
        ├── README.roboflow.txt
        ├── train/          # 10,140 ảnh
        │   ├── images/
        │   └── labels/
        ├── valid/          # 1,478 ảnh
        │   ├── images/
        │   └── labels/
        └── test/           # 747 ảnh
            ├── images/
            └── labels/
```

Xem chi tiết phân bố lớp và thống kê EDA tại [`Specs/dataset_specs.md`](Specs/dataset_specs.md).

---

## Hướng dẫn chạy & Tái lập nhanh (Quickstart)

Sau khi đã giải nén dữ liệu gốc vào `data/raw/`, thực hiện lần lượt từ thư mục gốc `SafePool/`:

### 1. Sinh dữ liệu sạch cho Stage 1 & Stage 2 (từ `clean Group Split`)

```bash
# Stage 1: Sinh nhãn 1-lớp (0: person) & data/stage1/manifest.json
python scripts/datasets/stage1/prepare_stage1_dataset.py

# Stage 2: Cắt ảnh ROI 128x128 (+20% ngữ cảnh) vào data/processed/
python scripts/prepare_roi_dataset.py
```

### 2. Chạy bộ kiểm thử tự động & Gate 2 Sanity Check

```bash
# Chạy toàn bộ unit tests của dự án
python -m pytest -v

# Chạy kiểm tra overfit 1 batch (Gate 2) cho cả Stage 2 và Stage 1
python -m scripts.sanity_check --stage all --sanity-check
```

### 3. Huấn luyện & Đánh giá Stage 1 — Person Localizer (`D1` Baseline)

```bash
# Huấn luyện đầy đủ mốc S1-01 (lưu trọng số tốt nhất tại checkpoints/stage1_d1_best.pt
# và tự động đối chiếu Standard NMS vs Gaussian Soft-NMS trên tập valid sạch)
python -m src.train_stage1

# Chạy thử nhanh vài batch (smoke test ~10 giây, không ghi đè checkpoint chính)
python -m src.train_stage1 --epochs 2 --max-batches 5 \
    --checkpoint checkpoints/tmp_s1.pt --summary checkpoints/tmp_s1.json

# Kiểm thử cắt lát đa chiều (Slicing Metrics) & xuất 30 mẫu lỗi tệ nhất trên tập valid sạch
python -m scripts.audit_stage1_baseline
```

### 4. Huấn luyện & Đánh giá Stage 2 — ROI Behavior Classifier (`M1` Baseline)

*(Đang triển khai tại nhánh `WP-3.1` — bổ sung lệnh chạy ngay khi hợp nhất).*
