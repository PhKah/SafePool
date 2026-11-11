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
│   └── processed/    # [Local / Git-ignored] Ảnh ROI cắt cho Stage 2 (tự sinh bằng script)
├── checkpoints/      # [Local / Git-ignored] Trọng số mô hình .pt / .onnx khi train & export
├── src/              # [Tracked on Git] Mã nguồn datasets, models, engine, inference
├── scripts/          # [Tracked on Git] Script chuẩn bị dữ liệu & export ONNX
├── tests/            # [Tracked on Git] Unit test kiểm tra split (0% leakage), shape, metrics
└── Specs/            # [Tracked on Git] Đặc tả kỹ thuật, roadmap, experiment_log & run_logs
```

> [!NOTE]
> Thư mục `data/raw/`, `data/processed/`, `checkpoints/` và các tệp trọng số/nén (`*.pt`, `*.onnx`, `*.zip`) nằm trong [`.gitignore`](.gitignore) nên không được đẩy lên Git. Thành viên mới sau khi `git clone` chỉ cần tải `data/raw/` theo hướng dẫn dưới đây và chạy script tiền xử lý (sẽ sinh `data/processed/` từ `data/raw/` + `data/splits/`).

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
2. Cài đặt thư viện `roboflow` và chạy đoạn mã sau từ thư mục gốc `SafePool/`:

```bash
pip install roboflow
```

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
