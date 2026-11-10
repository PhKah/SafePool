# SafePool

**SafePool — Edge AI Drowning Detection & Early Warning System**: Hệ thống thị giác máy tính và học sâu phát hiện nguy cơ đuối nước trên thiết bị biên.

## Tài liệu đặc tả (Specs)

- [Danh mục tài liệu đặc tả](Specs/README.md)
- [Đặc tả yêu cầu tối thiểu (`bare_minimum.md`)](Specs/bare_minimum.md)
- [Đặc tả bộ dữ liệu & EDA (`dataset_specs.md`)](Specs/dataset_specs.md)

---

## Dữ liệu (Dataset)

Dự án sử dụng bộ dữ liệu **Swimming and Drowning Detection** trên Roboflow Universe:

- **Liên kết duyệt dữ liệu:** https://universe.roboflow.com/university-g3h71/swimming-and-drowning-detection/browse?queryText=&pageSize=100&startingIndex=0&browseQuery=true
- **Trang dataset (v1):** https://universe.roboflow.com/university-g3h71/swimming-and-drowning-detection/dataset/1
- **Thông tin cơ bản:** `12,365` ảnh (`416×416`), **3 lớp** (`Drowning`, `Person out of water`, `Swimming`), giấy phép `CC BY 4.0`.
- **Định dạng yêu cầu:** **YOLOv8** (`yolov8`).

> [!NOTE]
> Thư mục `data/raw/` và các tệp nén (`*.zip`, `*.tar.gz`) đã được cấu hình trong [`.gitignore`](.gitignore) để không đẩy dữ liệu dung lượng lớn lên Git. Do đó, cần tải dữ liệu thủ công về máy trước khi huấn luyện hoặc chạy EDA.

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
