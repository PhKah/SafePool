# SafePool: Đặc Tả Bộ Dữ Liệu Thực Tế (Dataset Specification & EDA)

> **Mã dự án:** SafePool — Edge AI Drowning Detection  
> **Nguồn dữ liệu:** Roboflow Universe (`university-g3h71/swimming-and-drowning-detection/1`)  
> **Vị trí lưu trữ cục bộ:** [`data/raw/`](file:///Users/phamkhanh/Documents/2026.1/DL/Practice/SafePool/data/raw)  
> **Định dạng:** Chuẩn YOLO Annotation Format (`.txt`)  
> **Kích thước ảnh:** Đồng nhất **$416 \times 416$ pixels** (chuẩn Edge AI)  

---

## 1. Tổng Quan Phân Bố Dữ Liệu (Dataset Splits & Volume)

Bộ dữ liệu chứa tổng cộng **12,365 ảnh** với kịch bản thực tế nhiều người cùng bơi trong bể (tối đa lên đến 22 người/khung hình):

| Tập dữ liệu (Split) | Số lượng Ảnh | Độ phân giải | Tổng số Bounding Box | Số người TB / Ảnh | Tỷ lệ dữ liệu |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | **10,140** | $416 \times 416$ | **18,752** | ~1.85 người | 82.0% |
| **Validation** | **1,478** | $416 \times 416$ | **2,748** | ~1.86 người | 12.0% |
| **Test** | **747** | $416 \times 416$ | **1,521** | ~2.04 người | 6.0% |
| **Tổng cộng** | **12,365** | $416 \times 416$ | **23,021** | **100% có gán nhãn** | 100% |

---

## 2. Phân Bố Lớp Nhãn & Thách Thức Mất Cân Bằng (Class Imbalance)

Theo tệp cấu hình [`data/raw/data.yaml`](file:///Users/phamkhanh/Documents/2026.1/DL/Practice/SafePool/data/raw/data.yaml), bộ dữ liệu gồm **3 lớp nhãn**:

| Class ID | Tên Lớp (Label) | Train BBoxes | Tỷ lệ Train | Val BBoxes | Tỷ lệ Val | Test BBoxes | Tỷ lệ Test | Đánh giá & Rủi ro |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | **`Drowning`** | 10,867 | **57.95%** | 1,577 | **57.39%** | 787 | **51.74%** | **Lớp trọng tâm sống còn.** Cần tối ưu Recall $\ge 90\%$. |
| **1** | **`Person out of water`** | 616 | **3.28%** | 98 | **3.57%** | 71 | **4.67%** | **Thiểu số cực đoan (Extreme Minority).** Nguy cơ bị mạng bỏ qua nếu dùng Loss thông thường. |
| **2** | **`Swimming`** | 7,269 | **38.76%** | 1,073 | **39.05%** | 663 | **43.59%** | Trạng thái bình thường, cần phân biệt rõ với `Drowning` để tránh báo động giả. |

> [!WARNING]
> **Hiện tượng mất cân bằng lớp (Class Imbalance):**
> Lớp `Person out of water` chỉ chiếm **~3.3%** tổng số nhãn, trong khi `Drowning` chiếm tới **~58%**. 
> - **Hệ quả:** Nếu sử dụng hàm mất mát Cross-Entropy tiêu chuẩn và đo bằng *Accuracy*, mô hình có thể đạt Accuracy cao ảo nhưng bỏ qua hoàn toàn người ở bờ bể.
> - **Giải pháp kỹ thuật bắt buộc:** Áp dụng **Focal Loss** ($\gamma = 2.0, \alpha$) hoặc **Class-weighted Loss** để bù trừ trọng số cho lớp thiểu số, đồng thời đánh giá bằng **Per-class Recall & mAP@50**.

---

## 3. Đặc Thù Nhiều Mục Tiêu / Khung Hình (Multi-Target Distribution)

Khác với tập dữ liệu nhân tạo chỉ có 1 người, tập dữ liệu này phản ánh không gian bể bơi thực tế (trên toàn bộ **12,365 ảnh**, có **4,310 ảnh — ~34.86%** chứa từ 2 người trở lên):
- **Ảnh có 1 người:** **8,055** ảnh (~65.1% — Train: 6,656 | Val: 945 | Test: 454)
- **Ảnh có từ 2 – 5 người:** **3,935** ảnh (~31.8% — Train: 3,182 | Val: 486 | Test: 267)
- **Ảnh đông người (6 – 22 người):** **375** ảnh (~3.0% — Train: 302 | Val: 47 | Test: 26; tối đa 21 người ở Train và 22 người ở Test)

> [!IMPORTANT]
> **Yêu cầu đối với kiến trúc mô hình:**
> - Không thể dùng mạng Image Classification đơn lẻ trên toàn khung hình: cần chọn **Object Detection đa mục tiêu (1-Stage)** trên ảnh $416 \times 416$ hoặc **2-Stage Pipeline** (Stage 1 định vị người $\rightarrow$ Stage 2 phân loại ROI có mở rộng padding ngữ cảnh).
> - Phải áp dụng **Non-Maximum Suppression (NMS)** hoặc **Soft-NMS** hiệu quả để tách biệt các nạn nhân bơi cạnh nhau mà không bị triệt tiêu nhầm box.

---

## 4. Tương Thích Với Giới Hạn Edge AI

1. **Độ phân giải $416 \times 416$:**
   - Kích thước này rất tối ưu cho vi xử lý biên (Jetson Nano, Raspberry Pi 4/5, Intel NUC, Edge PC).
   - So với $640 \times 640$, ảnh $416 \times 416$ giúp giảm $(\frac{416}{640})^2 \approx 42.25\%$ khối lượng tính toán (FLOPs), hỗ trợ đạt mục tiêu **$\ge 30$ FPS** và **FLOPs $< 9.0$ GFLOPs** (lưu ý nếu dùng 2-Stage thì tổng FLOPs bằng `Stage 1 + N × Stage 2` với $N \le 22$).
   - **Lưu ý tiền xử lý Roboflow:** Theo [`data/raw/README.roboflow.txt`](file:///Users/phamkhanh/Documents/2026.1/DL/Practice/SafePool/data/raw/README.roboflow.txt), ảnh đã được `Resize to 416x416 (Stretch)` (kéo giãn không giữ tỷ lệ khung hình).
2. **Kỹ thuật Augmentation gốc từ Roboflow:**
   - Đã áp dụng xoay ngẫu nhiên $\pm 15^\circ$, điều chỉnh độ phơi sáng $\pm 15\%$, lật ngang/dọc $50\%$ (tạo 2 biến thể mỗi ảnh nguồn).
   - **Cần bổ sung trong training pipeline:** Thêm các augmentation xử lý nhiễu nước đặc thù (mô phỏng chói nắng *glare* cục bộ, bọt nước, *Motion / Gaussian Blur* nhẹ với kernel nhỏ `3×3`). **Không bật thêm `Vertical Flip`** trong PyTorch vì tập gốc đã lật dọc 50%.

---

## 5. Kiểm Định Chất Lượng Dữ Liệu Thực Tế (Leakage & BBox Scale Audit)

Kiểm tra trực tiếp trên toàn bộ **12,365 tệp ảnh** và **23,021 bounding boxes** tại [`data/raw/`](file:///Users/phamkhanh/Documents/2026.1/DL/Practice/SafePool/data/raw) ghi nhận hai đặc điểm quan trọng cần xử lý trước khi huấn luyện:

1. **Rò rỉ dữ liệu giữa các tập (Cross-split Data Leakage):**
   - Toàn bộ `12,365` ảnh thực chất chỉ xuất phát từ **`1,932` tiền tố ảnh nguồn** (phần tên trước `.rf.`): `Train` có `1,790` tiền tố (một số ảnh gốc lặp tới `36–58` bản sao), `Validation` có `941`, `Test` có `564`.
   - Có tới **`1,347 / 1,478` (91.1%)** ảnh `Validation` và **`668 / 747` (89.4%)** ảnh `Test` dùng chung tiền tố ảnh nguồn với tập `Train`.
   - Kiểm tra SHA-256 ghi nhận `12,337` hash ảnh duy nhất (`10` hash ảnh giống hệt nhau nằm ở nhiều split) và `10,048` hash nhãn duy nhất (`669` tệp nhãn `.txt` trùng khớp tuyệt đối nằm ở nhiều split).
   - **Hệ quả & giải pháp:** Điểm đánh giá trên bộ split mặc định của Roboflow sẽ bị **cao ảo**. Trong thực nghiệm (Giai đoạn 1 & 4), cần tạo thêm **Group Split sạch theo tiền tố ảnh/video nguồn** để kiểm chứng khả năng tổng quát hóa thực sự.
2. **Phân bố kích thước Bounding Box (Tỷ lệ mục tiêu nhỏ):**
   - Trên hệ tọa độ $416 \times 416$ (`23,021` boxes): chiều rộng `min = 2.0 px`, `p10 = 26.0 px`, `median = 78.0 px`, `p90 = 204.0 px`, `max = 416.0 px`; chiều cao `min = 2.5 px`, `p10 = 30.0 px`, `median = 100.0 px`, `p90 = 229.0 px`, `max = 416.0 px`.
   - Có **`3,335 / 23,021` (14.5%)** box nhỏ chuẩn COCO (`area < 32×32 px`) và **`531 / 23,021` (2.3%)** box có cạnh ngắn `< 16 px`.
   - **Hệ quả & giải pháp:** Nếu chọn hướng 2-Stage crop ROI ($128 \times 128$), cần đặt ngưỡng lọc box quá nhỏ (`min_box_size`) để tránh ảnh vỡ nhòe, mở rộng padding ngữ cảnh `15–25%` để giữ thông tin mặt nước/thành bể, và giữ cường độ blur/glare ở mức nhẹ.
