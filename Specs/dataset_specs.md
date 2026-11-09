# SafePool: Đặc Tả Bộ Dữ Liệu Thực Tế (Dataset Specification & EDA)

> **Mã dự án:** SafePool — Edge AI Drowning Detection  
> **Nguồn dữ liệu:** Roboflow Universe (`university-g3h71/swimming-and-drowning-detection/1`)  
> **Vị trí lưu trữ cục bộ:** [`data/raw/`](file:///Users/phamkhanh/Documents/2026.1/DL/Practice/SafePool/data/raw)  
> **Định dạng:** Chuẩn YOLO Annotation Format (`.txt`)  
> **Kích thước ảnh:** Đồng nhất **$416 \times 416$ pixels** (chuẩn Edge AI)  

---

## 1. Tổng Quan Phân Bố Dữ Liệu (Dataset Splits & Volume)

Bộ dữ liệu chứa tổng cộng **12,365 ảnh** với kịch bản thực tế nhiều người cùng bơi trong bể (tối đa lên đến 21 người/khung hình):

| Tập dữ liệu (Split) | Số lượng Ảnh | Độ phân giải | Tổng số Bounding Box | Số người TB / Ảnh | Tỷ lệ dữ liệu |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | **10,140** | $416 \times 416$ | **18,752** | ~1.85 người | 82.0% |
| **Validation** | **1,478** | $416 \times 416$ | **2,748** | ~1.86 người | 12.0% |
| **Test** | **747** | $416 \times 416$ | **1,521** | ~2.03 người | 6.0% |
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

Khác với tập dữ liệu nhân tạo chỉ có 1 người, tập dữ liệu này phản ánh không gian bể bơi thực tế:
- **Ảnh có 1 người:** 6,656 ảnh (~65.6%)
- **Ảnh có từ 2 – 5 người:** 3,182 ảnh (~31.4%)
- **Ảnh đông người (6 – 21 người):** 302 ảnh (~3.0%)

> [!IMPORTANT]
> **Yêu cầu đối với kiến trúc mô hình:**
> - Mô hình phải là **Object Detection đa mục tiêu (Multi-object Detection)**, không thể dùng mạng Classification toàn khung hình (Full-image classifier).
> - Phải áp dụng **Non-Maximum Suppression (NMS)** hoặc **Soft-NMS** hiệu quả để tách biệt các nạn nhân bơi cạnh nhau mà không bị triệt tiêu nhầm box.

---

## 4. Tương Thích Với Giới Hạn Edge AI

1. **Độ phân giải $416 \times 416$:**
   - Kích thước này rất tối ưu cho vi xử lý biên (Jetson Nano, Raspberry Pi 4/5, Intel NUC, Edge PC).
   - So với $640 \times 640$, ảnh $416 \times 416$ giúp giảm $(\frac{416}{640})^2 \approx 42\%$ khối lượng tính toán (FLOPs), hỗ trợ đạt mục tiêu **$\ge 30$ FPS** và **FLOPs $< 9.0$ GFLOPs**.
2. **Kỹ thuật Augmentation gốc từ Roboflow:**
   - Đã áp dụng xoay ngẫu nhiên $\pm 15^\circ$, điều chỉnh độ phơi sáng $\pm 15\%$, lật ngang/dọc $50\%$.
   - **Cần bổ sung trong training pipeline:** Thêm các augmentation xử lý nhiễu nước đặc thù (mô phỏng chói nắng, bọt nước, vệt mờ chuyển động *Motion Blur*).
