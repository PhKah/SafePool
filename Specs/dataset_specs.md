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

## 5. Phân Bố Nguồn & Đặc Thù Gom Nhóm Dữ Liệu (Group Hierarchy & Splits)

Phân tích trực tiếp trên toàn bộ **12,365 tệp ảnh** và **23,021 bounding boxes** tại [`data/raw/`](file:///Users/phamkhanh/Documents/2026.1/DL/Practice/SafePool/data/raw) xác định cấu trúc nguồn thực tế của dữ liệu:

### 5.1. Hai cấp độ nhóm nguồn (Source Hierarchy)
1. **Cấp độ 1 (Frame-level trước `.rf.`):**
   - Toàn bộ `12,365` ảnh thực chất chỉ xuất phát từ **`1,932` tiền tố frame duy nhất**.
   - Nhiều ảnh là các bản sao augmentation (lật, xoay $\pm 15^\circ$, đổi độ phơi sáng) sinh ra từ cùng 1 frame gốc.
   - *Ở split mặc định của Roboflow:* Có tới **91.1%** ảnh `valid/` (`1,347/1,478`) và **89.4%** ảnh `test/` (`668/747`) dùng chung tiền tố ảnh với `train/` (rò rỉ dữ liệu chéo).
2. **Cấp độ 2 (Video-level theo clip gốc):**
   - Có **35 video roots** chính (nhận diện qua chuỗi khung hình liên tiếp `_mp4`, `-mp4`, `_mov`, `_avi`, `clipchamp`). Khi gom cả các frame liên tiếp của cùng 1 video về chung nhóm, toàn bộ dữ liệu rút gọn còn **`1,363` nhóm độc lập**.
   - **Video siêu lớn (`drowning1_mp4`):** Riêng clip này chứa **1,048 ảnh** (~8.5% toàn bộ dữ liệu), gồm `1,299` box `Drowning`, `1,146` box `Swimming` và **0** box `Person out of water`.

### 5.2. Thống kê hai cấu hình phân chia tập dữ liệu (Dataset Splits)

| Tập dữ liệu | Cấu hình Split Mặc Định (Roboflow Leakage) | Cấu hình Split Sạch Theo Nhóm (Clean Group Split) |
| :--- | :---: | :---: |
| **Train** | 10,140 ảnh (82.0%) \| 18,752 boxes (C0: 10,867, C1: 616, C2: 7,269) | **8,655 ảnh (70.0%)** \| 16,321 boxes (C0: 8,347, C1: 549, C2: 7,425) |
| **Validation** | 1,478 ảnh (12.0%) \| 2,748 boxes (C0: 1,577, C1: 98, C2: 1,073) | **1,855 ảnh (15.0%)** \| 2,987 boxes (C0: 1,999, C1: 118, C2: 870) |
| **Test** | 747 ảnh (6.0%) \| 1,521 boxes (C0: 787, C1: 71, C2: 663) | **1,855 ảnh (15.0%)** \| 3,713 boxes (C0: 2,885, C1: 118, C2: 710) |
| **Tổng cộng** | **12,365 ảnh (100%) \| 23,021 boxes** | **12,365 ảnh (100%) \| 23,021 boxes** |
| **Tỷ lệ rò rỉ** | **91.1% Val & 89.4% Test trùng nguồn Train** | **0.0% rò rỉ (hoàn toàn độc lập theo nhóm)** |

### 5.3. Phân bố kích thước Bounding Box (Tỷ lệ mục tiêu nhỏ)
- Trên hệ tọa độ $416 \times 416$ (`23,021` boxes): 
  - Chiều rộng: `min = 2.0 px`, `p10 = 26.0 px`, `median = 78.0 px`, `p90 = 204.0 px`, `max = 416.0 px`.
  - Chiều cao: `min = 2.5 px`, `p10 = 30.0 px`, `median = 100.0 px`, `p90 = 229.0 px`, `max = 416.0 px`.
- Có **`3,335 / 23,021` (14.5%)** box nhỏ chuẩn COCO (`area < 32×32 px`) và **`531 / 23,021` (2.3%)** box có cạnh ngắn `< 16 px`.

---

## 6. Đặc Tả Định Dạng & Cấu Trúc Tệp (Format Specifications & Edge Cases)

### 6.1. Định dạng tệp dữ liệu
- **Tệp ảnh:** 100% tệp là định dạng `.jpg`, kích thước đồng nhất $416 \times 416$.
- **Tệp nhãn:** 100% tệp là định dạng văn bản UTF-8 (`.txt`) cùng tên với file ảnh tương ứng, mỗi dòng mô tả một bounding box theo chuẩn YOLO: `<class_id> <x_center> <y_center> <width> <height>` với tọa độ chuẩn hóa $[0, 1]$.
- **Không gian nhãn:** $C \in \{0, 1, 2\}$ với `0: Drowning`, `1: Person out of water`, `2: Swimming`. Không có nhãn rác hay class ID ngoài tập hợp này.

### 6.2. Cấu trúc quy ước đặt tên & Trường hợp biên (Naming Conventions & Edge Cases)
1. **Cấu trúc tên tệp:** Toàn bộ `12,365` tệp đều tuân theo mẫu `<source_prefix>.rf.<hash>.jpg`.
2. **Tiền tố có ký tự gạch ngang (`-`):** Có **116 tệp** có tên bắt đầu bằng ký tự `-` (ví dụ: `-Clipchamp-_mp4-35_jpg.rf.xxx.jpg`), đều thuộc nhóm video biên tập từ clipchamp.
3. **Phần mở rộng lồng nhau:** Các tiền tố frame chứa sẵn chuỗi `_jpg`, `_mp4`, `_mov` trước dấu tách `.rf.` (ví dụ: `youtube-2_jpg.rf.xxx.jpg`, `12_mp4-100_jpg.rf.xxx.jpg`). Điểm phân tách định danh duy nhất là chuỗi `.rf.`.
