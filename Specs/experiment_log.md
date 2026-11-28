# SafePool — Nhật Ký Thực Nghiệm Chính Thức (Master Experiment Log)

> Tuân thủ **Nguyên tắc 2** trong [`bare_minimum.md`](bare_minimum.md): Mọi thay đổi về kiến trúc, siêu tham số, hàm mất mát hay tiền xử lý/augmentation đều phải được ghi lại minh bạch kèm lý do kỹ thuật và kết quả trên **clean Group Split**.  
> **Phạm vi Git:** Tài liệu này và thư mục [`Specs/run_logs/`](run_logs/_template.md) được theo dõi trên Git để cả nhóm đồng bộ; file trọng số nặng (`*.pt`, `*.onnx`, `*.pth`) giữ cục bộ/Drive theo [`.gitignore`](../.gitignore).

---

## 1. Phân Quyền & Quy Trình Thực Nghiệm Nhóm (2-Tier Git Workflow)

| Thành phần | Quyền chỉnh sửa | Quy tắc kiểm soát |
| :--- | :--- | :--- |
| **Run Log cá nhân** (`Specs/run_logs/<RunID>_<ten>_<ten-ngan>.md`) | **Mọi thành viên** | **Chỉ thêm mới (*Append-only*):** Ai chạy người đó tạo file từ [`_template.md`](run_logs/_template.md). Không xóa hay sửa đè số liệu run cũ; chạy lại $\rightarrow$ cấp ID mới (`S2-02b`). |
| **Bảng Tổng Hợp Chính** (*Master Table* bên dưới) | **Chỉ PM / Tech Lead** *(hoặc merge PR sau khi Lead duyệt)* | **Cổng kiểm duyệt:** Chỉ các run chạy đúng `clean Group Split`, đúng `1` yếu tố thay đổi và đạt DoD mới được đưa vào Master Table và cập nhật mốc **Current Best**. |

### Quy trình 5 bước khi thành viên tự thử nghiệm
1. **Claim Run ID & Chốt giả thuyết có thể bác bỏ:** Kiểm tra WIP ($\le 1\text{--}2$ việc/người), báo lên nhóm để lấy Run ID kế tiếp (`S2-xx` / `S1-xx`), chốt **1 biến thay đổi duy nhất** và **ngưỡng Đạt / Bác bỏ định lượng trước khi chạy** dựa trên *Error Analysis* của Run gốc.
2. **Nhánh riêng & Pipeline cố định:** Tạo branch riêng (vd: `exp/S2-02-focal-loss`), chạy bằng script train/eval, `seed=42` & `clean Group Split` chuẩn chung của nhóm (tuyệt đối không đổi tập `valid` hay sửa công thức metric trong lúc thí nghiệm).
3. **Lưu log & Kiểm tra tại chỗ:** Tự động xuất cấu hình/lịch sử JSON kèm `commit hash`, `seed`, hyperparams, metrics tổng + *Slicing Metrics* và `Params / FLOPs / Size / FPS`. *(Giữ `.pt` tại local/Drive, không push lên Git).*
4. **Tạo Run Log & Mở PR:** Copy [`Specs/run_logs/_template.md`](run_logs/_template.md) thành file riêng của run đó, điền phần *Phân tích lỗi* và đề xuất **`Giữ (Accepted)`** hoặc **`Loại (Rejected)`**, rồi mở Pull Request.
5. **Duyệt & Cập nhật Master Table:** Lead đối chiếu kết quả thực tế với ngưỡng Đạt/Bác bỏ ban đầu và trần phần cứng để hợp nhất vào bảng dưới đây *(kể cả run `Loại (Rejected)` vẫn được ghi nhận minh bạch phục vụ bảng Ablation Study)*.

---

## 2. Nguyên Tắc Thực Nghiệm Khoa Học & Kiểm Soát Vòng Lặp (Giai Đoạn 3–4)

- **Bác bỏ giả thuyết bằng thực nghiệm (*Hypothesis-Driven & Empirical Falsification*):**
  - Mọi suy luận lý thuyết chỉ được xem là **manh mối**; bắt buộc đi đúng chuỗi `Soi 20–30 ca sai từ Run trước → Viết giả thuyết có ngưỡng Đạt/Bác bỏ định lượng TRƯỚC khi chạy → Kiểm thử cô lập 1 biến → Kết luận bằng số liệu`.
  - Nếu số liệu thực tế không đạt ngưỡng kỳ vọng hoặc gây tác dụng phụ lên nhóm an toàn sinh mạng, **bác bỏ ngay giải pháp (`Loại / Rejected`)** và giữ nguyên mốc `Current Best` cũ — không cố bảo vệ suy luận lý thuyết ban đầu.
- **Cô lập đơn biến (*Controlled Single-Variable Ablation*):** Mỗi Run ID chỉ thay đổi **1 yếu tố chính** (`Baseline` vs `Baseline + Module X`, giữ nguyên `clean Group Split` và `seed=42`) để giải trình chính xác nguyên nhân tăng/giảm hiệu năng.
- **Đánh giá đa chiều & Cổng loại trừ phần cứng (*Slicing Metrics & Hard Edge Gate*):**
  - Ngoài chỉ số tổng (`Macro F1` / `mAP@50`), bắt buộc đối chiếu các **phân khúc hiểm**: `0: Drowning` Recall & `FAR` (Stage 2); `Small-box Recall (<32²)`, `Crowded Recall (≥3 người)` & `Precision (@0.25)` (Stage 1).
  - Bất kỳ cải tiến nào làm **vượt trần ngân sách cứng của Stage đó** đều bị **loại bỏ**, dù tăng nhẹ độ chính xác:
    - **Stage 1 (`416×416`):** `Params ≤ 2.2M` | `Size ≤ 9.0 MB` (FP32) | `FLOPs ≤ 4.5 GFLOPs` | `Latency ≤ 18 ms`.
    - **Stage 2 (`128×128`/ROI):** `Params ≤ 1.3M` | `Size ≤ 5.0 MB` (FP32) | `FLOPs ≤ 0.18 GFLOPs/ROI` | `Latency ≤ 10 ms` (batch).
- **Nhịp chạy & Điểm dừng kỹ thuật (*Early-Stop, Diminishing Returns & Freeze*):**
  - **Nhịp chạy mục tiêu:** `30 phút – 2–3 giờ` / thử nghiệm; **Per-run Early Stopping:** `patience = 8–12 epochs` (kèm pha `warmup` khi cần cho `mAP@50`) trên `valid` sạch.
  - **Diminishing Returns:** Tối đa `3–5` giả thuyết trọng tâm sau Baseline `M1`/`D1`; dừng mở thêm slot nếu `2` vòng liên tiếp tăng `< +0.5%–1.0%` nhưng làm tăng FLOPs/Params.
  - **Model Freeze:** Chậm nhất **cuối Tuần 6** chốt trọng số tốt nhất thỏa mãn ngân sách Edge AI tổng (`Size < 15 MB`, `Params < 5.0M`, `FLOPs < 9.0 GFLOPs` với $N \le 22$, `≥ 30 FPS`).

---

## 3. Bảng Tổng Hợp — Stage 2: ROI Behavior Classifier

*Đánh giá trên `valid` (và chốt cuối trên `test`) của **clean Group Split** (`0: Drowning`, `1: Person out of water`, `2: Swimming`).*

| Run ID | Ngày | Owner | Phiên bản / WP | 1 Thay đổi chính | `Drowning` Recall | `Out-of-water` Recall | `Swimming` Recall | Macro F1 | FAR | Params / FLOPs / Size | Chi tiết Run | Trạng thái |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `S2-00` | 2026-11-23 | Khánh | `WP-2.2` Sanity | Mạng mồi `TinyStage2Probe`, 1 batch `12` ROI (`4/lớp`), no reg/aug, 50 ep | `100.0%` | `100.0%` | `100.0%` | `100.0%` | `0.0%` | `26.7K` / — / — | [`S2-00`](run_logs/S2-00_sanity_stage2.md) | **PASSED Gate 2** |
| `S2-01` | — | — | `WP-3.1` `M1` | Baseline CNN tối giản + GAP + `CrossEntropyLoss` | — | — | — | — | — | — | — | *Mốc Baseline* |

---

## 4. Bảng Tổng Hợp — Stage 1: Person Localizer (`0: person`, `416×416`)

*Gộp 3 lớp thành `0: person`, đánh giá trên **clean Group Split**.*

| Run ID | Ngày | Owner | Phiên bản / WP | 1 Thay đổi chính | Person Recall | `mAP@50` | Small-box Recall (`<32²`) | Params | FLOPs (`416²`) | Size (MB) | FPS | Chi tiết Run | Trạng thái |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `S1-00` | 2026-11-23 | Khánh | `WP-2.2` Sanity | Localizer mồi `TinyStage1Probe`, `2` ảnh (`7` GT), no reg/aug, 80 ep | `100.0%` | `100.0%` | — | `245.7K` | — | — | — | [`S1-00`](run_logs/S1-00_sanity_stage1.md) | **PASSED Gate 2** |
| `S1-01` | 2026-11-27 | Khánh | `WP-3.2` `D1` | Baseline CNN 1-scale `26×26` (Bounded Sigmoid + SmoothL1/GIoU + NMS, no aug) | `67.73%` | `40.23%` | `64.11%` | `1.07M` | `2.80 GFLOPs` | `4.11 MB` | `275.6` *(MPS)* | [`S1-01`](run_logs/S1-01_khanh_d1-baseline.md) | **Mốc Baseline (Current Best)** |

### 4.1. Kiểm Thử Đa Chiều & Chẩn Đoán Lỗi Mốc `S1-01` (`valid` — `1,855` ảnh / `2,987` GT)
- **Phân khúc hiểm (*Slicing Metrics @0.25*):** `Precision = 20.59%` (`5.30` box/ảnh) | `Sparse Recall (1–2 người) = 73.03%` vs **`Crowded Recall (≥3 người) = 59.49%`** | Theo nhãn hành vi gốc: **`0: Drowning = 67.68%`** (`1,353/1,999`), **`1: Out-of-water = 40.68%`** (`48/118`), **`2: Swimming = 71.49%`** (`622/870`).
- **Giải phẫu lỗi & Soi 30 mẫu tệ nhất** *(lệnh tái lập: `python -m scripts.audit_stage1_baseline`, chi tiết tại [`S1-01`](run_logs/S1-01_khanh_d1-baseline.md))*:
  - **`964` ca `FN`:** `59.3%` (`572` ca) đã bắt đúng vùng người nhưng lệch biên `0.25 ≤ IoU < 0.50`; `10.5%` (`101` ca) điểm thấp `< 0.25`; `30.2%` (`291` ca) sót hoàn toàn do **cảnh thiếu sáng tối đen**, **lóa nắng/lệch phổ màu** (khi chưa bật Augmentation) và **người siêu nhỏ `10–15 px` ở góc quay cao** (nhỏ hơn `stride = 16 px`).
  - **`7,800` hộp `FP`:** `44.5%` (`3,470` hộp) là chùm hộp vây quanh người thật; `55.5%` (`4,330` hộp) có `IoU < 0.10` (trong đó nhiều hộp `score ≥ 0.96` thực tế bắt trúng người thật bị gán thiếu nhãn ở cảnh phao bơi đông người, phần còn lại là nhiễu sóng lóa nắng & phao phân làn).

---

## 5. Đánh Giá Chuỗi End-to-End (`Stage 1 → NMS → Stage 2`)

*Đo trên tập `test` của **clean Group Split** (có thể ghi thêm cột tham chiếu split mặc định Roboflow).*

| Eval ID | Ngày | Stage 1 + Stage 2 | Threshold / NMS | E2E `Drowning` Recall | E2E Macro F1 | FAR | Tổng Params | Tổng FLOPs ($N=1$ / $N=22$) | Tổng Size | FPS (ONNX) | Kết luận |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `E2E-01` | — | `WP-4.end` Freeze | — | — | — | — | — | — | — | — | *Chờ chốt Tuần 6* |
