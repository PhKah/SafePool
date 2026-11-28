# Run <RUN_ID>: <Tên ngắn của thử nghiệm>

> **Quy tắc:** Sao chép mẫu này thành `Specs/run_logs/<RUN_ID>_<ten-tv>_<ten-ngan>.md` (ví dụ: `S2-02_minh_focal-loss.md`).  
> **Tính bất biến:** Không xóa hay sửa đè kết quả cũ; nếu chạy lại với cấu hình mới, tạo Run ID mới (`S2-02b`).

---

## 1. Thông Tin Định Danh & Giả Thuyết Kiểm Thử

- **Run ID:** `<RUN_ID>` *(vd: `S2-02` hoặc `S1-02`)*
- **Ngày chạy:** `YYYY-MM-DD`
- **Người thực hiện (Single Owner):** `<Tên thành viên>`
- **Liên kết Work Package:** `<WP-x.x / EXP-SLOT-k>`
- **Nhánh Git / Commit Hash:** `<branch> @ <short-sha>`
- **Kế thừa từ Run:** `<Base Run ID>` *(vd: `S2-01` hoặc `S1-01`)*
- **Điểm yếu quan sát từ Run gốc:** `<Trích số liệu / nhóm lỗi cụ thể từ Error Analysis của Run trước>`
- **1 thay đổi duy nhất so với Run gốc:** `<Mô tả ngắn đúng 1 yếu tố can thiệp; giữ cố định các thành phần còn lại>`
- **Giả thuyết có thể bác bỏ:** `<Lỗi [A] là do [B]. Nếu can thiệp [C] thì chỉ số [D] sẽ cải thiện mà không làm tổn hại [F]>`
- **Tiêu chí Đạt / Bác bỏ định lượng *(chốt TRƯỚC khi chạy)*:**
  - **Đạt (`Accepted`):** `<vd: mAP@50 tăng ≥ +2.0%, Person Recall giảm ≤ 1.0%, nằm trong trần FLOPs/Latency của Stage>`
  - **Bác bỏ (`Rejected`):** `<vd: không vượt ngưỡng trên, làm tụt Recall nhóm nguy hiểm, hoặc vượt trần Edge AI>`

---

## 2. Cấu Hình Tái Lập (Reproducibility Config)

- **Dữ liệu & Split:** `clean Group Split` *(bắt buộc khóa cố định; có thể ghi thêm kết quả split gốc để tham chiếu)*
- **Input / ROI size:** `<416×416 | 128×128 | 224×224>`
- **Hàm mất mát (Loss):** `<CrossEntropy | Smoothed Class Weights | Focal Loss ...>`
- **Optimizer / LR / Scheduler:** `<AdamW / SGD, lr=..., CosineAnnealingLR>`
- **Epochs / Batch Size / Seed:** `<e.g. 50 epochs (Early-stop tại epoch 28), bs=64, seed=42>`
- **Augmentation áp dụng:** `<Liệt kê rõ; lưu ý không bật RandomVerticalFlip>`
- **File cấu hình & lịch sử huấn luyện (local JSON):** `<vd: checkpoints/<run_id>_summary.json>`
- **Vị trí lưu trọng số (không commit lên Git):** `<Google Drive / local checkpoints/...>`

---

## 3. Kết Quả Đo Đạc (`valid` — `clean Group Split`)

### Nếu là Stage 2 (ROI Behavior Classifier)
| Metric | Giá trị đạt được | So với Run gốc (`+/-`) | Đối chiếu ngưỡng Đạt/Trần |
| :--- | :---: | :---: | :---: |
| **`0: Drowning` Recall** *(Mục tiêu $\ge 90\%$)* | `—` | `—` | `—` |
| **`1: Person out of water` Recall** | `—` | `—` | `—` |
| **`2: Swimming` Recall** | `—` | `—` | `—` |
| **Macro F1-Score** | `—` | `—` | `—` |
| **False Alarm Rate (FAR)** | `—` | `—` | `—` |
| **Params / FLOPs (`128²`) / Size (MB) / Latency** | `—` | `—` | `≤ 1.3M / ≤ 0.18G / ≤ 5MB / ≤ 10ms` |

### Nếu là Stage 1 (Person Localizer)
| Metric (Tổng thể & Phân khúc hiểm) | Giá trị đạt được | So với Run gốc (`+/-`) | Đối chiếu ngưỡng Đạt/Trần |
| :--- | :---: | :---: | :---: |
| **Person Recall (`@score_thr=0.25`)** | `—` | `—` | `—` |
| **Precision (`@score_thr=0.25`)** | `—` | `—` | `—` |
| **`mAP@50`** | `—` | `—` | `—` |
| **Small-box Recall (`< 32×32 px`)** | `—` | `—` | `—` |
| **Crowded Recall (`≥ 3` người/ảnh)** | `—` | `—` | `—` |
| **Params / FLOPs (`416²`) / Size (MB) / Latency (FPS)** | `—` | `—` | `≤ 2.2M / ≤ 4.5G / ≤ 9MB / ≤ 18ms` |

---

## 4. Phân Tích Lỗi & Kết Luận Thực Nghiệm (Error Analysis & Decision)

- **Kiểm chứng giả thuyết & Hiệu ứng phụ:** `<Số liệu thực tế chứng minh giả thuyết ĐÚNG hay BÁC BỎ? Có phát sinh lỗi phụ ở nhóm khác không?>`
- **Mô hình vẫn đang sai ở đâu?** *(Ví dụ: nhầm `Swimming` có bọt sóng thành `Drowning`, bỏ sót người nhỏ `< 32 px`, hay tăng trễ...)*
- **Kiểm tra DoD trước khi mở PR:**
  - [ ] Đã chạy trên đúng `clean Group Split` (`seed=42`) và cùng hàm eval chuẩn của nhóm.
  - [ ] Đã kiểm tra đủ *Slicing Metrics* và thỏa mãn trần cứng ngân sách Edge AI của Stage.
  - [ ] Người thực hiện hiểu và giải thích được toàn bộ code thay đổi (kể cả phần Coding Agent sinh ra).
  - [ ] Không đẩy file trọng số nặng (`.pt`, `.onnx`) lên Git.
- **Quyết định cuối cùng:** `CHẤP NHẬN — Cập nhật Current Best (Accepted)` | `LOẠI BỎ — Giữ nguyên mốc cũ (Rejected)`
- **Gợi ý cho vòng tiếp theo:** `<...>`
