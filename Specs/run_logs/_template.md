# Run <RUN_ID>: <Tên ngắn của thử nghiệm>

> **Quy tắc:** Sao chép mẫu này thành `Specs/run_logs/<RUN_ID>_<ten-tv>_<ten-ngan>.md` (ví dụ: `S2-02_minh_focal-loss.md`).  
> **Tính bất biến:** Không xóa hay sửa đè kết quả cũ; nếu chạy lại với cấu hình mới, tạo Run ID mới (`S2-02b`).

---

## 1. Thông Tin Định Danh & Giả Thuyết

- **Run ID:** `<RUN_ID>` *(vd: `S2-02` hoặc `S1-02`)*
- **Ngày chạy:** `YYYY-MM-DD`
- **Người thực hiện (Single Owner):** `<Tên thành viên>`
- **Liên kết Work Package:** `<WP-x.x / EXP-SLOT-k>`
- **Nhánh Git / Commit Hash:** `<branch> @ <short-sha>`
- **Kế thừa từ Run:** `<Base Run ID>` *(vd: `S2-01`)*
- **1 thay đổi duy nhất so với Run gốc:** `<Mô tả ngắn 1 yếu tố thay đổi>`
- **Giả thuyết kỹ thuật:** `<Dựa trên lỗi nào của Run trước? Kỳ vọng cải thiện chỉ số nào?>`

---

## 2. Cấu Hình Tái Lập (Reproducibility Config)

- **Dữ liệu & Split:** `clean Group Split` *(bắt buộc; có thể ghi thêm kết quả split gốc để tham chiếu)*
- **Input / ROI size:** `<416×416 | 128×128 | 224×224>`
- **Hàm mất mát (Loss):** `<CrossEntropy | Smoothed Class Weights | Focal Loss ...>`
- **Optimizer / LR / Scheduler:** `<AdamW / SGD, lr=..., CosineAnnealingLR>`
- **Epochs / Batch Size / Seed:** `<e.g. 50 epochs (Early-stop tại epoch 28), bs=64, seed=42>`
- **Augmentation áp dụng:** `<Liệt kê rõ; lưu ý không bật RandomVerticalFlip>`
- **Vị trí lưu trọng số (không commit lên Git):** `<Google Drive / local checkpoints/...>`

---

## 3. Kết Quả Đo Đạc (`valid` — `clean Group Split`)

### Nếu là Stage 2 (ROI Behavior Classifier)
| Metric | Giá trị đạt được | So với Run gốc (`+/-`) |
| :--- | :---: | :---: |
| **`0: Drowning` Recall** *(Mục tiêu $\ge 90\%$)* | `—` | `—` |
| **`1: Person out of water` Recall** | `—` | `—` |
| **`2: Swimming` Recall** | `—` | `—` |
| **Macro F1-Score** | `—` | `—` |
| **False Alarm Rate (FAR)** | `—` | `—` |
| **Params / FLOPs / Size (MB)** | `—` | `—` |

### Nếu là Stage 1 (Person Localizer)
| Metric | Giá trị đạt được | So với Run gốc (`+/-`) |
| :--- | :---: | :---: |
| **Person Recall** | `—` | `—` |
| **`mAP@50`** | `—` | `—` |
| **Small-box Recall (`< 32×32 px`)** | `—` | `—` |
| **Params / FLOPs (`416²`) / Size (MB) / FPS** | `—` | `—` |

---

## 4. Phân Tích Lỗi & Kết Luận (Error Analysis & Decision)

- **Mô hình đang sai ở đâu?** *(Ví dụ: nhầm `Swimming` có bọt sóng thành `Drowning`, bỏ sót người nhỏ `< 32 px`, hay tăng trễ vượt ngân sách...)*
- **Kiểm tra DoD trước khi mở PR:**
  - [ ] Đã chạy trên đúng `clean Group Split` và cùng hàm eval chuẩn của nhóm.
  - [ ] Người thực hiện hiểu và giải thích được toàn bộ code thay đổi (kể cả phần Coding Agent sinh ra).
  - [ ] Không đẩy file trọng số nặng (`.pt`, `.onnx`) lên Git.
- **Đề xuất quyết định:** `Giữ (Update Current Best)` | `Loại (Giữ nguyên mốc cũ)`
- **Gợi ý cho vòng tiếp theo:** `<...>`
