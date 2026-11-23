# Run S2-00: Sanity Check Overfit 1 Batch (Stage 2 — ROI Classifier)

> **Quy tắc:** Nhật ký thực nghiệm bất biến (Append-only) cho Gate 2 (`WP-2.2`).

---

## 1. Thông Tin Định Danh & Giả Thuyết

- **Run ID:** `S2-00`
- **Ngày chạy:** `2026-11-23`
- **Người thực hiện (Single Owner):** Phạm Nam Khánh
- **Liên kết Work Package:** `WP-2.2` (Gate 2)
- **Nhánh Git / Commit Hash:** `task/wp-2.2` (tách từ `main @ 9a4fa3c`)
- **Kế thừa từ Run:** `—` *(Run kiểm chứng đầu tiên của Stage 2)*
- **1 thay đổi duy nhất so với Run gốc:** Kiểm chứng khả năng overfit 1 batch cân bằng (`12` ROI thật) với mạng mồi `TinyStage2Probe`.
- **Giả thuyết kỹ thuật:** Nếu toàn bộ đường ống `Stage2ROIDataset → train_one_epoch → evaluate → metrics` sạch bug (không sai shape, kênh màu RGB, ánh xạ nhãn `0, 1, 2` hay đứt gradient), mạng mồi tối giản phải kéo Loss `< 0.05` và đạt `100%` Recall / Macro F1, `0%` FAR trên chính batch đó sau 50 epochs.

---

## 2. Cấu Hình Tái Lập (Reproducibility Config)

- **Dữ liệu & Split:** `clean Group Split` (`data/processed/train/`, 1 batch cân bằng gồm `4` ảnh mỗi lớp `0, 1, 2`)
- **Input / ROI size:** `128×128` (`RGB`, `float32` trong `[0, 1]`)
- **Mạng mồi:** `TinyStage2Probe` (`Conv 3→16→32→64` + `AdaptiveAvgPool2d((4, 4))` + `Linear(1024→3)`, `26,659` params)
- **Hàm mất mát (Loss):** `CrossEntropyLoss()` (không trọng số)
- **Optimizer / LR / Scheduler:** `Adam(lr=3e-3, weight_decay=0.0)`, không Scheduler
- **Epochs / Batch Size / Seed / Device:** `50` epochs, `bs=12`, `seed=42`, `device=mps`
- **Augmentation áp dụng:** Tắt hoàn toàn (`transform=None`, không Dropout, không Weight Decay)
- **Lệnh tái lập:** `python3 -m scripts.sanity_check --stage 2 --sanity-check`

---

## 3. Kết Quả Đo Đạc (Trên đúng 1 batch Sanity Check)

| Metric | Giá trị đạt được | Ngưỡng Gate 2 |
| :--- | :---: | :---: |
| **Train Loss** | `1.0986 → 0.000336` | `< 0.05` (**Đạt**) |
| **`0: Drowning` Recall** | **`100.00%`** | `100%` (**Đạt**) |
| **`1: Person out of water` Recall** | **`100.00%`** | `100%` (**Đạt**) |
| **`2: Swimming` Recall** | **`100.00%`** | `100%` (**Đạt**) |
| **Macro F1-Score** | **`100.00%`** | `100%` (**Đạt**) |
| **False Alarm Rate (FAR)** | **`0.00%`** | `0%` (**Đạt**) |
| **Params** | `26,659` | — |

---

## 4. Phân Tích & Kết Luận (Decision)

- **Kết quả:** Mạng mồi `TinyStage2Probe` hội tụ mượt từ `1.0986` xuống `0.000336` trên GPU Apple Silicon (`mps`), đạt `100%` trên cả 3 lớp và `0%` FAR. *(Ghi nhận thêm: thử nghiệm ban đầu với `AdaptiveAvgPool2d(1)` giảm chậm hơn còn `0.2376` ở epoch 50 do mất thông tin không gian ROI; đổi sang `AdaptiveAvgPool2d((4, 4))` giúp hội tụ tuyệt đối về `0.000336`).*
- **Kiểm tra DoD:**
  - [x] Chạy trên dữ liệu thật từ `clean Group Split` và dùng trực tiếp `train_one_epoch` / `evaluate` của `WP-2.1`.
  - [x] Vượt qua `tests/test_sanity_check.py::test_stage2_sanity_check_passes_gate2`.
  - [x] Không đẩy file trọng số nặng lên Git.
- **Quyết định:** **PASSED GATE 2** — Đường ống Stage 2 sạch bug, sẵn sàng sang `WP-3.1` (`M1` Baseline).
