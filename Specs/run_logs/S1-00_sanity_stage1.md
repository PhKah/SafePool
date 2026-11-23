# Run S1-00: Sanity Check Overfit 1 Batch (Stage 1 — Person Localizer)

> **Quy tắc:** Nhật ký thực nghiệm bất biến (Append-only) cho Gate 2 (`WP-2.2`).

---

## 1. Thông Tin Định Danh & Giả Thuyết

- **Run ID:** `S1-00`
- **Ngày chạy:** `2026-11-23`
- **Người thực hiện (Single Owner):** Phạm Nam Khánh
- **Liên kết Work Package:** `WP-2.2` (Gate 2)
- **Nhánh Git / Commit Hash:** `task/wp-2.2` (tách từ `main @ 9a4fa3c`)
- **Kế thừa từ Run:** `—` *(Run kiểm chứng đầu tiên của Stage 1)*
- **1 thay đổi duy nhất so với Run gốc:** Kiểm chứng khả năng overfit 1 batch nhỏ (`2` ảnh `416×416` thật, `7` ground-truth boxes `person`) với detector lưới `13×13` tối giản `TinyStage1Probe`.
- **Giả thuyết kỹ thuật:** Nếu toàn bộ đường ống `Stage1Dataset → stage1_collate_fn (tuple targets) → train_one_epoch (move_to_device) → Detection Loss → Decode + NMS → detection_metrics` sạch bug (không sai hệ tọa độ YOLO `cx,cy,w,h` vs `xyxy`), mạng mồi phải kéo Detection Loss `< 0.10` và đạt `person_recall ≥ 95%`, `mAP@50 ≥ 95%`.

---

## 2. Cấu Hình Tái Lập (Reproducibility Config)

- **Dữ liệu & Split:** `clean Group Split` (`data/stage1/manifest.json`, 1 batch gồm `2` ảnh thật `416×416`, tổng `7` boxes `0: person`)
- **Input size:** `416×416` (`RGB`, `float32` trong `[0, 1]`)
- **Mạng mồi:** `TinyStage1Probe` (5 tầng Conv stride 2 `416→13` + head `1×1` xuất `[B, 5, 13, 13]`, `245,669` params)
- **Hàm mất mát (Loss):** `Stage1SimpleDetectionLoss` (`BCEWithLogitsLoss(pos_weight=5.0)` + `5.0 × SmoothL1Loss`)
- **Optimizer / LR / Scheduler:** `Adam(lr=3e-3, weight_decay=0.0)`, không Scheduler
- **Epochs / Batch Size / Seed / Device:** `80` epochs, `bs=2`, `seed=42`, `device=mps`
- **Augmentation áp dụng:** Tắt hoàn toàn (`transform=None`, không Weight Decay)
- **Lệnh tái lập:** `python3 -m scripts.sanity_check --stage 1 --sanity-check`

---

## 3. Kết Quả Đo Đạc (Trên đúng 1 batch Sanity Check)

| Metric | Giá trị đạt được | Ngưỡng Gate 2 |
| :--- | :---: | :---: |
| **Detection Loss** | `0.9645 → 0.001199` | `< 0.10` (**Đạt**) |
| **Person Recall (`IoU ≥ 0.5`)** | **`100.00%`** (`7 / 7` GT) | `≥ 95%` (**Đạt**) |
| **`mAP@50`** | **`100.00%`** | `≥ 95%` (**Đạt**) |
| **Params** | `245,669` | — |

---

## 4. Phân Tích & Kết Luận (Decision)

- **Kết quả:** Mạng mồi `TinyStage1Probe` hội tụ mạnh từ `0.9645` xuống `0.001199` trên GPU `mps`, định vị đúng trọn vẹn `7 / 7` người (`Person Recall = 100%`, `mAP@50 = 100%`). Xác nhận khớp nối `Stage1Dataset` (`tuple` targets) $\leftrightarrow$ `train_one_epoch` (`move_to_device`) $\leftrightarrow$ `detection_metrics` hoạt động chuẩn xác.
- **Kiểm tra DoD:**
  - [x] Chạy trên dữ liệu thật từ `clean Group Split` và dùng trực tiếp `train_one_epoch` / `detection_metrics` của `WP-2.1`.
  - [x] Vượt qua `tests/test_sanity_check.py::test_stage1_sanity_check_passes_gate2`.
  - [x] Không đẩy file trọng số nặng lên Git.
- **Quyết định:** **PASSED GATE 2** — Đường ống Stage 1 sạch bug, sẵn sàng sang `WP-3.2` (`D1` Baseline).
