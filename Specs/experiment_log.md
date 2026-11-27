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
1. **Claim Run ID & Giả thuyết:** Kiểm tra WIP ($\le 1\text{--}2$ việc/người), báo lên nhóm để lấy Run ID kế tiếp (tránh trùng ý tưởng) và chốt **1 biến thay đổi duy nhất**.
2. **Nhánh riêng & Pipeline chuẩn:** Tạo branch riêng (vd: `exp/S2-02-focal-loss`), chạy bằng script train/eval & `clean Group Split` chuẩn chung của nhóm (không tự sửa công thức metric trong lúc thí nghiệm).
3. **Lưu log & Kiểm tra tại chỗ:** Ghi nhận `commit hash`, `seed`, hyperparams, metrics, `Params / FLOPs / Size`. *(Giữ `.pt` tại local/Drive, không push lên Git).*
4. **Tạo Run Log & Mở PR:** Copy [`Specs/run_logs/_template.md`](run_logs/_template.md) thành file riêng của run đó, điền phần *Phân tích lỗi* và đề xuất *Giữ / Loại*, rồi mở Pull Request.
5. **Duyệt & Cập nhật Master Table:** Lead kiểm tra tính tái lập và hợp nhất kết quả vào bảng dưới đây (cùng trạng thái `Current Best` nếu vượt mốc cũ).

---

## 2. Quy Tắc Kiểm Soát Vòng Lặp (Early-Stop & Model Freeze)

- **Nhịp chạy mục tiêu:** `30 phút – 2–3 giờ` / thử nghiệm.
- **Per-run Early Stopping:** `patience = 8–12 epochs` theo `Drowning Recall` + `Macro F1` / `mAP@50` trên `valid` sạch.
- **Diminishing Returns:** Tối đa `3–5` giả thuyết chính sau Baseline `M1`; dừng mở thêm nếu `2` vòng liên tiếp tăng `< +0.5%–1.0%` nhưng tăng FLOPs.
- **Model Freeze:** Chậm nhất **cuối Tuần 6** chốt trọng số tốt nhất thỏa mãn ngân sách Edge AI tổng (`Size < 15 MB`, `Params < 5.0M`, `FLOPs < 9.0 GFLOPs` với $N \le 22$, `≥ 30 FPS`).
- **Trần ngân sách phân bổ theo Stage** *(chi tiết tại [`Specs/roadmap.md`](roadmap.md))*:
  - **Stage 1 (`416×416`):** `Params ≤ 2.2M` | `Size ≤ 9.0 MB` (FP32) | `FLOPs ≤ 4.5 GFLOPs` | `Latency ≤ 18 ms`.
  - **Stage 2 (`128×128`/ROI):** `Params ≤ 1.3M` | `Size ≤ 5.0 MB` (FP32) | `FLOPs ≤ 0.18 GFLOPs/ROI` | `Latency ≤ 10 ms` (batch).

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

---

## 5. Đánh Giá Chuỗi End-to-End (`Stage 1 → NMS → Stage 2`)

*Đo trên tập `test` của **clean Group Split** (có thể ghi thêm cột tham chiếu split mặc định Roboflow).*

| Eval ID | Ngày | Stage 1 + Stage 2 | Threshold / NMS | E2E `Drowning` Recall | E2E Macro F1 | FAR | Tổng Params | Tổng FLOPs ($N=1$ / $N=22$) | Tổng Size | FPS (ONNX) | Kết luận |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `E2E-01` | — | `WP-4.end` Freeze | — | — | — | — | — | — | — | — | *Chờ chốt Tuần 6* |
