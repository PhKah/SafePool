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
- **Model Freeze:** Chậm nhất **cuối Tuần 6** chốt trọng số tốt nhất thỏa mãn ngân sách Edge AI (`Size < 15 MB`, `Params < 5.0M`, `FLOPs < 9.0 GFLOPs`, `≥ 30 FPS`).

---

## 3. Bảng Tổng Hợp — Stage 2: ROI Behavior Classifier

*Đánh giá trên `valid` (và chốt cuối trên `test`) của **clean Group Split** (`0: Drowning`, `1: Person out of water`, `2: Swimming`).*

| Run ID | Ngày | Owner | Phiên bản / WP | 1 Thay đổi chính | `Drowning` Recall | `Out-of-water` Recall | `Swimming` Recall | Macro F1 | FAR | Params / FLOPs / Size | Chi tiết Run | Trạng thái |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `S2-00` | — | — | `WP-2.2` Sanity | Mạng mồi, 1 batch `8–16` ROI, no reg/aug | — | — | — | — | — | — | — | *Chờ chạy* |
| `S2-01` | — | — | `WP-3.1` `M1` | Baseline CNN tối giản + GAP + `CrossEntropyLoss` | — | — | — | — | — | — | — | *Mốc Baseline* |

---

## 4. Bảng Tổng Hợp — Stage 1: Person Localizer (`0: person`, `416×416`)

*Gộp 3 lớp thành `0: person`, đánh giá trên **clean Group Split**.*

| Run ID | Ngày | Owner | Phiên bản / WP | 1 Thay đổi chính | Person Recall | `mAP@50` | Small-box Recall (`<32²`) | Params | FLOPs (`416²`) | Size (MB) | FPS | Chi tiết Run | Trạng thái |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `S1-00` | — | — | `WP-2.2` Sanity | Localizer mồi, `1–4` ảnh, no reg/aug | — | — | — | — | — | — | — | — | *Chờ chạy* |
| `S1-01` | — | — | `WP-3.2` `D1` | Baseline Localizer 1-lớp `person` + NMS | — | — | — | — | — | — | — | — | *Mốc Baseline* |

---

## 5. Đánh Giá Chuỗi End-to-End (`Stage 1 → NMS → Stage 2`)

*Đo trên tập `test` của **clean Group Split** (có thể ghi thêm cột tham chiếu split mặc định Roboflow).*

| Eval ID | Ngày | Stage 1 + Stage 2 | Threshold / NMS | E2E `Drowning` Recall | E2E Macro F1 | FAR | Tổng Params | Tổng FLOPs ($N=1$ / $N=22$) | Tổng Size | FPS (ONNX) | Kết luận |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `E2E-01` | — | `WP-4.end` Freeze | — | — | — | — | — | — | — | — | *Chờ chốt Tuần 6* |
