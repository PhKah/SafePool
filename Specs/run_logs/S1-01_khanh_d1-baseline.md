# Run S1-01: Mô Hình Gốc Stage 1 (`D1` — Single-Scale Person Localizer `26×26` From Scratch)

> **Quy tắc:** Nhật ký thực nghiệm bất biến (*Append-only*) cho `WP-3.2` theo chuẩn [`Specs/run_logs/_template.md`](_template.md).

---

## 1. Thông Tin Định Danh & Giả Thuyết

- **Run ID:** `S1-01`
- **Ngày chạy:** `2026-11-27`
- **Người thực hiện (Single Owner):** Phạm Nam Khánh
- **Liên kết Work Package:** `WP-3.2` (`Stage 1 Baseline D1 + NMS / Soft-NMS + Edge Profiling`)
- **Nhánh Git / Commit Base:** `task/wp-3.2` *(tách từ `main @ 9c64623`)*
- **Kế thừa từ Run:** `S1-00` *(Gate 2 Sanity Check)*
- **1 thay đổi duy nhất so với Run gốc:** Thiết lập mốc chuẩn đầu tiên (`Baseline D1`) cho bài toán định vị người 1 lớp (`0: person`) huấn luyện **100% From Scratch** trên toàn bộ tập sạch `clean Group Split` (`8,655` ảnh `train` / `1,855` ảnh `valid`), kiến trúc CNN đơn quy mô xuất lưới `26×26`, không dùng augmentation.
- **Giả thuyết kỹ thuật:**
  1. Lưới đơn quy mô `26×26` (`stride = 16 px` trên ảnh `416×416`) kết hợp tích chập giãn nở (`dilation = 2`, vùng tiếp nhận `~207×207 px`) đủ sức tách biệt tâm người bơi và định vị phần lớn mục tiêu trong ngân sách cực gọn (`~1.07M` params, `~2.80 GFLOPs` — chưa tới `65%` trần phân bổ của Stage 1).
  2. Do cố tình giữ kiến trúc **1-scale, gán 1 ô tâm/người và tắt hoàn toàn augmentation (`transform=None`)** để làm đối chứng chuẩn cho Giai đoạn 4 (*Ablation Study*), mô hình sẽ bộc lộ rõ các giới hạn tự nhiên trên hồ bơi chưa từng thấy ở `valid` (hiện tượng quá khớp miền màu nước/ánh sáng, sai số co giãn khung bao `0.25 ≤ IoU < 0.50`, và suy giảm Recall ở cảnh đông người `≥ 3` người hoặc mục tiêu nhỏ `< 32×32 px`).

---

## 2. Cơ Sở Lý Thuyết & Lý Do Lựa Chọn Kỹ Thuật (Design Rationale)

Mọi quyết định thiết kế trong `Stage1BaselineD1` ([`src/models/stage1_baseline.py`](../../src/models/stage1_baseline.py)) đều đối chiếu trực tiếp từ thống kê EDA ([`Specs/dataset_specs.md`](../dataset_specs.md)) và 3 cơ chế bảo vệ số học/phần cứng cho Detector huấn luyện từ đầu:

1. **Lưới đầu ra `26×26` (`stride = 16 px`) + Dilated Head (`RF ≈ 207 px`):**
   - *Cơ sở EDA:* Dữ liệu có `14.5%` box nhỏ `< 32×32 px` (`p10 = 26×30 px`), `p90 = 204×229 px` và `34.86%` ảnh đông người (tối đa `22` người/ảnh).
   - *Kiểm chứng thực tế trên `valid`:* Lưới `26×26` đạt **`0 / 2,987` ca trùng ô tâm (`0.00%` grid collision)**, khắc phục triệt để nhược điểm gộp người của lưới `13×13` mà không tốn `4×` chi phí head như lưới `52×52`.
2. **Safeguard 1 — Triệt tiêu bùng nổ Gradient bằng Bounded Sigmoid Mapping:**
   - Thay vì tham số hóa chiều rộng/cao bằng hàm mũ không bị chặn $e^t$ (đạo hàm $e^t$ khuếch đại gradient gây tràn `NaN/Inf` khi khởi tạo ngẫu nhiên *From Scratch*), `D1` ánh xạ toàn bộ $(t_x, t_y, t_w, t_h)$ qua $\sigma(t) \in (0, 1)$ với đạo hàm bị chặn chặt $\sigma'(t) \le 0.25$.
   - Khởi tạo bias nhánh objectness $b_{\text{obj}} = -\ln((1-\pi)/\pi) \approx -4.595$ ($\pi = 0.01$) và bias kích thước $b_{w,h} = -1.4$ ($\sim 83\text{ px}$, sát trung vị người bơi) giúp loss epoch đầu êm ái (`3.20`), không có bất kỳ bước `NaN` nào.
3. **Hàm mất mát `Stage1BaselineLoss` (`BCE(pos_weight=15.0) + 5.0 × (SmoothL1 + GIoU)`):**
   - Tỷ lệ ô nền / ô có tâm người trên lưới `26×26` xấp xỉ `365 : 1`; `pos_weight = 15.0` ngăn mạng sụp đổ về nghiệm đoán toàn nền.
   - Kết hợp `SmoothL1(β=0.1)` với `1 - GIoU` (bất biến theo tỷ lệ kích thước) giúp tối ưu đồng thời sai số tọa độ tâm và diện tích giao-hợp trực tiếp phục vụ `mAP@50`.
4. **Safeguard 2 — Chống ngắt non do trễ pha `mAP@50` (`warmup_epochs = 12`):**
   - `mAP@50` là hàm bậc thang rời rạc ($\text{IoU} \ge 0.5$); `EarlyStopping(patience=10, warmup_epochs=12)` không cộng dồn `bad_epochs` trong 12 epoch đầu để tránh dừng non khi hàm loss liên tục vẫn đang kéo khung bao về vùng khả thi.
5. **Safeguard 3 — Giải mã SIMD toàn phần & Batched `Top-K` trước NMS:**
   - Giải mã toàn bộ `676` ô lưới bằng PyTorch broadcasting + `torch.topk(k=100)` theo lô ngay trên `MPS` (chỉ truyền `≤ 100` ứng viên/ảnh về CPU chạy `torchvision.ops.nms` / `soft_nms_xyxy`), chặn cứng độ phức tạp cặp đôi $\mathcal{O}(M^2)$ và giữ độ trễ toàn Stage 1 ở mức **`3.63 ms/ảnh` (`275.6 FPS`)**.

---

## 3. Cấu Hình Tái Lập (Reproducibility Config)

- **Dữ liệu & Split:** `clean Group Split` (`data/stage1/manifest.json` — `8,655` `train` / `1,855` `valid`, `2,987` GT boxes `person` trên `valid`, `0%` rò rỉ video nguồn)
- **Input size:** `416×416` (`RGB`, `float32 ∈ [0, 1]`)
- **Kiến trúc:** `Stage1BaselineD1` (`7` khối `ConvBNSiLU` backbone + `1` khối `head_refine` `dilation=2` + `Conv 1×1` xuất `[B, 5, 26, 26]`, **0% pretrained**)
- **Hàm mất mát (Loss):** `BCEWithLogitsLoss(pos_weight=15.0) + 5.0 × (SmoothL1Loss(β=0.1) + (1 - GIoU))`
- **Optimizer / LR / Scheduler:** `AdamW(lr=1e-3, weight_decay=1e-4)` + `CosineAnnealingLR(T_max=40, eta_min=1e-5)` + `grad_clip=5.0`
- **Epochs / Batch Size / Seed / Device:** Tối đa `40` epochs (`EarlyStopping` kích hoạt tại **epoch `22`**, **Best Epoch = `11`**), `bs=16`, `seed=42`, `device=mps` (`42.90` phút)
- **Augmentation áp dụng:** **Không bật (`transform=None`)**; tuân thủ tuyệt đối quy tắc không bật `RandomVerticalFlip`.
- **Lệnh tái lập:**
  ```bash
  python3 -m src.train_stage1 --epochs 40 --batch-size 16 --lr 1e-3 --weight-decay 1e-4 --grad-clip 5.0 --patience 10 --warmup-epochs 12 --seed 42 --device auto
  ```
- **Vị trí lưu trọng số cục bộ (git-ignored):** `checkpoints/stage1_d1_best.pt` & `checkpoints/stage1_d1_summary.json`

---

## 4. Kết Quả Đo Đạc (`valid` — `clean Group Split`, Best Epoch = `11`)

### 4.1. Bảng chỉ số chính & Đối chiếu Standard NMS vs Gaussian Soft-NMS (`IoU ≥ 0.50`, `score_thr = 0.25`)

| Hậu xử lý | Person Recall (`≥0.25`) | `mAP@50` (Full PR Curve) | Small-box Recall (`< 32×32 px`, `482` GT) | Regular-box Recall (`≥ 32×32 px`, `2,505` GT) | Ghi chú |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Standard NMS (`iou_thr=0.5`)** | **`67.73%`** (`2,023/2,987`) | **`40.23%`** | `64.11%` (`309/482`) | **`68.42%`** (`1,714/2,505`) | **Cấu hình chuẩn chốt cho `S1-01`** *(Peak Recall trong quá trình train đạt `70.51%` ở Ep 18)* |
| **Gaussian Soft-NMS (`σ=0.5`)** | `64.81%` (`1,936/2,987`) | `39.87%` | **`65.35%`** (`315/482`, `+1.24%`) | `64.71%` (`1,621/2,505`) | Giữ thêm được người nhỏ sát nhau (`+6` GT nhỏ), nhưng làm suy giảm điểm các box người vừa/lớn dưới ngưỡng `0.25` |

### 4.2. Độ nhạy theo ngưỡng điểm tin cậy (`score_thr`, với Standard NMS)

| Ngưỡng `score_thr` | Person Recall | Precision (`@thr`) | Small-box Recall (`<32²`) | Số box dự đoán / `1,855` ảnh (`TB/ảnh`) |
| :---: | :---: | :---: | :---: | :---: |
| `0.15` | **`72.95%`** | `15.01%` | `64.32%` | `14,517` (`7.83` box/ảnh) |
| **`0.25` *(Mặc định)*** | **`67.73%`** | `20.59%` | **`64.11%`** | `9,823` (`5.30` box/ảnh) |
| `0.35` | `63.27%` | `25.86%` | `62.86%` | `7,308` (`3.94` box/ảnh) |
| `0.45` | `58.86%` | `31.21%` | `62.24%` | `5,632` (`3.04` box/ảnh) |
| `0.55` | `53.30%` | `36.32%` | `59.96%` | `4,383` (`2.36` box/ảnh) |

### 4.3. Kiểm tra Ngân sách Edge AI (Đối chiếu trần Stage 1 tại [`Specs/roadmap.md`](../roadmap.md))

| Chỉ số Edge AI | Giá trị `Stage1BaselineD1` | Trần phân bổ Team Stage 1 | Mức sử dụng ngân sách | Kết luận |
| :--- | :---: | :---: | :---: | :---: |
| **Tham số (`Params`)** | **`1,072,485` (`1.07M`)** | `≤ 2.20M` | `48.7%` | **ĐẠT** (dư `1.13M` cho FPN/Attention) |
| **Chi phí tính toán (`FLOPs @ 416²`)** | **`2.7983 GFLOPs`** | `≤ 4.50 GFLOPs` | `62.2%` | **ĐẠT** (dư `1.70 GFLOPs`) |
| **Dung lượng trọng số FP32 (`Size`)** | **`4.11 MB`** | `≤ 9.00 MB` | `45.7%` | **ĐẠT** |
| **Độ trễ E2E Stage 1 (`Model + Decode + NMS`)** | **`3.63 ms/frame` (`275.6 FPS` trên `MPS`)** | `≤ 18.0 ms/frame` (`≥ 55 FPS`) | `20.2%` | **ĐẠT VƯỢT MỨC** (`p95 = 4.07 ms`) |

---

## 5. Phân Tích Lỗi Định Lượng & Kết Luận (Error Analysis & Decision)

### 5.1. Giải phẫu `964 / 2,987` mục tiêu bị bỏ sót (`FN` tại `score_thr = 0.25`, `IoU ≥ 0.50`)
1. **Nhóm lỗi #1 — Sai lệch biên hộp bao `poor_loc` chiếm đa số tuyệt đối (`572 / 964 = 59.34%` tổng FN!):**
   - Có tới **`2,595 / 2,987` người (`86.88%`)** thực tế đã được `D1` phát hiện đúng vùng vị trí với `score ≥ 0.25` và `IoU ≥ 0.25`, nhưng `572` trường hợp chỉ đạt `0.25 ≤ IoU < 0.50` (bị tính đồng thời là `1 FP + 1 FN` dưới thước đo bậc thang `mAP@50`).
   - *Nguyên nhân gốc:* Kiến trúc đơn quy mô `26×26` (`stride=16`) thiếu đặc trưng biên phân giải cao từ tầng `52×52` (`stride=8`), đồng thời quy tắc chỉ gán **1 ô tâm duy nhất** là dương (`1.0`) trong khi các ô lân cận nằm ngay trên thân người bị ép nhãn âm (`0.0`) làm nhiều ô cạnh nhau cùng phát xung lấp lửng (`~5.30` box/ảnh $\rightarrow$ `Precision = 20.59%`, kéo `mAP@50` xuống `40.23%`).
2. **Nhóm lỗi #2 — Quá khớp miền hồ bơi khi chưa có Data Augmentation (`val_loss` chạm đáy sớm ở Epoch 11):**
   - `train_loss` giảm mượt liên tục `3.2016 → 0.8853` (Ep 1 → Ep 22), nhưng trên `clean Group Split` (nơi tập `valid` gồm các góc máy/màu nước/ánh nắng chưa từng xuất hiện ở `train`), `val_loss` chạm đáy `2.4986` tại **Epoch 11** rồi đi ngang/tăng nhẹ lên `2.58–2.62`. Đây là hệ quả tất yếu và rất rõ ràng của việc huấn luyện *From Scratch* với `transform = None`.
3. **Nhóm lỗi #3 — Suy giảm mạnh ở khung hình đông người (`-13.54%`) & mục tiêu nhỏ (`< 32×32 px`):**
   - Khung hình thưa (`1–2` người/ảnh): `Person Recall` đạt **`73.03%`** (`1,327 / 1,817`).
   - Khung hình đông (`≥ 3` người/ảnh): `Person Recall` tụt xuống **`59.49%`** (`696 / 1,170`).
   - Mục tiêu nhỏ (`< 32×32 px`): đạt `64.11%` (`309 / 482`), thấp hơn `4.31%` so với nhóm vừa/lớn (`68.42%`).

### 5.2. Phụ Lục Kiểm Thử Đa Chiều & Soi Trực Quan 30 Mẫu Lỗi (Theo Chuẩn Mới `Specs/experiment_log.md`)

> Tái lập tự động bằng lệnh: `python3 -m scripts.audit_stage1_baseline` *(chỉ chạy trên `valid` sạch, `seed=42`, `0%` chạm `test`; báo cáo JSON & lưới ảnh lưu tại `checkpoints/audit_s1_01/`, xem trực quan tại `notebooks/stage1_d1_audit_experiment_log.ipynb`)*.

#### A. Bảng Slicing Metrics đầy đủ (`@score_thr = 0.25`, `IoU ≥ 0.50`)

| Thước đo (Tổng thể & Phân khúc hiểm) | Standard NMS (`S1-01`) | Gaussian Soft-NMS (`σ=0.5`) | Ghi chú tác động lên chuỗi 2-Stage |
| :--- | :---: | :---: | :--- |
| **Person Recall (`@0.25`)** | **`67.73%`** (`2,023 / 2,987`) | `64.81%` (`1,936 / 2,987`) | Mốc chuẩn tổng thể của `D1` |
| **Precision (`@0.25`)** / **F1 (`@0.25`)** | **`20.59%`** / **`31.58%`** | `21.35%` / `32.11%` | Xuất trung bình `5.30` box/ảnh (`9,823` box / `1,855` ảnh) |
| **`mAP@50`** *(Full PR Curve)* | **`40.23%`** | `39.87%` | Diện tích toàn đường cong Precision–Recall |
| **Small-box Recall (`< 32×32 px`)** | `64.11%` (`309 / 482`) | **`65.35%`** (`315 / 482`) | Soft-NMS giữ tốt hơn nhóm người nhỏ sát nhau |
| **Crowded Recall (`≥ 3` người/ảnh)** | **`59.49%`** (`696 / 1,170`) | `55.64%` (`651 / 1,170`) | Thấp hơn **`-13.54%`** so với cảnh thưa `1–2` người (`73.03%`) |
| **Recall riêng lớp gốc `0: Drowning`** | **`67.68%`** (`1,353 / 1,999`) | `65.58%` (`1,311 / 1,999`) | **Trần giới hạn trực tiếp của E2E `Drowning Recall` hiện tại** |
| **Recall riêng lớp gốc `1: Out of water`** | **`40.68%`** (`48 / 118`) | `38.98%` (`46 / 118`) | Điểm mù lớn nhất do thiên lệch nền nước xanh ở tập train |
| **Recall riêng lớp gốc `2: Swimming`** | **`71.49%`** (`622 / 870`) | `66.55%` (`579 / 870`) | Nhóm dễ nhận diện nhất trên mặt nước |

#### B. Giải phẫu `7,800` hộp thừa (`FP`) & Phát hiện từ soi trực quan `30` ca tệ nhất (`15 FN` + `15 FP`)
- **Phân rã `7,800` hộp `FP` (`@0.25`):**
  - **`3,470` hộp (`44.5%`) là `Duplicate Cluster` (`0.10 ≤ IoU < 0.50` quanh GT thật):** Các ô lưới lân cận tâm người cùng phát xung nhưng chồng lấn `< 0.50` nên lọt qua Standard NMS.
  - **`4,330` hộp (`55.5%`) là `Background FP` (`IoU < 0.10` với mọi GT đã gán nhãn).**
- **Kết quả quan sát bằng mắt trên `15` ca bỏ sót `0: Drowning` tệ nhất (`worst_15_drowning_fn.png`):**
  1. **Mất tương phản ở cảnh thiếu sáng / nước tối đen (`valid[1608, 1613, 1616]`):** Nạn nhân rất lớn ở gần camera (`~160×200 px`) nhưng ánh sáng cực tối làm tắt kích hoạt tích chập (`IoU = 0.00`).
  2. **Cháy sáng lóa nắng & biến đổi phổ màu cực đoan (`valid[1614, 227, 1580, 1584]`):** Bọt trắng lóa gắt (`valid[1614]`) hoặc nước bị lệch màu tím hồng / xanh lá mạ (`valid[227, 1580]`) khiến mạng chưa học Augmentation bị "mù".
  3. **Mục tiêu siêu nhỏ `< 16×24 px` từ camera góc cao/drone (`valid[1578, 1579, 1581, 318]`):** Chiều ngang người chỉ `10–15 px` — nhỏ hơn bước nhảy `stride = 16 px` của lưới `26×26`.
- **Kết quả quan sát bằng mắt trên `15` ca `Background FP` điểm cao nhất `score = 0.96–0.99` (`worst_15_background_fp.png`):**
  1. **Phát hiện quan trọng về nhiễu thiếu nhãn gốc (*Missing Annotation* trong Roboflow):** Trong đa số các ca `score ≥ 0.96` ở bể đông phao bơi (`valid[1591, 668, 877, 879, 881, 722, 723]`), hộp dự đoán của `D1` **thực chất bắt trúng người thật 100%** nhưng ảnh gốc Roboflow quên gán nhãn cho những người đó $\rightarrow$ `Precision` (`20.59%`) và `mAP@50` (`40.23%`) đang bị chấm khắt khe hơn thực tế.
  2. **Báo giả quang học thật sự (`valid[1640, 176, 133]`):** Đốm nắng khúc xạ trắng trên gợn sóng và phao phân làn bị nhầm thành người bơi nhỏ ở xa.

---

### 5.3. Kiểm tra DoD & Chốt Giả Thuyết Có Ngưỡng Bác Bỏ Cho Vòng Tiếp Theo
- [x] Đã chạy trên đúng `clean Group Split` (`seed=42`) và dùng trực tiếp `train_one_epoch` + `detection_metrics` chuẩn của nhóm (`WP-2.1`).
- [x] Đã đo đầy đủ *Slicing Metrics*, soi trực quan 30 ca lỗi tệ nhất và vượt qua cổng cứng `STAGE1_BUDGET` (`1.07M ≤ 2.2M`, `2.80G ≤ 4.5G`, `4.11MB ≤ 9.0MB`, `3.63ms ≤ 18ms`).
- [x] Không bật `RandomVerticalFlip`; không đẩy file trọng số `.pt` lên Git; vượt `100%` bộ kiểm thử tự động (`pytest`).
- **Quyết định cuối cùng:** **`CHẤP NHẬN LÀM MỐC BASELINE (Current Best — Stage 1)`**
- **Giả thuyết có điều kiện bác bỏ định lượng chuyển sang Giai đoạn 4 (`EXP-SLOT`):**
  1. **Ứng viên 1 (`S1-02` — Photometric & Geometric Pool Augmentation, `+0` FLOPs suy luận):**
     - *1 biến thay đổi:* Bật `RandomHorizontalFlip` + `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)` + `RandomScale/Translate` (tuyệt đối không `VerticalFlip`) để chữa trực tiếp nhóm lỗi tối đen (`valid[1608]`), lóa trắng (`valid[1614]`), lệch màu (`valid[227]`) và quá khớp sau Epoch 11.
     - *Ngưỡng Đạt (`Accepted`):* `mAP@50` tăng **`≥ +3.0%`** (`≥ 43.2%`), `Person Recall` & `0: Drowning Recall` tăng **`≥ +2.0%`**, `Params/FLOPs` giữ nguyên `1.07M / 2.80G`. *Bác bỏ (`Rejected`)* nếu không đạt ngưỡng trên.
  2. **Ứng viên 2 (`S1-03` — Lightweight 2-Scale FPN `52×52 + 26×26` + Multi-cell/IoU-aware Assignment):**
     - *1 biến thay đổi:* Thêm nhánh `52×52` (`stride = 8 px`) và cải tiến gán nhãn quanh tâm để chữa trực tiếp `572` ca lệch biên `0.25 ≤ IoU < 0.50`, mục tiêu `10–16 px` (`valid[1578, 318]`) và `3,470` hộp dư quanh người.
     - *Ngưỡng Đạt (`Accepted`):* `mAP@50` tăng **`≥ +5.0%`**, `Small-box Recall` & `Crowded Recall` tăng **`≥ +4.0%`**, giữ chặt trong trần `STAGE1_BUDGET` (`≤ 2.2M params`, `≤ 4.5 GFLOPs`, `≤ 18 ms`).
