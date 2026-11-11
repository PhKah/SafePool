# SafePool Specifications (Specs)

Thư mục chứa toàn bộ tài liệu kỹ thuật, đặc tả bài toán, tiêu chuẩn phần cứng biên và nhật ký nghiên cứu cho dự án **SafePool: Edge AI Drowning Detection & Early Warning System**.

## Danh mục tài liệu

1. [**`bare_minimum.md`**](file:///Users/phamkhanh/Documents/2026.1/DL/Practice/SafePool/Specs/bare_minimum.md)
   - **Phần 1: Nguyên tắc & Quy định chung** (Tự thiết kế mạng từ đầu - *from scratch*, Nhật ký thực nghiệm minh bạch).
   - **Phần 2: Yêu cầu tối thiểu** (Pipeline Localization & Classification, Xử lý nhiễu môi trường nước, Gom cụm chuỗi thời gian *Temporal Aggregation*, Ablation Study, Bộ chỉ số an toàn sinh mạng).
   - **Phần 3: Tiêu chí đóng khung cho Edge AI** (Budget phần cứng: Size <15MB, Params <5M, FLOPs <9G, $\ge$ 30 FPS, Recall $\ge$ 90%, TTD < 4s, luồng dữ liệu 640x640 và Frame Sampling).
2. [**`dataset_specs.md`**](dataset_specs.md)
   - Đặc tả chi tiết bộ dữ liệu thực tế `data/raw/` (`12,365` ảnh `416×416`, `23,021` bounding boxes, 3 lớp `Drowning`, `Person out of water`, `Swimming`), phân bố mất cân bằng lớp và đa mục tiêu.
3. [**`roadmap.md`**](roadmap.md)
   - Lộ trình 5 cổng kiểm soát theo hướng **2-Stage From Scratch** (Clean Split & dữ liệu cho Stage 1 + Stage 2, Sanity Check, vòng lặp Baseline → Error Analysis → Ablation & Model Freeze Tuần 6, IoU Tracking + Sliding Window & Edge Demo) và timeline gợi ý 8 tuần.
4. [**`experiment_log.md`**](experiment_log.md) & [**`run_logs/_template.md`**](run_logs/_template.md)
   - Nhật ký thực nghiệm dùng chung trên Git theo mô hình **2 tầng** (Run Log cá nhân *append-only* trong `Specs/run_logs/` + Master Table do PM/Lead kiểm duyệt qua Pull Request).
