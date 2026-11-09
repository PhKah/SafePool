# SafePool: Đặc Tả Yêu Cầu Tối Thiểu (Bare Minimum Specifications)

> **Mã dự án:** SafePool  
> **Chủ đề:** Hệ thống thị giác máy tính và học sâu phát hiện nguy cơ đuối nước (Drowning Detection & Early Warning System)  
> **Phiên bản:** 1.0 (Baseline Specification)  

---

## Phần 1: Nguyên Tắc & Quy Định Chung

### Nguyên tắc 1 (Cốt lõi): Tự thiết kế và huấn luyện từ đầu (From Scratch)
> [!IMPORTANT]
> **Mục tiêu chủ đạo của dự án:** Tự thiết kế và xây dựng một mạng học (neural network / model architecture) từ đầu, **không sử dụng** các mô hình đã được huấn luyện sẵn (pre-trained models) hoặc các kiến trúc đóng gói sẵn.
> - Toàn bộ quá trình phải dựa trên các chủ đích thiết kế rõ ràng.
> - Trải qua các thử nghiệm thực nghiệm có kiểm soát để từ đó đưa ra kết luận kỹ thuật và đúc kết kinh nghiệm thực tiễn.

### Nguyên tắc 2 (Tính minh bạch): Nhật ký thực nghiệm chi tiết
> [!NOTE]
> Mọi thay đổi về:
> - Siêu tham số (**hyperparameters**)
> - Cấu trúc tầng (**layers**)
> - Phương pháp tiền xử lý (**preprocessing & augmentation**)
> 
> đều **bắt buộc phải được ghi lại chi tiết** trong nhật ký thử nghiệm (Experiment Logs) kèm theo lý do kỹ thuật khi thực hiện.

---

## Phần 2: Yêu Cầu Tối Thiểu Đối Với Đề Tài Phát Hiện Đuối Nước (Drowning Detection)

### 1. Kiến trúc Hệ thống & Luồng xử lý (Architecture & Pipeline)
- **Pipeline đa nhiệm (Multi-task Pipeline):**
  - Quá trình xử lý cần tách biệt hoặc tích hợp rõ ràng 2 bước:
    1. **Định vị vị trí con người trong nước** (*Localization / Object Detection*).
    2. **Phân loại trạng thái đuối nước** (*Classification / Inference*).
- **Nút thắt kỹ thuật - Nhiễu môi trường nước:**
  - Bắt buộc có giải pháp xử lý nhiễu đặc thù của môi trường nước (gợn sóng, bọt nước, khúc xạ / phản chiếu ánh sáng, che khuất một phần cơ thể).
  - Cần áp dụng có chủ đích:
    - Kỹ thuật tăng cường dữ liệu chuyên biệt (**Data Augmentation**: lật, cắt, đổi màu, nhiễu sáng,...).
    - Các khối tập trung (**Attention Mechanisms**).
    - Hoặc điều chỉnh hàm mất mát (**Loss Function** phù hợp).
- **Xử lý chuỗi thời gian (Temporal Aggregation):**
  - Vì đuối nước là một chuỗi hành vi biến đổi theo thời gian, **không thể kết luận chỉ bằng một ảnh tĩnh đơn lẻ**.
  - **Mức tối thiểu (Bare Minimum):** Tích hợp cơ chế gom/làm mịn theo chuỗi khung hình (*Multi-frame Temporal Aggregation / Sliding Window*).
  - Cảnh báo chỉ kích hoạt khi hành vi bất thường kéo dài $N$ khung hình (hoặc $X$ giây), giúp triệt tiêu triệt để báo động giả do giơ tay, đập nước hay sóng vỗ tức thời.

---

### 2. Khả Thi Thực Nghiệm & Huấn Luyện (Compute & Experimentation)
- **Tốc độ lặp nhanh (Fast Iteration Loop):**
  - Thời gian huấn luyện cho 1 lượt chạy thử nghiệm (*1 experiment*) chỉ nên dao động từ **30 phút đến 2–3 tiếng** trên GPU cá nhân hoặc Google Colab (T4 / P100).
  - Cần tránh các kiến trúc quá nặng nề làm mất tới 24 tiếng cho 1 epoch để đảm bảo quỹ thời gian cho việc gỡ lỗi (debug) và tối ưu hóa lặp lại.
- **Khả năng chẩn đoán & Thử nghiệm loại trừ (Ablation Study):**
  - Đề tài phải đủ linh hoạt để thiết lập các phép thử nghiệm loại trừ rõ ràng.
  - **Bắt buộc so sánh hiệu năng giữa:**
    - Mô hình Baseline (cơ bản) vs. Mô hình Cải tiến.
    - Kết quả Trước vs. Sau khi áp dụng: *Data Augmentation*, *Regularization*, *Focal Loss*, hoặc *Quantization*.

---

### 3. Thước Đo Đánh Giá Đa Chiều (Multi-metric Evaluation)
- **Loại bỏ sự phụ thuộc vào Accuracy:**
  - Do dữ liệu đuối nước có tính chất **mất cân bằng lớp nghiêm trọng (class imbalance)**, chỉ số *Accuracy* đơn lẻ hoàn toàn không phản ánh đúng hiệu năng thực tế.
- **Bộ chỉ số chuẩn mực:**
  - Báo cáo đầy đủ các metric toán học khách quan:
    - **mAP / IoU** cho tác vụ định vị (Localization / Detection).
    - **Precision, Recall, F1-Score** cho tác vụ phân loại (Classification).
- **Chỉ số an toàn sinh mạng (Life-safety Critical Metrics):**
  - Hai chỉ số sống còn nhất là:
    - **Recall (Độ nhạy):** Tỷ lệ không bỏ sót nạn nhân đang gặp nguy hiểm.
    - **False Alarm Rate (FAR):** Tỷ lệ báo động giả gây nhiễu cho cứu hộ.
  - Các nghiên cứu tối ưu hóa cần đo thêm **TTD (Time-to-Detection)**: Thời gian từ khi bắt đầu xuất hiện dấu hiệu bất thường đến khi hệ thống phát tín hiệu cảnh báo.

---

### 4. Yêu Cầu Triển Khai & Trực Quan Hóa (Deployment & Demo)
- **Tốc độ suy luận thời gian thực (Real-time Latency):**
  - Tính chất khẩn cấp liên quan đến sinh mạng đòi hỏi mô hình phải đạt tốc độ tối thiểu **20 – 30 FPS** trên thiết bị triển khai (Edge Devices, CCTV, Drone, GPU) để đảm bảo cảnh báo không bị trễ.
- **Đóng gói & Giao diện:**
  - Mô hình hoàn thiện cần được hỗ trợ xuất sang các định dạng chuẩn, gọn nhẹ (như **ONNX**, **TensorRT**).
  - Hệ thống ưu tiên khả năng tạo giao diện trực quan hóa dạng bounding box hoặc luồng video bằng **Streamlit** hoặc **Gradio** trong thời gian ngắn để phục vụ demo trực quan thực tế.

---

## Phần 3: Tiêu Chí Đóng Khung Cho Bài Toán Edge AI (Edge AI Constraints & Deployment Budget)

### 1. Tiêu chí Giới hạn Phần cứng Biên & Tài nguyên Tính toán (Hardware & Resource Budget)
Các thiết bị biên bị giới hạn nghiêm ngặt về VRAM, dung lượng lưu trữ và khả năng tản nhiệt. Do đó, bài toán đặt ra các tiêu chí đóng khung trực tiếp cho mô hình:
- **Dung lượng file trọng số (Model Size):** Tối đa `< 5MB – 15MB` để nạp vừa vào bộ nhớ ROM/RAM của thiết bị nhúng.
- **Số lượng tham số (Parameter Count - Params):** Đạt mức `< 2.0 – 5.0 triệu tham số` để tránh làm tràn bộ nhớ SRAM/VRAM.
- **Tải tính toán (FLOPs):** Giới hạn mức tính toán `< 6.0 – 9.0 GFLOPs` nhằm ngăn ngừa hiện tượng quá nhiệt (*thermal throttling*) và tiêu tốn điện năng trên chip biên.
- **Tốc độ suy luận & Độ trễ (Inference Speed / Latency Budget):** Tốc độ xử lý thực tế phải đạt $\ge$ **30 FPS** (hoặc Latency `< 33ms/frame`) trực tiếp trên chip nhúng.
- **Định dạng tối ưu hóa (Optimization Export):** Yêu cầu mô hình có khả năng xuất trọng số từ PyTorch (`.pt`) sang các định dạng biên như **ONNX**, **TensorRT** hoặc áp dụng **FP16 / INT8 Quantization** để tăng tốc suy luận.

---

### 2. Tiêu chí Luồng Dữ liệu Đầu vào & Tiền xử lý (Data Stream & Preprocessing Criteria)
Thiết bị Edge AI không thể xử lý trực tiếp luồng video thô 1080p/4K từ camera ngoài trời mà không bị giật lag:
- **Kích thước ảnh nạp vào Mạng (Input Resolution):** Đặt tiêu chí tự động nắn/resize ảnh thô về chuẩn **$640 \times 640$** hoặc **$416 \times 416$** pixels trước khi đưa vào mô hình.
- **Tần số trích xuất khung hình (Frame Sampling Strategy):** Thay vì bắt Edge AI phải chạy liên tục 30 FPS video thô, áp dụng chiến lược trích xuất **1 – 5 FPS** hoặc áp dụng **Cửa sổ trượt (Sliding Window 8 – 16 frames)**. Tiêu chí này giúp giảm **70 – 80% tải tính toán** mà vẫn đảm bảo bắt trọn diễn tiến hành vi đuối nước.

---

### 3. Tiêu chí Hiệu năng An toàn Cứu hộ (Safety-Critical Evaluation Metrics)
Đuối nước là bài toán liên quan trực tiếp đến tính mạng con người, do đó các chỉ số đo lường phải phản ánh tính an toàn:
- **Độ nhạy / Tỷ lệ phát hiện (Recall / Drowning Recall - DR):** Đặt tiêu chí Recall cho lớp Đuối nước phải đạt $\ge$ **90 – 95%**. Trong bài toán này, thà chấp nhận báo động nhầm (*False Positive*) còn hơn bỏ sót nạn nhân chìm dưới nước (*False Negative*).
- **Thời gian phản hồi cảnh báo sớm (Time-to-Detection - TTD):** Thời gian từ khi xuất hiện hành vi vùng vẫy đến khi hệ thống kích hoạt chuông/SMS phải đạt `< 3.2 – 5.0 giây`.
- **Lọc Báo động giả (False Alarm Reduction):** Phải tích hợp khối Chú ý (*Attention*) hoặc Bộ đệm thời gian (*Temporal Aggregation*) để triệt tiêu các ca báo động giả do sóng vỗ hay chói nắng.

---

### 4. Tiêu chí Môi trường & Góc quay Camera ngoài trời (Environmental Optical Constraints)
- **Khả năng chịu nhiễu quang học:** Mô hình phải tự động triệt tiêu được hiện tượng chói nắng mặt nước (*glare*), bọt nước và gợn sóng ngoài trời.
- **Nhận diện mục tiêu nhỏ & Che khuất (Small Targets & Occlusion):** Phải tích hợp các khối hợp nhất đặc trưng đa quy mô (như **BiFPN**, **LSKA** hay **C2PSAiSCSA**) để nhận diện được người bơi ở xa camera hoặc bị người khác che khuất một phần.

---

### 5. Tiêu chí Phạm vi Triển khai & Demo cho Đồ án (Execution Scope - 8 tuần)
Để đồ án khả thi trong 8 tuần:
- **Môi trường Demo:** Huấn luyện trên Cloud (Google Colab GPU T4), sau đó nạp file trọng số nhẹ (`.onnx` hoặc `.pt`) chạy suy luận trên Laptop / PC đóng vai trò là "Edge Node".
- **Tín hiệu Đầu ra:** Đọc luồng video ngoài trời (RTSP stream hoặc file video MP4 test) và kích hoạt Cảnh báo tại chỗ (Local Buzzer / Alert UI) ngay trên máy client mà **không cần kết nối Internet**.

---

### Bảng Tóm Tắt Tiêu Chí Khung Bài Toán (Edge AI Blueprint)

| Nhóm Tiêu chí | Thông số / Ngưỡng Đóng khung cho Đồ án | Ý nghĩa Kỹ thuật |
| :--- | :--- | :--- |
| **Kích thước File** | Dung lượng `< 15MB` (Params `< 5M`) | Dễ nạp vào VRAM/RAM thiết bị biên, tránh tràn bộ nhớ. |
| **Tải tính toán** | FLOPs `< 9.0 GFLOPs` | Tránh quá nhiệt (*thermal throttling*) & quá tải chip nhúng. |
| **Tốc độ Suy luận** | Tối thiểu $\ge$ **30 FPS** trên Edge PC | Đảm bảo phản hồi thời gian thực (*Real-time*). |
| **Độ nhạy Cứu hộ** | Recall lớp Đuối nước $\ge$ **90%** | Tuyệt đối không bỏ sót nạn nhân (*Life-safety critical*). |
| **Thời gian Cảnh báo** | Time-to-Detection (TTD) `< 4.0 giây` | Tận dụng "thời gian vàng" cứu hộ khẩn cấp. |
| **Tối ưu Luồng Video** | Input $640 \times 640$ + Frame Sampling (1-5 FPS / Sliding Window) | Tiết kiệm 70 – 80% tài nguyên xử lý trên thiết bị biên. |

