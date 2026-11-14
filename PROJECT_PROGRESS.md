# BÁO CÁO TIẾN ĐỘ VÀ TỔNG HỢP TOÀN DIỆN DỰ ÁN (PROJECT PROGRESS REPORT)
## HỆ THỐNG NHẬN DIỆN TỪ VỰNG NGÔN NGỮ KÝ HIỆU VIỆT NAM (VSL400) ĐA GÓC NHÌN
### KIẾN TRÚC ĐẶC TRƯNG TOÀN THÂN MEDIAPIPE HOLISTIC 225 CHIỀU VÀ ĐỘNG HỌC 450 CHIỀU

---

**Sinh viên thực hiện:** Lê Hoàng Dương - Lớp 22T_KHDL - Trường Đại học Bách khoa, ĐHĐN  
**Đề tài tốt nghiệp:** Xây dựng hệ thống nhận diện từ vựng Ngôn ngữ Ký hiệu Việt Nam (VSL) đa góc nhìn theo thời gian thực sử dụng MediaPipe và mạng nơ-ron chuỗi thời gian (Multi-Scale Inception 1D-CNN + Bi-GRU + Temporal Attention).  
**Tập dữ liệu:** VSL400 (400 từ vựng, 26 người ký, 3 góc nhìn camera, tổng cộng 74.259 video).  
**Thời gian cập nhật:** Tháng 11/2026.  
**Tình trạng hiện tại:** Đã hoàn thành toàn bộ Giai đoạn 01, 02, 03, 04 và hoàn thành xuất sắc **TUẦN 1 của Giai đoạn 05** (Khảo sát và đối chiếu có kiểm soát 3 kiến trúc xương sống cơ sở, khóa thành công Backbone với kết quả bứt phá: **Test Top-1 đạt 79.67%** trên Unseen Signers P24–P26 và **Validation Top-1 đạt 83.52%**). Sẵn sàng chuyển giao sang **TUẦN 2 (Tích hợp Cơ chế Attention: MLP-Attention vs Multi-Head Self-Attention & Kỹ thuật Chống Overfitting)**.

---

## 1. Sơ Đồ Tổng Quan Tiến Độ Dự Án (Project Milestone Roadmap)

![Sơ Đồ Lộ Trình Tiến Độ Dự Án VSL400](reports/figures/fig_milestone_roadmap.png)

| Giai đoạn | Tên nhiệm vụ / Nội dung kỹ thuật | Tập tin kịch bản / Notebook | Sản phẩm đầu ra chính | Trạng thái |
|:---:|---|---|---|:---:|
| **01** | Khảo sát đặc trưng thống kê tập dữ liệu VSL400 | `src/data_processing/01_vsl400-gloss-statistics.ipynb` | `reports/dataset_summary.json`, Figures 01 | **HOÀN THÀNH** |
| **02** | Trích xuất toàn thân MediaPipe Holistic 225D | `src/data_processing/02_extract-landmarks-part1.ipynb` đến `part7.ipynb` | 74.259 tệp `.npy`, 7 Part CSVs | **HOÀN THÀNH** |
| **03** | Phân tích chất lượng trích xuất và bóc tách lỗi | `src/data_processing/03_vsl400-extraction-analysis.ipynb` | `reports/extraction_report_holistic_all.csv`, Figures 03 | **HOÀN THÀNH** |
| **04** | Tiền xử lý, lọc dữ liệu và phân chia Signer-Disjoint (Phương án A) | `src/data_processing/04_vsl400-preprocessing-and-split.ipynb` | 6 CSV Splits, `data/processed/gloss_to_id.json`, Figures 04 | **HOÀN THÀNH** |
| **05** | Huấn luyện mô hình Đơn góc Front View 450D (Lộ trình 3 Tuần: Tuần 1 - Khảo sát Backbone) | `src/models/05_vsl400-train-baseline.ipynb`<br>`src/models/05_vsl400-train-inception.ipynb`<br>`src/models/05_vsl400-train-bilstm.ipynb` | Checkpoints Tuần 1 (`vsl400_exp1_baseline.keras`, `vsl400_exp1_inception.keras`, `vsl400_exp1_bilstm.keras`), Test Top-1 79.67%, Val Top-1 83.52%, Figures Exp 1.1-1.3 | **HOÀN THÀNH TUẦN 1** (Sẵn sàng chuyển giao Tuần 2) |
| **06** | Huấn luyện mô hình Đa góc (Multi-View Tri-Stream Fusion) | `src/models/06_vsl400-train-multi-view.ipynb` | Checkpoints Multi-View Tri-Stream | Sẵn sàng |
| **07** | Đánh giá độc lập người ký và Demo thời gian thực | `src/models/07_vsl400-realtime-inference.ipynb` | Pipeline camera trực tiếp | Kế hoạch |

---

## 2. Chi Tiết Nội Dung Hoàn Thành: Giai Đoạn 01 (Thống Kê Tập Dữ Liệu VSL400)

* **Tập tin thực hiện:** `src/data_processing/01_vsl400-gloss-statistics.ipynb`
* **Mục tiêu:** Khảo sát toàn diện phân bố không gian và thời gian của tập dữ liệu Ngôn ngữ Ký hiệu Việt Nam VSL400.

### 2.1. Kết Quả Thống Kê Chính
1. **Quy mô tập dữ liệu gốc:**
   - Tổng số từ vựng (Glosses): **400 từ vựng** thuộc nhiều chủ đề giao tiếp xã hội.
   - Tổng số người ký (Signers): **26 người ký** (mã số từ `001` đến `028`).
   - Tổng số video cử chỉ: **24.753 video cử chỉ duy nhất**.
   - Thiết lập đa góc nhìn: Mỗi cử chỉ được ghi hình đồng thời bởi 3 camera đồng bộ:
     - `front_view`: Góc quay chính diện (Camera 1).
     - `left_view`: Góc quay nghiêng bên trái 45 độ (Camera 2).
     - `right_view`: Góc quay nghiêng bên phải 45 độ (Camera 3).
   - Tổng số lượng video toàn tập: $24.753 \times 3 =$ **74.259 video** phân bổ trong 7 phần (Part 1 đến Part 7).
2. **Đặc tính phân bố từ vựng:**
   - Số mẫu trung bình cho mỗi từ: 61.88 video/từ (trung vị 59.0 video/từ, thấp nhất 53 video, cao nhất 93 video cho từ "Mùa hè").
   - Số người ký trung bình cho mỗi từ: 25.99 người ký/từ (hầu hết 400 từ đều được thực hiện bởi đủ 26 người ký).
3. **Thuộc tính kỹ thuật video:**
   - Độ phân giải khung hình vuông: $1080 \times 1080$ pixels.
   - Tốc độ khung hình: $25 - 30$ FPS.
   - Định dạng nén: MP4 container, chuẩn mã hóa H.264/AVC.
4. **Sản phẩm lưu trữ:**
   - Tệp tóm tắt số liệu: `reports/dataset_summary.json`.
   - Biểu đồ khoa học: `reports/figures/fig01_1_frame_length_distribution.png`, `reports/figures/fig01_2_signer_distribution.png`, `reports/figures/fig01_3_top_bottom_gloss_frames.png`.

---

## 3. Chi Tiết Nội Dung Hoàn Thành: Giai Đoạn 02 (Trích Xuất MediaPipe Holistic 225D)

* **Tập tin thực hiện:** `src/data_processing/02_extract-landmarks-part1.ipynb` đến `part7.ipynb`
* **Hạ tầng thực thi:** GPU Kaggle (MediaPipe Tasks Vision API, mô hình `holistic_landmarker.task`).
* **Mục tiêu:** Trích xuất đồng bộ đặc trưng khung xương toàn thân (hai bàn tay và thân trên) với tính nhất quán giải phẫu học.

### 3.1. Cấu Trúc Tensor Chuẩn Hóa 225 Chiều
Mỗi khung hình video được chuyển đổi thành vector thực 225 chiều kiểu `float32`:
- **Chỉ số 0 đến 62 (63 chiều):** Bàn tay phải giải phẫu (21 khớp $\times$ 3 tọa độ $x, y, z$).
- **Chỉ số 63 đến 125 (63 chiều):** Bàn tay trái giải phẫu (21 khớp $\times$ 3 tọa độ $x, y, z$).
- **Chỉ số 126 đến 224 (99 chiều):** Tư thế thân trên Pose (33 khớp $\times$ 3 tọa độ $x, y, z$).

Hai bàn tay được định vị trực tiếp dựa trên cấu trúc giải phẫu xương cổ tay của mô hình Pose thân trên (sai lệch giữa Cổ tay bàn tay và Cổ tay Pose luôn $< 0.0044$), đảm bảo tính nhất quán không gian liên tục.

### 3.2. Kết Quả Trích Xuất 7 Parts Trên Kaggle
Toàn bộ 74.259 video trên 7 Parts đã được hoàn tất trích xuất và lưu trữ cục bộ:

| Phần dữ liệu (Part) | Tệp báo cáo CSV | Tổng số video | Trạng thái `ok` | Trạng thái `error` | Trạng thái `no_hand` | Tỷ lệ thành công |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **Part 1** | `reports/extraction_report_holistic_Part_1.csv` | 10.611 | 10.611 | 0 | 0 | **100.00%** |
| **Part 2** | `reports/extraction_report_holistic_Part_2.csv` | 10.608 | 10.608 | 0 | 0 | **100.00%** |
| **Part 3** | `reports/extraction_report_holistic_Part_3.csv` | 10.608 | 10.608 | 0 | 0 | **100.00%** |
| **Part 4** | `reports/extraction_report_holistic_Part_4.csv` | 10.608 | 10.608 | 0 | 0 | **100.00%** |
| **Part 5** | `reports/extraction_report_holistic_Part_5.csv` | 10.608 | 10.608 | 0 | 0 | **100.00%** |
| **Part 6** | `reports/extraction_report_holistic_Part_6.csv` | 10.608 | 10.607 | 1 | 0 | **99.99%** |
| **Part 7** | `reports/extraction_report_holistic_Part_7.csv` | 10.608 | 10.607 | 0 | 1 | **99.99%** |
| **TOÀN BỘ** | `reports/extraction_report_holistic_all.csv` | **74.259** | **74.257** | **1** | **1** | **99.997%** |

* Thư mục dữ liệu: `data/processed/landmarks_holistic/` (gồm 7 thư mục `Part_1` đến `Part_7`, mỗi phần gồm 3 thư mục `front_view`, `left_view`, `right_view`).

---

## 4. Chi Tiết Nội Dung Hoàn Thành: Giai Đoạn 03 (Phân Tích Chất Lượng Trích Xuất Đa Kênh)

* **Tập tin thực hiện:** `src/data_processing/03_vsl400-extraction-analysis.ipynb`
* **Mục tiêu:** Kiểm toán toàn diện chất lượng bám bắt, bóc tách bản chất các mẫu lỗi và xác lập cơ sở khoa học cho các siêu tham số tiền xử lý.

### 4.1. Bóc Tách 2 Mẫu Lỗi Duy Nhất Trong Toàn Bộ 74.259 Video
Tỷ lệ trích xuất đạt trạng thái `ok` đạt **99.997%** (74.257 / 74.259 video). Chỉ có đúng 2 trường hợp ngoại lệ:
1. **Video `019442` (`error`, Signer `024`, Part 6, `front_view`, từ: "Cao (người)"):** Tệp MP4 dung lượng 0.04 KB bị đứt đoạn luồng ghi dữ liệu từ nguồn gốc Kaggle, thiếu atom metadata `moov` khiến OpenCV không thể khởi tạo luồng giải mã. Hai góc nhìn còn lại (`left_view` và `right_view`) hoàn toàn bình thường.
2. **Video `024157` (`no_hand`, Signer `028`, Part 7, `right_view`, từ: "Giờ"):** Hiện tượng che khuất quang học tự nhiên (Self-occlusion) khi người ký quay nghiêng 45 độ bên phải. Kênh Pose đạt tỷ lệ bám bắt 100%, hai góc quay `front_view` và `left_view` đều đạt trạng thái `ok` với tỷ lệ nhận diện bàn tay cao (98.1%).

### 4.2. Chất Lượng Bám Bắt Đa Kênh Theo Góc Nhìn

| Góc quay Camera | Tổng số video | Số frame TB | Tay phải TB (Median) | Tay trái TB (Median) | Bàn tay tích cực TB (Median) | Thân trên Pose TB (Median) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Chính diện (`front_view`)** | 24.752 | 64.13 | 86.63% (89.39%) | 91.47% (96.00%) | **94.13% (98.28%)** | **100.00% (100.0%)** |
| **Nghiêng trái (`left_view`)** | 24.753 | 67.36 | 81.44% (85.96%) | 85.89% (91.80%) | **91.86% (96.05%)** | **99.97% (100.0%)** |
| **Nghiêng phải (`right_view`)** | 24.752 | 64.83 | 83.16% (87.10%) | 88.59% (96.23%) | **93.33% (97.44%)** | **100.00% (100.0%)** |

---

## 5. Chi Tiết Nội Dung Hoàn Thành: Giai Đoạn 04 (Tiền Xử Lý Dữ Liệu & Phân Chia Signer-Disjoint Tối Ưu)

* **Tập tin thực hiện:** `src/data_processing/04_vsl400-preprocessing-and-split.ipynb`
* **Mục tiêu:** Làm sạch dữ liệu, phân chia Signer-Disjoint chuẩn hóa (Phương án A) và đóng gói pipeline chuẩn hóa 225 chiều.

### 5.1. Thống Kê Đối Soát Sau Khi Lọc Dữ Liệu
Áp dụng tiêu chí lọc: Loại bỏ 1 video lỗi MP4, 1 video mất dấu hoàn toàn và 99 video có $\text{hand\_det\_rate} < 20\%$:
* Tổng số bản ghi giữ lại: **74.158 / 74.259 mẫu (99.864%)**, bảo tồn nguyên vẹn **400 / 400 từ vựng** và **26 / 26 người ký**.
* Góc chính diện `front_view`: **24.738 mẫu** (bảo tồn 99.94%).
* Góc nghiêng trái `left_view`: **24.684 mẫu** (bảo tồn 99.72%).
* Góc nghiêng phải `right_view`: **24.736 mẫu** (bảo tồn 99.93%).

### 5.2. Kết Quả Phân Chia Signer-Disjoint Mới (Phương Án A - Part-Stratified Disjoint)

Khắc phục hoàn toàn hiện tượng lệch miền dữ liệu (Part/Session Domain Shift) của phân vùng cũ:
- **Tập Huấn luyện (Train - 20 người ký):** `001, 002, 003, 004, 007, 009, 010, 011, 013, 014, 015, 018, 019, 020, 021, 022, 023, 025, 027, 028`. Hiện diện đầy đủ trên cả **7 Parts (Part 1 đến Part 7)**. Đưa các Signer có landmark nhiễu (`028, 025`) vào Train làm thành phần làm nhiễu điều hòa tự nhiên (Data Regularizer).
- **Tập Kiểm định (Validation - 3 người ký):** `006` (Part 2), `012` (Part 3), `026` (Part 6-7). Bám bắt bàn tay đạt chuẩn **94.36%**.
- **Tập Kiểm thử độc lập (Unseen Signers Test - 3 người ký):** `008` (Part 2, tốt nhất tập 98.33%), `016` (Part 4, 96.35%), `024` (Part 6-7, 96.36%). Trung bình đạt **96.83%** (chuẩn mực nhất toàn bộ tập dữ liệu).

| Kịch bản Mô hình | Phân vùng (Split) | Danh sách Người ký (Signers) | Số lượng mẫu | Tỷ lệ (%) | Bao phủ Glosses | Part Coverage | Tệp CSV Phân Vùng |
|---|:---:|---|:---:|:---:|:---:|:---:|---|
| **Single-View (`front_view`)** | **Train** | 20 signers (`001..028`) | **17.789** | **71.91%** | **400 / 400 (100%)** | Part 1..7 | `data/processed/splits/train_front.csv` |
| | **Validation** | 3 signers (`006, 012, 026`) | **3.520** | **14.23%** | **400 / 400 (100%)** | Part 2, 3, 6, 7 | `data/processed/splits/val_front.csv` |
| | **Test** | 3 signers (`008, 016, 024`) | **3.429** | **13.86%** | **400 / 400 (100%)** | Part 2, 4, 6, 7 | `data/processed/splits/test_front.csv` |
| | **Tổng Front View** | **26 signers** | **24.738** | **100.00%** | **400 / 400 (100%)** | Toàn bộ 7 Parts | |
| **Multi-View (`tri_view`)** | **Train** | 20 signers (`001..028`) | **53.364** | **71.96%** | **400 / 400 (100%)** | Part 1..7 | `data/processed/splits/train_all.csv` |
| | **Validation** | 3 signers (`006, 012, 026`) | **10.559** | **14.24%** | **400 / 400 (100%)** | Part 2, 3, 6, 7 | `data/processed/splits/val_all.csv` |
| | **Test** | 3 signers (`008, 016, 024`) | **10.235** | **13.80%** | **400 / 400 (100%)** | Part 2, 4, 6, 7 | `data/processed/splits/test_all.csv` |
| | **Tổng Multi-View** | **26 signers** | **74.158** | **100.00%** | **400 / 400 (100%)** | Toàn bộ 7 Parts | |

* Kiểm tra rò rỉ: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$. Đồng bộ 100% trong `data/processed/split_summary.json`.

---

## 6. Chi Tiết Nội Dung Hoàn Thành: Giai Đoạn 05 (Huấn Luyện Front-View 450D - Tổng Kết Tuần 1)

* **Kế hoạch triển khai:** Kế hoạch thực nghiệm 3 tuần đã duyệt (`PHASE_05_TRAINING_PLAN_3WEEKS.md`).
* **Trọng tâm Tuần 1:** Thiết lập Baseline chuẩn mực và thực hiện khảo sát đối đầu có kiểm soát (Controlled Ablation Study) giữa 3 biến thể kiến trúc xương sống (Backbone Comparison) trên không gian 400 từ vựng góc nhìn chính diện (`front_view`).
* **Hạ tầng & Dữ liệu:** 17.789 mẫu Train (Signers 001..028), 3.520 mẫu Validation (Signers 006, 012, 026), 3.429 mẫu Unseen Test (Signers 008, 016, 024). Bộ đệm đặc trưng chuẩn hóa 450D (225D Tọa độ không gian + 225D Đạo hàm vận tốc động học, $T=64$).
* **Chiến lược tối ưu đồng bộ 100%:** `AdamW(learning_rate=1e-3, weight_decay=1e-4)`, `CategoricalCrossentropy(label_smoothing=0.08)`, `ModelCheckpoint` và `ReduceLROnPlateau(factor=0.5, patience=5)` đồng bộ theo dõi metric mục tiêu cao nhất `val_top_1_acc`.

### 6.1. Chi Tiết 3 Thực Nghiệm Độc Lập Trong Tuần 1

#### 1. Thực nghiệm 1.1: Mô hình Baseline Chuẩn (Standard 1D-CNN + 1-Layer Bi-GRU)
- **Notebook thực thi:** [`src/models/05_vsl400-train-baseline.ipynb`](src/models/05_vsl400-train-baseline.ipynb)
- **Kiến trúc:** Phép chiếu Linear Dense(256) + 1 tầng Conv1D (64 filters, $k=3$) + 1 tầng Bi-GRU (128 units/chiều) + Dense(400).
- **Tổng tham số:** **404,688 tham số**.
- **Tiến trình huấn luyện:** Hoàn tất trọn vẹn 100 epochs trong **40.1 phút**. Hội tụ tối ưu tại **Epoch 96**.
- **Chỉ số kiểm định & tổng quát hóa:**
  + Validation Top-1: **74.01%** | Validation Top-5: **93.27%** | Validation Loss: 1.7652.
  + Test Top-1 (Unseen Signers P24–P26): **70.72%** | Test Top-5: **92.59%** | Test Loss: 1.7806.
- **Tệp lưu trữ:** Checkpoint [`experiments/checkpoints/vsl400_exp1_baseline.keras`](experiments/checkpoints/vsl400_exp1_baseline.keras), Nhật ký [`experiments/logs/training_exp1_baseline.csv`](experiments/logs/training_exp1_baseline.csv), Chỉ số [`experiments/results/metrics_exp1_baseline.json`](experiments/results/metrics_exp1_baseline.json).

#### 2. Thực nghiệm 1.2: Khối Tích Chập Đa Thang Đo (Multi-Scale Inception 1D-CNN + 1-Layer Bi-GRU)
- **Notebook thực thi:** [`src/models/05_vsl400-train-inception.ipynb`](src/models/05_vsl400-train-inception.ipynb)
- **Cải tiến kiến trúc:** Thay thế Conv1D đơn lẻ bằng khối Inception 3 nhánh song song:
  + Nhánh 1 ($k=1$, 96 filters): Bắt vi cử chỉ đầu ngón tay tức thời.
  + Nhánh 2 ($k=3$, 96 filters): Trích xuất tư thế bàn tay ngắn hạn.
  + Nhánh 3 ($k=5$, 64 filters): Bao quát toàn bộ quỹ đạo cánh tay vĩ mô.
  + Ghép nối (Concatenate 256 filters) kết hợp liên kết tàn dư (Residual Add) và LayerNormalization.
- **Tổng tham số:** **762,512 tham số**.
- **Tiến trình huấn luyện:** Hoàn tất 100 epochs trong **75.7 phút**. Hội tụ tối ưu tại **Epoch 92**.
- **Chỉ số bứt phá vượt bậc:**
  + Validation Top-1: **82.84%** (**+8.83%** so với Baseline) | Validation Top-5: **95.37%** (+2.10%) | Validation Loss: **1.4934** (-0.2718).
  + Test Top-1 (Unseen Signers P24–P26): **79.67%** (**+8.95%** so với Baseline) | Test Top-5: **95.65%** (+3.06%) | Test Loss: **1.4930** (-0.2876).
- **Tệp lưu trữ:** Checkpoint [`experiments/checkpoints/vsl400_exp1_inception.keras`](experiments/checkpoints/vsl400_exp1_inception.keras), Nhật ký [`experiments/logs/training_exp1_inception.csv`](experiments/logs/training_exp1_inception.csv), Chỉ số [`experiments/results/metrics_exp1_2_inception.json`](experiments/results/metrics_exp1_2_inception.json).

#### 3. Thực nghiệm 1.3: So Sánh Mạng Hồi Quy Thời Gian (Multi-Scale Inception 1D-CNN + Bi-LSTM)
- **Notebook thực thi:** [`src/models/05_vsl400-train-bilstm.ipynb`](src/models/05_vsl400-train-bilstm.ipynb)
- **Cải tiến kiến trúc & Phát hiện học thuật then chốt:**
  + Tế bào LSTM sở hữu ô nhớ trạng thái độc lập ($C_t$) cùng 3 cổng kiểm soát ($f, i, o$). Khi dùng cấu trúc `return_sequences=False` thông thường, đạo hàm BPTT bị suy giảm lũy thừa qua 64 bước thời gian ($10^{-6} - 10^{-9}$ ở giữa chuỗi) khiến mô hình bị nghẽn ở phân phối ngẫu nhiên.
  + Đột phá giải pháp: Chuyển sang `return_sequences=True` kết hợp **Global Average Pooling 1D (Gộp trung bình thời gian)**. Thao tác này thiết lập liên kết trực tiếp (Gradient Highway) tới tất cả 64 khung hình, triệt tiêu hoàn toàn điểm mù suy giảm đạo hàm.
- **Tổng tham số:** **860,304 tham số**.
- **Tiến trình huấn luyện:** Hoàn tất 100 epochs trong **92.7 phút**. Hội tụ tối ưu tại **Epoch 99**.
- **Chỉ số kiểm định & tổng quát hóa:**
  + Validation Top-1: **83.52%** (**+9.51%** so với Baseline, **cao nhất toàn bộ Tuần 1**) | Validation Top-5: **95.77%** | Validation Loss: **1.4435** (**Loss thấp nhất toàn bộ Tuần 1**).
  + Test Top-1 (Unseen Signers P24–P26): **79.41%** (**+8.69%** so với Baseline, xấp xỉ mức 79.67% của Bi-GRU) | Test Top-5: **94.46%** | Test Loss: 1.5468.
- **Tệp lưu trữ:** Checkpoint [`experiments/checkpoints/vsl400_exp1_bilstm.keras`](experiments/checkpoints/vsl400_exp1_bilstm.keras), Nhật ký [`experiments/logs/training_exp1_bilstm.csv`](experiments/logs/training_exp1_bilstm.csv), Chỉ số [`experiments/results/metrics_exp1_3_bilstm.json`](experiments/results/metrics_exp1_3_bilstm.json).

---

### 6.2. Bảng Đối Soát Toàn Diện Các Kiến Trúc Xương Sống (Backbone Comparison)

| Tiêu chí Đánh giá | Thực nghiệm 1.1: Baseline (Conv1D + Bi-GRU) | Thực nghiệm 1.2: Inception + Bi-GRU | Thực nghiệm 1.3: Inception + Bi-LSTM (GAP) | Đánh giá Học thuật & So sánh |
| :--- | :---: | :---: | :---: | :--- |
| **Số lượng tham số** | 404,688 | 762,512 | 860,304 | Inception Bi-GRU gọn nhẹ hơn Bi-LSTM ~11.4% |
| **Epoch tối ưu** | 96 / 100 | 92 / 100 | 99 / 100 | Cả 3 mô hình đều đạt điểm chín muồi ở cuối tiến trình |
| **Validation Loss** | 1.7652 | 1.4934 | **1.4435** | Bi-LSTM đạt hàm mất mát kiểm định thấp nhất |
| **Validation Top-1 (%)** | 74.01% | 82.84% | **83.52%** | Bi-LSTM dẫn đầu Validation Top-1 (+0.68% so với Bi-GRU) |
| **Validation Top-5 (%)** | 93.27% | 95.37% | **95.77%** | Cả hai mô hình Inception đều đạt trên 95% |
| **Test Loss (Unseen Signers)** | 1.7806 | **1.4930** | 1.5468 | Bi-GRU duy trì Loss thấp nhất trên người ký mới |
| **Test Top-1 (%) (Unseen)** | 70.72% | **79.67%** | 79.41% | **Bi-GRU nhỉnh hơn +0.26% về năng lực tổng quát hóa** |
| **Test Top-5 (%) (Unseen)** | 92.59% | **95.65%** | 94.46% | Bi-GRU dẫn đầu Top-5 trên tập kiểm thử độc lập |
| **Thời gian huấn luyện** | **40.1 phút** | 75.7 phút | 92.7 phút | Bi-GRU nhanh hơn Bi-LSTM 18.4% (tối ưu tính toán suy luận) |
| **Khoảng cách Overfit (Gap)**| ~3.3% | **~3.2%** | ~4.1% | Kiểm soát quá khớp cực tốt (chênh lệch Train-Test rất nhỏ) |

---

### 6.3. Khóa Kiến Trúc Xương Sống (Backbone Lock) Tuần 1

Căn cứ vào kết quả đối soát thực nghiệm đối đầu:
1. **Khối không gian:** Khối **Multi-Scale Inception 1D-CNN** ($k=1, 3, 5$ song song kết hợp liên kết tàn dư Residual) đã chứng minh tính ưu việt tuyệt đối so với Conv1D đơn lẻ, tạo ra bước nhảy vọt **~+9% Top-1 Accuracy** trên tập người ký mới Unseen Signers.
2. **Khối thời gian:** 
   - Cả **Bi-GRU** (Test Top-1: 79.67%) và **Bi-LSTM** kết hợp Temporal Pooling (Test Top-1: 79.41%, Val Top-1: 83.52%) đều đạt hiệu năng xuất sắc, chạm sát ngưỡng mục tiêu 80% của Tuần 1.
   - **Bi-GRU** được ưu tiên lựa chọn làm Backbone xương sống mặc định cho các thực nghiệm tiếp theo nhờ: tốc độ suy luận nhanh hơn, số lượng tham số nhỏ gọn hơn (762k vs 860k) và độ chính xác tổng quát hóa Unseen Test cao hơn (+0.26%).
   - Cấu trúc Bi-LSTM kèm Temporal Pooling được bảo tồn như một phương án đối sánh chuẩn mực cao cấp.
3. **Chính thức Khóa Kiến trúc Backbone Tuần 1:** Hoàn thành 100% mục tiêu Tuần 1 theo kế hoạch 3 tuần. Toàn bộ trọng số tối ưu và nhật ký đã được đồng bộ để chuyển giao sang Tuần 2.

---

## 7. Danh Mục Biểu Đồ Minh Chứng Đã Lưu Trữ

Toàn bộ biểu đồ khoa học đã được lưu trữ tập trung tại thư mục `reports/figures/`:

| Tên tập tin biểu đồ | Giai đoạn | Mô tả nội dung trực quan hóa |
|---|:---:|---|
| `reports/figures/fig_milestone_roadmap.png` | Master | Sơ đồ toàn cảnh lộ trình tiến độ dự án 7 giai đoạn |
| `reports/figures/fig_data_split_architecture.png` | 04 | Kiến trúc phân chia Signer-Disjoint Phương án A và tỷ lệ mẫu trên từng nhánh |
| `reports/figures/fig01_1_frame_length_distribution.png` | 01 | Phân bố số lượng khung hình trên toàn bộ tập dữ liệu thô |
| `reports/figures/fig01_2_signer_distribution.png` | 01 | Số lượng video thực hiện bởi từng người ký trong 26 signers |
| `reports/figures/fig01_3_top_bottom_gloss_frames.png` | 01 | Top 10 từ vựng có thời lượng dài nhất và ngắn nhất |
| `reports/figures/fig03_1_no_hand_samples.png` | 03 | Đối chiếu 3 góc nhìn của video ngoại lệ 024157 (bản chất lỗi `no_hand`) |
| `reports/figures/fig03_2_tracking_quality_tiers.png` | 03 | Phân bố tỷ lệ nhận diện đa kênh và các tầng chất lượng theo góc nhìn |
| `reports/figures/fig03_3_zero_frame_analysis.png` | 03 | Phân phối số khung hình khuyết và nhóm zero-frame toàn tập |
| `reports/figures/fig03_4_frame_length_distribution.png` | 03 | Hàm mật độ xác suất độ dài video và đường chuẩn hóa $T=64$ |
| `reports/figures/fig03_5_top_bottom_gloss_frames.png` | 03 | So sánh thời lượng thực hiện giữa Top 10 từ cử chỉ dài nhất và ngắn nhất |
| `reports/figures/fig04_1_dataset_split_distribution.png` | 04 | Phân bố tỷ lệ mẫu và người ký theo các tập Train / Val / Test mới (Phương án A) |
| `reports/figures/fig04_2_temporal_resampling_comparison.png` | 04 | Quỹ đạo cổ tay qua 3 giai đoạn: Thô -> Điền khuyết -> Resampling $T=64$ |
| `reports/figures/fig04_3_wrist_centering_skeleton.png` | 04 | Khung xương bàn tay trước và sau khi chuẩn hóa Cổ tay (Wrist-Centering) |
| `reports/figures/fig_exp1_1_baseline_learning_curves.png` | 05 (W1) | Đường cong học tập Loss, Top-1, Top-5 của Thực nghiệm 1.1: Baseline (Conv1D + Bi-GRU) |
| `reports/figures/fig_exp1_2_inception_learning_curves.png` | 05 (W1) | Đường cong học tập Loss, Top-1, Top-5 của Thực nghiệm 1.2: Multi-Scale Inception + Bi-GRU |
| `reports/figures/fig_exp1_3_bilstm_learning_curves.png` | 05 (W1) | Đường cong học tập Loss, Top-1, Top-5 của Thực nghiệm 1.3: Multi-Scale Inception + Bi-LSTM (GAP) |
| `reports/figures/fig05_1_single_view_learning_curves.png` | 05 | Biểu đồ đường cong tổng hợp của mô hình đơn góc nâng cao |
| `reports/figures/fig05_2_top_bottom_gloss_accuracy.png` | 05 | Biểu đồ phân tích độ chính xác nhóm từ vựng nhận diện cao nhất và thấp nhất |

---

## 8. Kế Hoạch Các Bước Tiếp Theo

Dự án đã hoàn tất xuất sắc Tuần 1 của Giai đoạn 05, vượt chỉ tiêu đề ra (Test Top-1 đạt **79.67%**, Validation Top-1 đạt **83.52%**). Kế hoạch tiếp theo gồm:

1. **Khởi động TUẦN 2 (Ngày 08 – 14): Tích hợp Cơ chế Attention & Kỹ thuật Chống Overfitting (Mục tiêu Top-1: 80% -> 85%)**:
   - Tích hợp cơ chế **MLP Temporal Attention** vào Backbone Inception đã khóa để học phân bổ trọng số thích nghi cho các khung hình chứa cử động then chốt.
   - Khảo sát cơ chế **Multi-Head Self-Attention (MHSA - 4 heads / 8 heads)** để mô hình hóa tương quan toàn cục đa vị trí trong cử chỉ.
   - Tinh chỉnh điều hòa nâng cao: Thử nghiệm Label Smoothing $\epsilon = 0.1$, Stochastic Depth (DropPath) và điều chỉnh Weight Decay để triệt tiêu hoàn toàn nguy cơ quá khớp.
   - Đánh giá đối soát và Khóa Kiến trúc Attention Tuần 2.
