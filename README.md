# Handwritten Digit Recognizer (0-9)

Dự án phân loại chữ số viết tay từ 0 đến 9 sử dụng Mạng nơ-ron tích chập (CNN) và kỹ thuật Transfer Learning trên nền tảng TensorFlow/Keras.

## 📌 Tính năng chính
- Xử lý và chuẩn hóa ảnh thang độ xám kích thước 28x28.
- Kiến trúc CNN tùy biến kết hợp L2 Regularization để kiểm soát Overfitting.
- Thuật toán tối ưu hóa: **Adam Optimizer** (kết hợp Momentum và RMSprop).
- Phân tích chẩn đoán lỗi mô hình qua tập Train và Cross-Validation (Bias & Variance).
- Hỗ trợ inference dự đoán nhanh nhãn chữ số cho ảnh đơn lẻ.

## 📂 Cấu trúc thư mục
```plaintext
digit-recognizer/
├── data/                 # Chứa dữ liệu (train.csv, test.csv, mnist.npz...)
├── notebooks/            # Chứa các file thử nghiệm (.ipynb)
├── src/                  # Mã nguồn chính (.py)
│   ├── model.py          # Định nghĩa kiến trúc mạng CNN / Transfer Learning
│   ├── train.py          # Script nạp data và huấn luyện model
│   └── inference.py      # Script nhận diện một ảnh chữ số bất kỳ
├── saved_models/         # Nơi lưu checkpoint model (.keras, .h5)
├── .gitignore            # Khai báo các file KHÔNG đưa lên GitHub
├── requirements.txt      # Danh sách các thư viện cần cài đặt
└── README.md             # Giới thiệu dự án, kiến trúc và hướng dẫn chạy
```

## 🚀 Cài đặt & Hướng dẫn chạy

### 1. Clone repository
```bash
git clone https://github.com/<username>/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

### 2. Thiết lập môi trường ảo và cài đặt thư viện
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Cài đặt phụ thuộc
pip install -r requirements.txt
```

### 3. Huấn luyện mô hình
```bash
python src/train.py
```
Checkpoint tốt nhất sẽ được tự động lưu vào thư mục `saved_models/digit_model.keras`.

### 4. Dự đoán chữ số trên ảnh thực tế (Inference)
```bash
python src/inference.py --image path/to/digit_image.png
```

## 🧠 Kiến trúc mô hình
Mô hình CNN được xây dựng theo kiến trúc nhiều tầng tích chập và pooling:
1. **Input Layer:** Tensor ảnh `(28, 28, 1)` với pixel chuẩn hóa về khoảng `[0, 1]`.
2. **Convolutional Blocks:** Tích hợp `Conv2D`, `BatchNormalization`, `ReLU`, và `MaxPooling2D`.
3. **Regularization:** Áp dụng `L2 Regularization` (Weight Decay) và `Dropout` nhằm hạn chế hiện tượng Overfitting.
4. **Classification Head:** Tầng `Dense` với hàm kích hoạt `Softmax` xuất ra xác suất cho 10 lớp chữ số (0 - 9).
