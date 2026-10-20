import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageOps

# Đường dẫn tới mô hình đã huấn luyện
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models", "digit_model.keras")

def predict_custom_image(image_path):
    if not os.path.exists(MODEL_SAVE_PATH):
        print("Chưa tìm thấy mô hình! Hãy chạy 'python src/train.py' trước.")
        return

    if not os.path.exists(image_path):
        print(f"Không tìm thấy file ảnh tại đường dẫn: {image_path}")
        return

    # 1. Tải mô hình đã lưu
    print("Đang nạp mô hình...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    # 2. Tiền xử lý ảnh đầu vào
    # Mở ảnh và chuyển sang ảnh thang độ xám (Grayscale - 'L')
    img = Image.open(image_path).convert("L")
    
    # Thay đổi kích thước về 28x28 pixels
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # ĐẢO NGƯỢC MÀU (Rất quan trọng nếu bạn chụp ảnh chữ đen trên nền giấy trắng):
    # MNIST dùng nền đen (0) và nét chữ trắng (255). 
    # Nếu ảnh của bạn là nền trắng nét đen, cần đảo ngược lại.
    img = ImageOps.invert(img)

    # Chuyển ảnh thành mảng numpy và chuẩn hóa giá trị về [0.0, 1.0]
    img_array = np.array(img, dtype="float32") / 255.0
    
    # Định hình lại kích thước thành (1, 28, 28, 1) để khớp với đầu vào của CNN
    img_array = img_array.reshape(1, 28, 28, 1)

    # 3. Dự đoán (Inference)
    predictions = model.predict(img_array)
    predicted_digit = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    print(f"\n========================================")
    print(f" KẾT QUẢ DỰ ĐOÁN: Chữ số [{predicted_digit}]")
    print(f" Độ tin cậy: {confidence:.2f}%")
    print(f"========================================")

    # Hiển thị ảnh sau khi xử lý để kiểm tra mắt thường
    plt.imshow(img_array.squeeze(), cmap="gray")
    plt.title(f"Model du doan: {predicted_digit} ({confidence:.1f}%)")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Điền đường dẫn tới bức ảnh bạn muốn test vào đây
    # Ví dụ: đường dẫn tới một ảnh lưu trong thư mục data hoặc ngoài máy tính
    test_img_path = input("Nhập đường dẫn tuyệt đối hoặc tương đối tới bức ảnh cần test: ").strip()
    # Loại bỏ dấu ngoặc kép nếu bạn lỡ copy "Copy as Path" trên Windows
    test_img_path = test_img_path.strip('"').strip("'")
    
    predict_custom_image(test_img_path)