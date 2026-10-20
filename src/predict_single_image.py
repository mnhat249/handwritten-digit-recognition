import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models", "digit_model.keras")

def preprocess_real_world_image(image_path):
    # 1. Mở ảnh và chuyển sang thang độ xám (Grayscale)
    img = Image.open(image_path).convert("L")

    # 2. Tự động kiểm tra nền: Nếu đa số pixel sáng (>127) tức là nền trắng chữ đen -> Đảo ngược màu
    img_np = np.array(img)
    if np.mean(img_np) > 127:
        img = ImageOps.invert(img)
        img_np = np.array(img)

    # 3. Ngưỡng hóa (Thresholding) để loại bỏ nhiễu nền và làm rõ nét chữ
    img_np = np.where(img_np > 40, 255, 0).astype("uint8")
    img = Image.fromarray(img_np)

    # 4. CẮT BỎ KHOẢNG TRỐNG THỪA (Auto-Crop vùng chứa chữ số)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # 5. Làm dày nét chữ (MaxFilter) để khắc phục nét vẽ mảnh bằng chuột trong Paint
    # Tính toán kích thước bộ lọc dựa trên độ phân giải ảnh gốc
    filter_size = max(3, int(min(img.size) / 12))
    if filter_size % 2 == 0:
        filter_size += 1
    img = img.filter(ImageFilter.MaxFilter(filter_size))

    # 6. Đưa chữ số vào khung vuông 20x20 rồi đệm viền đen thành 28x28 (Chuẩn cấu trúc MNIST)
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    offset_x = (28 - img.width) // 2
    offset_y = (28 - img.height) // 2
    canvas.paste(img, (offset_x, offset_y))

    # 7. Chuẩn hóa về [0.0, 1.0] và reshape thành (1, 28, 28, 1)
    img_array = np.array(canvas, dtype="float32") / 255.0
    return img_array.reshape(1, 28, 28, 1)

def predict_custom_image(image_path):
    if not os.path.exists(MODEL_SAVE_PATH):
        print("Chưa tìm thấy mô hình! Hãy chạy 'py -3.11 src/train.py' trước.")
        return

    if not os.path.exists(image_path):
        print(f"Không tìm thấy file ảnh tại: {image_path}")
        return

    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    img_array = preprocess_real_world_image(image_path)

    # Dự đoán
    predictions = model.predict(img_array)
    predicted_digit = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    print(f"\n========================================")
    print(f" KẾT QUẢ DỰ ĐOÁN: Chữ số [{predicted_digit}]")
    print(f" Độ tin cậy: {confidence:.2f}%")
    print(f"========================================")

    # Hiển thị bức ảnh 28x28 thực tế mà mô hình nhìn thấy
    plt.imshow(img_array.squeeze(), cmap="gray")
    plt.title(f"Model du doan: {predicted_digit} ({confidence:.1f}%)")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    test_img_path = input("Nhập đường dẫn tới bức ảnh cần test (ví dụ: data/4.png): ").strip()
    test_img_path = test_img_path.strip('"').strip("'")
    predict_custom_image(test_img_path)