import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_CSV_PATH = os.path.join(BASE_DIR, "data", "test.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models", "digit_model.keras")

def run_inference():
    if not os.path.exists(MODEL_SAVE_PATH):
        print("Chưa tìm thấy mô hình! Hãy chạy 'python src/train.py' trước.")
        return

    # 1. Tải mô hình đã huấn luyện
    print("Đang nạp mô hình từ:", MODEL_SAVE_PATH)
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    # 2. Đọc dữ liệu test.csv (không có cột nhãn label)
    print("Đang đọc dữ liệu test từ:", TEST_CSV_PATH)
    df_test = pd.read_csv(TEST_CSV_PATH)
    X_test = df_test.values.astype("float32") / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1)

    # 3. Chọn ngẫu nhiên 5 bức ảnh trong tập Test để dự đoán
    random_indices = np.random.choice(len(X_test), size=5, replace=False)
    sample_images = X_test[random_indices]

    # Dự đoán xác suất cho 10 chữ số (0-9)
    predictions = model.predict(sample_images)

    # 4. Hiển thị kết quả
    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(random_indices):
        pred_digit = np.argmax(predictions[i])
        confidence = np.max(predictions[i]) * 100

        plt.subplot(1, 5, i + 1)
        plt.imshow(sample_images[i].squeeze(), cmap="gray")
        plt.title(f"Dự đoán: {pred_digit}\n({confidence:.1f}%)")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_inference()