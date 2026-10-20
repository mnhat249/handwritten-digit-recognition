import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from model import build_digit_model

# 1. Định nghĩa đường dẫn file dựa trên cấu trúc dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models", "digit_model.keras")

def load_and_preprocess_data(csv_path, cv_ratio=0.2):
    print("Đang đọc dữ liệu từ:", csv_path)
    df = pd.read_csv(csv_path)

    # Tách nhãn y (cột 'label') và ma trận điểm ảnh X (784 cột còn lại)
    y_full = df["label"].values
    X_full = df.drop(columns=["label"]).values

    # Chuẩn hóa pixel từ [0, 255] về [0.0, 1.0] và định hình về (N, 28, 28, 1)
    X_full = X_full.astype("float32") / 255.0
    X_full = X_full.reshape(-1, 28, 28, 1)

    # Chia tập Train (80%) và Cross-Validation (20%) để đánh giá Bias/Variance
    split_idx = int(len(X_full) * (1 - cv_ratio))
    X_train, X_cv = X_full[:split_idx], X_full[split_idx:]
    y_train, y_cv = y_full[:split_idx], y_full[split_idx:]

    print(f"Kích thước tập Train: {X_train.shape} | Tập CV: {X_cv.shape}")
    return X_train, y_train, X_cv, y_cv

def main():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Bước 1: Nạp và tiền xử lý dữ liệu
    X_train, y_train, X_cv, y_cv = load_and_preprocess_data(TRAIN_CSV_PATH)

    # Bước 2: Khởi tạo mô hình
    model = build_digit_model(lambda_reg=1e-4)
    model.summary()

    # Tự động lưu lại bản model có độ chính xác trên tập CV cao nhất
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    # Bước 3: Huấn luyện mô hình
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=64,
        validation_data=(X_cv, y_cv),
        callbacks=[checkpoint_cb]
    )

    # Bước 4: Chẩn đoán kết quả J_train và J_cv
    train_acc = history.history["accuracy"][-1]
    cv_acc = history.history["val_accuracy"][-1]
    print(f"\nHoàn tất! Độ chính xác Train: {train_acc*100:.2f}% | CV: {cv_acc*100:.2f}%")
    print(f"Mô hình tốt nhất đã được lưu tại: {MODEL_SAVE_PATH}")

    # Vẽ biểu đồ Loss & Accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="J_train (Train Loss)")
    plt.plot(history.history["val_loss"], label="J_cv (CV Loss)")
    plt.title("Biểu đồ Hàm mất mát (Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="CV Accuracy")
    plt.title("Biểu đồ Độ chính xác (Accuracy)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()  