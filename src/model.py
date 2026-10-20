import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_digit_model(input_shape=(28, 28, 1), num_classes=10, lambda_reg=1e-4):
    """
    Kiến trúc mạng CNN kết hợp Data Augmentation:
    - Tự động xoay, dịch chuyển, phóng to/thu nhỏ ảnh trong lúc Train.
    - Tự động vô hiệu hóa Augmentation khi chạy Inference/Test.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # --- KHỐI DATA AUGMENTATION (MỞ RỘNG DỮ LIỆU) ---
        # Xoay ngẫu nhiên trong khoảng [-10%, +10%] (~36 độ)
        layers.RandomRotation(factor=0.10, fill_mode="constant", fill_value=0.0),
        # Dịch chuyển ngang/dọc ngẫu nhiên tối đa 10% kích thước ảnh
        layers.RandomTranslation(height_factor=0.10, width_factor=0.10, fill_mode="constant", fill_value=0.0),
        # Phóng to/thu nhỏ ngẫu nhiên từ -15% đến +15% (giúp quen với chữ to/nhỏ)
        layers.RandomZoom(height_factor=(-0.15, 0.15), width_factor=(-0.15, 0.15), fill_mode="constant", fill_value=0.0),

        # --- KHỐI TRÍCH XUẤT ĐẶC TRƯNG 1 ---
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(lambda_reg)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # --- KHỐI TRÍCH XUẤT ĐẶC TRƯNG 2 ---
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(lambda_reg)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # --- KHỐI TRÍCH XUẤT ĐẶC TRƯNG 3 (Tăng chiều sâu để học hình học phức tạp) ---
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(lambda_reg)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # --- KHỐI PHÂN LOẠI (OUTPUT LAYER) ---
        layers.Flatten(),
        layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(lambda_reg)),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])

    # Thuật toán tối ưu Adam
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model