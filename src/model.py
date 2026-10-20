import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_digit_model(input_shape=(28, 28, 1), num_classes=10, lambda_reg=1e-4):
    """
    Khởi tạo kiến trúc mạng CNN phân loại chữ số 0-9:
    - Tầng đầu: học cạnh, đường viền cơ bản.
    - Tầng giữa: ghép thành đường cong, vòng khép kín.
    - Tầng cuối: 10 units tương ứng nhãn 0-9.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Khối trích xuất đặc trưng 1 (Low-level features)
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(lambda_reg)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Khối trích xuất đặc trưng 2 (Mid-level features)
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(lambda_reg)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Khối tổng hợp đặc trưng bậc cao & Phân loại (Output layer)
        layers.Flatten(),
        layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(lambda_reg)),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    # Sử dụng thuật toán tối ưu Adam (kết hợp Momentum và RMSprop)
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