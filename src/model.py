"""Model definition for handwritten digit recognition."""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_cnn_model(
    input_shape: tuple = (28, 28, 1),
    num_classes: int = 10,
    l2_reg: float = 1e-4,
) -> tf.keras.Model:
    """Build and return a Convolutional Neural Network (CNN) with L2 regularization."""
    regularizer = regularizers.l2(l2_reg) if l2_reg > 0 else None

    model = models.Sequential(
        [
            # Block 1
            layers.Input(shape=input_shape),
            layers.Conv2D(
                32,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizer,
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(
                32,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizer,
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Block 2
            layers.Conv2D(
                64,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizer,
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(
                64,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizer,
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Fully connected head
            layers.Flatten(),
            layers.Dense(128, kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


if __name__ == "__main__":
    test_model = build_cnn_model()
    test_model.summary()
