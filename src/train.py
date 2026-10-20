"""Training script for handwritten digit recognition model."""

import os
import argparse
import tensorflow as tf
from model import build_cnn_model


def load_mnist_data():
    """Load and preprocess MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize to [0, 1] range and add channel dimension (28, 28, 1)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train[..., None]
    x_test = x_test[..., None]

    return (x_train, y_train), (x_test, y_test)


def train(epochs: int = 15, batch_size: int = 64, learning_rate: float = 1e-3):
    """Train the CNN model and save the best weights."""
    os.makedirs("saved_models", exist_ok=True)
    checkpoint_path = os.path.join("saved_models", "digit_model.keras")

    print("[INFO] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    print(f"[INFO] Train samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")

    print("[INFO] Building model...")
    model = build_cnn_model(input_shape=(28, 28, 1), num_classes=10)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    print("[INFO] Starting training...")
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    print("[INFO] Evaluating model on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[RESULT] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")
    print(f"[INFO] Best model saved to: {checkpoint_path}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train handwritten digit recognition model.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
