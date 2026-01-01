#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CNN on CIFAR-10: Preprocess -> Define -> Train -> Evaluate -> Visualize

This single script:
  1) Loads & preprocesses CIFAR-10 (preview, normalize, one-hot)
  2) Builds the CNN (Sequential model: Conv/BatchNorm/MaxPool/Dropout blocks -> Dense head)
  3) Compiles (Adam, categorical cross-entropy, accuracy)
  4) Trains for N epochs with validation split; saves curves
  5) Evaluates on test set; optionally saves model & confusion matrix

Usage:
    python cnn_cifar10_full.py --model compact --epochs 20 --batch-size 64 --val-split 0.1 --save-model --plot-cm

Author: Rahul
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, Dropout,
    Flatten, Dense
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Optional: confusion matrix
try:
    from sklearn.metrics import confusion_matrix
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -----------------------
# Globals & Config
# -----------------------
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------
# Utilities
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 CNN in Keras/TensorFlow.")
    parser.add_argument("--model", choices=["compact", "stepwise"], default="compact",
                        help="Choose the CNN architecture implementation.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of training data to use for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--save-model", action="store_true", help="Save trained model to outputs.")
    parser.add_argument("--plot-cm", action="store_true", help="Plot confusion matrix (requires scikit-learn).")
    parser.add_argument("--preview", action="store_true", help="Preview first 10 training images.")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    # For stricter determinism, uncomment:
    # import tensorflow as tf
    # tf.random.set_seed(seed)


# -----------------------
# Step 1: Preprocessing
# -----------------------
def load_and_preprocess():
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Preview shapes
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_test : {x_test.shape},  y_test : {y_test.shape}")

    # Normalize to [0, 1]
    print("Normalizing pixel values to [0, 1]...")
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # One-hot encode labels
    print("One-hot encoding labels...")
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test  = to_categorical(y_test, NUM_CLASSES)

    return (x_train, y_train), (x_test, y_test)


def preview_first_n_images(images, labels, n=10, class_names=None, out_path=None):
    n = min(n, images.shape[0])
    plt.figure(figsize=(12, 3))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(images[i])
        label_idx = int(labels[i]) if labels.ndim == 1 else int(np.argmax(labels[i]))
        title = class_names[label_idx] if class_names else str(label_idx)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved preview: {out_path}")
        plt.close()
    else:
        plt.show()


# -----------------------
# Step 2: Model Definition
# -----------------------
def build_cifar10_cnn_stepwise(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                               conv_dropout=0.25, dense_dropout=0.5):
    """
    Step-by-step Sequential model with 3 Conv blocks + Dense head:
    - Conv(32) -> BN -> MaxPool(2x2) -> Dropout(0.25)
    - Conv(64) -> BN -> MaxPool(2x2) -> Dropout(0.25)
    - Conv(128)-> BN -> MaxPool(2x2) -> Dropout(0.25)
    - Flatten -> Dense(512, ReLU) -> BN -> Dropout(0.5)
    - Dense(10, Softmax)
    """
    model = Sequential(name="cifar10_cnn_stepwise")

    # Block 1
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_dropout))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_dropout))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_dropout))

    # Dense head
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(dense_dropout))

    # Output layer
    model.add(Dense(num_classes, activation="softmax"))

    # Compile
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cifar10_cnn_compact(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                              conv_dropout=0.25, dense_dropout=0.5):
    """
    Compact Sequential model definition with the same architecture in a single list.
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(conv_dropout),

        # Block 2
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(conv_dropout),

        # Block 3
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(conv_dropout),

        # Dense head
        Flatten(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(dense_dropout),

        # Output
        Dense(num_classes, activation="softmax"),
    ], name="cifar10_cnn_compact")

    # Compile
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# -----------------------
# Step 3: Training & Evaluation
# -----------------------
def compile_model(model):
    # Redundant compile for clarity (safe to call again)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, x_train, y_train, batch_size, epochs, val_split):
    print(f"Training model: epochs={epochs}, batch_size={batch_size}, val_split={val_split}")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        ModelCheckpoint(filepath=os.path.join(OUT_DIR, "best_model.h5"),
                        monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=1
    )
    return history


def plot_history(history, out_path):
    print("Plotting training & validation curves...")
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(12, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Training Accuracy", color="steelblue")
    plt.plot(val_acc, label="Validation Accuracy", color="darkorange")
    plt.title("Accuracy Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Training Loss", color="steelblue")
    plt.plot(val_loss, label="Validation Loss", color="darkorange")
    plt.title("Loss Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved curves: {out_path}")


def evaluate_model(model, x_test, y_test):
    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc * 100:.2f}%  |  Test loss: {test_loss:.4f}")
    return test_loss, test_acc


def plot_confusion_matrix(model, x_test, y_test, out_path):
    if not SKLEARN_AVAILABLE:
        print("scikit-learn not available; skipping confusion matrix.")
        return

    print("Generating confusion matrix...")
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Set)")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASS_NAMES)

    # Normalize rows
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    threshold = cm_norm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            val_norm = cm_norm[i, j]
            color = "white" if val_norm > threshold else "black"
            plt.text(j, i, f"{val}\n({val_norm:.2f})", ha="center", va="center", color=color, fontsize=8)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix: {out_path}")


# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    # Load & preprocess
    (x_train, y_train), (x_test, y_test) = load_and_preprocess()

    # Optional preview
    if args.preview:
        preview_path = os.path.join(OUT_DIR, "preview_first_10.png")
        preview_first_n_images(x_train, y_train, n=10, class_names=CLASS_NAMES, out_path=preview_path)

    # Build model (choose compact or stepwise)
    if args.model == "compact":
        model = build_cifar10_cnn_compact()
    else:
        model = build_cifar10_cnn_stepwise()

    print(model.summary())  # Model architecture overview
    model = compile_model(model)

    # Train
    history = train_model(
        model,
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_split=args.val_split
    )

    # Plot curves
    curves_path = os.path.join(OUT_DIR, "accuracy_loss_curves.png")
    plot_history(history, curves_path)

    # Evaluate
    test_loss, test_acc = evaluate_model(model, x_test, y_test)

    # Save model
    if args.save_model:
        save_path = os.path.join(OUT_DIR, "cifar10_cnn.h5")
        model.save(save_path)
        print(f"Saved trained model to: {save_path}")

    # Confusion matrix
    if args.plot_cm:
        cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
        plot_confusion_matrix(model, x_test, y_test, cm_path)


if __name__ == "__main__":
    main()
