'''
🚀 What this script does

Loads & preprocesses MNIST (28×28)
Defines a tunable Keras model with:

hidden1 units: 32 → 512 (step 32)
dropout1: 0.1 → 0.5 (step 0.1)
hidden2 units: 16 → 128 (step 16)
dropout2: 0.1 → 0.5 (step 0.1)
learning_rate: {1e-4, 1e-3, 1e-2}


Supports Random Search, Bayesian Optimization, or Hyperband
Uses callbacks: EarlyStopping, ReduceLROnPlateau, TensorBoard
Prints best hyperparameters & evaluates best model on test set
Saves:

Best model → tuner_logs/<project>/artifacts/best_model.keras
Best hyperparameters → best_hyperparameters.json

# Run a random search (20 trials, 20 epochs):
pip install tensorflow keras-tuner
python run_hparam_tuning.py --tuner random --max_trials 20 --epochs 20 --batch_size 64 \
  --directory tuner_logs --project_name mnist_demo

# Try Bayesian optimization:
python run_hparam_tuning.py --tuner bayesian --max_trials 30

# Try Hyperband:
python run_hparam_tuning.py --tuner hyperband --max_epochs 30 --factor 3

# Inspect trials with TensorBoard:
tensorboard --logdir tuner_logs/mnist_tuning



'''
# run_hparam_tuning.py
"""
Hyperparameter tuning script for a dense neural network on MNIST using Keras Tuner.

This single file:
  • Loads and preprocesses MNIST (28x28) data
  • Defines a tunable model (hidden sizes, dropout rates, learning rate)
  • Runs hyperparameter search (RandomSearch, BayesianOptimization, or Hyperband)
  • Applies helpful callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
  • Reports best hyperparameters and test performance
  • Saves the best model and best hyperparameters to disk

Usage examples:
  python run_hparam_tuning.py \
      --tuner random --max_trials 20 --epochs 20 --batch_size 64 \
      --directory tuner_logs --project_name mnist_demo

  python run_hparam_tuning.py --tuner bayesian --max_trials 30
  python run_hparam_tuning.py --tuner hyperband --max_epochs 30 --factor 3

Notes:
  • Requires: tensorflow, keras-tuner (a.k.a. keras_tuner)
    Install: pip install tensorflow keras-tuner
  • Labels are one-hot encoded for 'categorical_crossentropy'.
  • The script sets seeds for basic reproducibility.
"""

from __future__ import annotations
import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

try:
    import keras_tuner as kt  # Keras Tuner
except Exception as e:  # pragma: no cover
    raise ImportError(
        "keras-tuner is required. Install with: pip install keras-tuner"
    ) from e


# -----------------------------
# Reproducibility helpers
# -----------------------------

def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------------
# Data loading & preprocessing
# -----------------------------

def load_mnist_splits(validation_size: int = 10_000) -> Tuple[np.ndarray, ...]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize to [0,1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Create validation split from training set
    if validation_size > 0:
        x_val = x_train[-validation_size:]
        y_val = y_train[-validation_size:]
        x_train = x_train[:-validation_size]
        y_train = y_train[:-validation_size]
    else:
        x_val = x_train[:0]
        y_val = y_train[:0]

    # Flatten to vectors of size 784
    x_train = x_train.reshape((-1, 784))
    x_val = x_val.reshape((-1, 784))
    x_test = x_test.reshape((-1, 784))

    # One-hot encode labels for categorical crossentropy (10 classes)
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_val, y_val, x_test, y_test


# -----------------------------
# Tunable model definition
# -----------------------------

def build_model(hp: 'kt.HyperParameters') -> keras.Model:
    """Builds and compiles a tunable dense network for MNIST.

    Hyperparameters exposed to the tuner:
      - hidden1 (Int): 32..512 (step 32)
      - dropout1 (Float): 0.1..0.5 (step 0.1)
      - hidden2 (Int): 16..128 (step 16)
      - dropout2 (Float): 0.1..0.5 (step 0.1)
      - learning_rate (Choice): {1e-4, 1e-3, 1e-2}
    """
    model = keras.Sequential(name="tunable_mlp")

    model.add(layers.Input(shape=(784,), name="input_784"))

    # Hidden layer 1
    units_h1 = hp.Int("hidden1", min_value=32, max_value=512, step=32)
    model.add(Dense(units_h1, activation="relu", name="dense_hidden1"))

    rate_d1 = hp.Float("dropout1", min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(rate_d1, name="dropout1"))

    # Hidden layer 2
    units_h2 = hp.Int("hidden2", min_value=16, max_value=128, step=16)
    model.add(Dense(units_h2, activation="relu", name="dense_hidden2"))

    rate_d2 = hp.Float("dropout2", min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(rate_d2, name="dropout2"))

    # Output layer (fixed)
    model.add(Dense(10, activation="softmax", name="output"))

    # Learning rate choice
    lr = hp.Choice("learning_rate", values=[1e-4, 1e-3, 1e-2])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -----------------------------
# Tuning runners
# -----------------------------

@dataclass
class TuningConfig:
    tuner: str = "random"  # random | bayesian | hyperband
    max_trials: int = 20
    executions_per_trial: int = 1
    max_epochs: int = 20  # used by search, and by Hyperband's max_epochs
    factor: int = 3       # Hyperband reduction factor
    directory: str = "tuner_logs"
    project_name: str = "mnist_tuning"
    batch_size: int = 64
    seed: int = 42
    validation_size: int = 10_000


def make_tuner(cfg: TuningConfig) -> 'kt.engine.tuner.Tuner':
    if cfg.tuner.lower() == "random":
        tuner = kt.RandomSearch(
            hypermodel=build_model,
            objective="val_accuracy",
            max_trials=cfg.max_trials,
            executions_per_trial=cfg.executions_per_trial,
            directory=cfg.directory,
            project_name=cfg.project_name,
            seed=cfg.seed,
        )
    elif cfg.tuner.lower() == "bayesian":
        tuner = kt.BayesianOptimization(
            hypermodel=build_model,
            objective="val_accuracy",
            max_trials=cfg.max_trials,
            executions_per_trial=cfg.executions_per_trial,
            directory=cfg.directory,
            project_name=cfg.project_name,
            seed=cfg.seed,
        )
    elif cfg.tuner.lower() == "hyperband":
        tuner = kt.Hyperband(
            hypermodel=build_model,
            objective="val_accuracy",
            max_epochs=cfg.max_epochs,
            factor=cfg.factor,
            directory=cfg.directory,
            project_name=cfg.project_name,
            seed=cfg.seed,
        )
    else:
        raise ValueError("tuner must be one of: random | bayesian | hyperband")
    return tuner


# -----------------------------
# Main execution
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning on MNIST with Keras Tuner")
    parser.add_argument("--tuner", type=str, default="random", choices=["random", "bayesian", "hyperband"], help="Tuner algorithm")
    parser.add_argument("--max_trials", type=int, default=20, help="Max trials for Random/Bayesian tuners")
    parser.add_argument("--executions_per_trial", type=int, default=1, help="Repeat each trial N times and average")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per trial")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--directory", type=str, default="tuner_logs", help="Directory to save tuner logs")
    parser.add_argument("--project_name", type=str, default="mnist_tuning", help="Project name under directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validation_size", type=int, default=10_000, help="Validation set size carved from training")
    # Hyperband-specific
    parser.add_argument("--factor", type=int, default=3, help="Hyperband reduction factor")
    parser.add_argument("--max_epochs", type=int, default=30, help="Hyperband max epochs (also cap for training)")

    args = parser.parse_args()

    # Seed everything for reproducibility
    set_global_seed(args.seed)

    # Prepare data
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_splits(args.validation_size)

    # Build tuner
    cfg = TuningConfig(
        tuner=args.tuner,
        max_trials=args.max_trials,
        executions_per_trial=args.executions_per_trial,
        max_epochs=args.max_epochs,
        factor=args.factor,
        directory=args.directory,
        project_name=args.project_name,
        batch_size=args.batch_size,
        seed=args.seed,
        validation_size=args.validation_size,
    )
    tuner = make_tuner(cfg)

    # Callbacks: early stopping, LR reduction, tensorboard
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-5),
        keras.callbacks.TensorBoard(log_dir=os.path.join(cfg.directory, cfg.project_name, "tb_logs")),
    ]

    print("\n[INFO] Starting hyperparameter search...")
    tuner.search(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs if cfg.tuner != "hyperband" else cfg.max_epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n[INFO] Hyperparameter search complete. Selecting best model...")
    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]

    # Evaluate on test set
    test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"[RESULT] Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    # Save best model and hyperparameters
    save_dir = os.path.join(cfg.directory, cfg.project_name, "artifacts")
    os.makedirs(save_dir, exist_ok=True)

    best_model_path = os.path.join(save_dir, "best_model.keras")
    best_model.save(best_model_path)

    hp_json_path = os.path.join(save_dir, "best_hyperparameters.json")
    with open(hp_json_path, "w", encoding="utf-8") as f:
        json.dump(best_hp.values, f, indent=2)

    print("\n[SAVED]")
    print(f"  • Best model: {best_model_path}")
    print(f"  • Best hyperparameters: {hp_json_path}")
    print("\n[BEST HP]")
    for k, v in best_hp.values.items():
        print(f"  {k}: {v}")

    print("\n[HINT]")
    print("  • Launch TensorBoard to inspect trials:")
    print(f"    tensorboard --logdir {os.path.join(cfg.directory, cfg.project_name)}")


if __name__ == "__main__":
    main()

