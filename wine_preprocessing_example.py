
# -*- coding: utf-8 -*-
"""
File: wine_preprocessing_example.py
Date: 2025-12-20

Description
-----------
It demonstrates:
  1) Loading data from a local CSV (wine.csv). If the file does not exist,
     it is created from scikit-learn's built-in dataset for reproducibility.
  2) Inspecting the data with .head().
  3) (Optional) Label encoding of the target column (Class). The built-in
     Wine dataset already uses numeric labels 0/1/2.
  4) Converting to NumPy arrays.
  5) Separating features (X) and target (y).
  6) Standardizing numeric features using StandardScaler.
  7) One-hot encoding the target using tf.keras.utils.to_categorical.
  8) Train/test split with stratification.
  9) (Optional) A tiny Keras classifier to verify the pipeline end-to-end.

Usage
-----
$ python wine_preprocessing_example.py

Output
------
Prints dataset preview, shapes, and (optionally) a test accuracy from a simple model.

Notes
-----
- The created CSV will have 178 rows × 14 columns (13 features + 1 target 'Class').
- Feature names follow scikit-learn's wine feature names:
  ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
   'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
   'proanthocyanins', 'color_intensity', 'hue',
   'od280/od315_of_diluted_wines', 'proline']
- The target column is 'Class' with values {0, 1, 2}.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

CSV_PATH = "wine.csv"  # Output/input CSV file name


def ensure_wine_csv(csv_path: str = CSV_PATH) -> None:
    """Create wine.csv if it doesn't exist using scikit-learn's built-in dataset.

    This makes the script self-contained and easy to run on any machine.
    """
    if os.path.exists(csv_path):
        return

    from sklearn.datasets import load_wine

    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["Class"] = wine.target  # numeric labels 0/1/2
    df.to_csv(csv_path, index=False)


def main():
    # 1) Ensure CSV exists (creates it from sklearn data if missing)
    ensure_wine_csv(CSV_PATH)

    # 2) Load data and review content
    wine_data = pd.read_csv(CSV_PATH)

    print("\nLoaded Data :\n------------------------------------")
    print(wine_data.head())

    # 3) Label-encode the target column 'Class' (robust step)
    #    If 'Class' is already numeric, LabelEncoder will keep it as-is
    #    (aside from potentially remapping values to 0..K-1 if necessary).
    label_encoder = preprocessing.LabelEncoder()
    wine_data['Class'] = label_encoder.fit_transform(wine_data['Class'])

    # 4) Convert to NumPy array
    np_wine = wine_data.to_numpy()

    # 5) Separate features and target
    #    - First 13 columns are numeric features
    #    - Last column (index 13) is the target 'Class'
    X_data = np_wine[:, 0:13]
    y_data = np_wine[:, 13]

    print("\nFeatures before scaling :\n------------------------------------")
    print(X_data[:5, :])
    print("\nTarget before encoding :\n------------------------------------")
    print(y_data[:5])

    # 6) Standardize numeric features
    scaler = StandardScaler().fit(X_data)
    X_scaled = scaler.transform(X_data)

    # 7) One-hot encode target (3 classes in the UCI Wine dataset)
    y_onehot = tf.keras.utils.to_categorical(y_data, num_classes=3)

    print("\nFeatures after scaling :\n------------------------------------")
    print(X_scaled[:5, :])
    print("\nTarget after one-hot-encoding :\n------------------------------------")
    print(y_onehot[:5, :])

    # 8) Train/test split (use stratify to preserve class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_onehot, test_size=0.10, random_state=42, stratify=y_onehot
    )

    print("\nTrain Test Dimensions:\n------------------------------------")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # 9) Optional: A small Keras classifier to verify the pipeline
    #    You can comment this block out if you only need the preprocessing.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        epochs=30,           # reduced epochs for a quick run
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
