# 🧠 CNN on CIFAR‑10 with Keras/TensorFlow — Preprocess → Define → Train

This file, following with python code, walks through a minimal, reproducible pipeline to build a **Convolutional Neural Network (CNN)** on the **CIFAR‑10** dataset using **Keras/TensorFlow**:

### Steps:
1. **Preparing to build a CNN in Python** (data loading, preview, normalization, one‑hot encoding)  
2. **Defining a CNN in Python** (Sequential model with Conv/BatchNorm/MaxPool/Dropout blocks → Dense head)  
3. **Training a CNN in Python** (compile with Adam + categorical cross‑entropy, fit for 20 epochs, visualize curves, evaluate on test set)

---

## 📚 Dataset: CIFAR‑10

- **10 classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck  
- **Images:** 50,000 train / 10,000 test  
- **Resolution:** 32×32 RGB  
- Loaded via `tf.keras.datasets.cifar10` (channels‑last input: `(32, 32, 3)`)

---

## 🗂️ Project Structure
.
├─ README.md
├─ requirements.txt
└─ cnn_cifar10_full.py       # Single script: preprocess + model + training + evaluation


---

## ✅ Requirements

```txt
tensorflow>=2.9.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0   # optional; only needed for confusion matrix

## Install

``
pip install -r requirements.txt
``

OR

``
pip install tensorflow numpy matplotlib scikit-learn
``

## 🚀 Quick Start (End‑to‑End)
- Run the full pipeline (preprocessing → define → train → evaluate):

python cnn_cifar10_full.py --model compact --epochs 20 --batch-size 64 --val-split 0.1 --save-model --plot-cm
``

### Outputs:
- Console: model summary, training progress, final test accuracy and loss
- Images: outputs/accuracy_loss_curves.png (+ optional outputs/confusion_matrix.png)
- Model: outputs/cifar10_cnn.h5 (if --save-model is used)

## 🧰 Preparing to Build a CNN in Python (Preprocessing)

We:
- Load CIFAR‑10: x_train, y_train, x_test, y_test
- Preview a few samples to sanity‑check inputs
- Normalize pixels by dividing by 255 → values in [0, 1] to help model convergence
- One‑hot encode labels (0..9 → 10‑dim vectors) for multi‑class softmax

## 🏗️ Defining a CNN in Python (Model Architecture)
- We use a Keras Sequential model with channels‑last input (32, 32, 3):

### Conv Block ×3 (filters 32 → 64 → 128):
- Conv2D(3×3, ReLU, padding="same")
- BatchNormalization() (faster training, stable gradients)
- MaxPooling2D(2×2)
- Dropout(0.25) (reduces overfitting)

### Dense Head:

- Flatten()
- Dense(512, ReLU)
- BatchNormalization()
- Dropout(0.5)

### Output:
- Dense(10, Softmax) (multi‑class probabilities)
- I provide two implementations: step‑by‑step (.add) and compact (single layer list). Choose with --model stepwise or --model compact.

## 🏋️ Training a CNN in Python

- Compile: optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
- Fit: epochs=20, batch_size=64, validation_split=0.1
- Visualize: plot training vs validation accuracy/loss across epochs
- Evaluate: report final test accuracy & loss; optionally create a confusion matrix

**Healthy training behavior:**

- Validation curves track training curves closely → limited overfitting
- Expect ~70–75% test accuracy for this baseline (varies by run) without augmentation

## 🛠️ Troubleshooting
- Matplotlib on headless servers:

```
  import matplotlib
  matplotlib.use("Agg")  # before importing pyplot
```
- GPU/CPU control:
```
CUDA_VISIBLE_DEVICES="" python cnn_cifar10_full.py
```
- Shape errors: Ensure input shape is (32, 32, 3) and labels are one‑hot encoded with 10 classes.
- Overfitting: Increase dropout, add data augmentation (tf.keras.layers.RandomFlip/RandomRotation), or reduce epochs.

 Following code shows all these in detail. Refer cnn_cifar10_full.py

