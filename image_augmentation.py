## 🐍 TensorFlow / Keras Example

### 1) Load Dataset

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
import numpy as np


DATA_DIR = "data/train"  # contains 'cats/' and 'dogs/'
IMG_SIZE = (180, 180)
BATCH_SIZE = 32

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    labels="inferred",
    label_mode="int",
    seed=1234
)

IMG_SIZE = (180, 180)
BATCH_SIZE = 32
SEED = 42

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Optional performance boosts
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

## 2) Preview Utility

def show_batch(images, labels=None, class_names=None, cols=6, title="Preview"):
    plt.figure(figsize=(12, 8))
    for i in range(min(len(images), cols * 2)):
        ax = plt.subplot(2, cols, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        if labels is not None and class_names is not None:
            ax.set_title(class_names[int(labels[i])])
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Example usage
for images, labels in train_ds.take(1):
    show_batch(images, labels, class_names, title="Original images")

# 🛠️ Augmentation Pipelines
# Apply only to training data. Do not augment validation or test sets.
# A) Geometric Transformations
data_aug_geo = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),   # or "horizontal_and_vertical"
    tf.keras.layers.RandomRotation(0.15),       # ~±15%
    tf.keras.layers.RandomZoom(0.2),            # ±20% zoom
    tf.keras.layers.RandomTranslation(0.1, 0.1) # ±10% shift
], name="geo_augment")

# B) Color & Intensity Transformations
# Note: RandomBrightness exists in newer TF versions; if missing, use a custom layer below.

data_aug_color = tf.keras.Sequential([
    tf.keras.layers.RandomContrast(0.2),        # ±20% contrast
], name="color_augment")
Custom Brightness Layer (fallback):

class RandomBrightness(tf.keras.layers.Layer):
    def __init__(self, factor=0.2, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, images, training=None):
        if not training:
            return images
        # images expected in uint8 [0,255] or float [0,1]; normalize to [0,1]
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        delta = tf.random.uniform([], -self.factor, self.factor)
        images = tf.clip_by_value(tf.image.adjust_brightness(images, delta), 0.0, 1.0)
        return tf.image.convert_image_dtype(images, dtype=images.dtype)

data_aug_color = tf.keras.Sequential([
    tf.keras.layers.RandomContrast(0.2),
    RandomBrightness(0.2),
], name="color_augment")

# C) Occlusion / Cutout (Random Erasing)

class RandomErasing(tf.keras.layers.Layer):
    def __init__(self, max_erase_area=0.2, **kwargs):
        super().__init__(**kwargs)
        self.max_erase_area = max_erase_area

    def call(self, images, training=None):
        if not training:
            return images

        images = tf.cast(images, tf.float32)
        batch_size = tf.shape(images)[0]
        h = tf.shape(images)[1]
        w = tf.shape(images)[2]
        c = tf.shape(images)[3]

        # Rectangle sizes (up to max_erase_area of image size)
        max_eh = tf.cast(tf.cast(h, tf.float32) * self.max_erase_area, tf.int32)
        max_ew = tf.cast(tf.cast(w, tf.float32) * self.max_erase_area, tf.int32)
        erase_h = tf.maximum(tf.random.uniform([batch_size], 1, tf.maximum(max_eh, 2), dtype=tf.int32), 1)
        erase_w = tf.maximum(tf.random.uniform([batch_size], 1, tf.maximum(max_ew, 2), dtype=tf.int32), 1)

        # Random positions
        y = tf.random.uniform([batch_size], 0, tf.cast(h - erase_h, tf.float32), dtype=tf.float32)
        x = tf.random.uniform([batch_size], 0, tf.cast(w - erase_w, tf.float32), dtype=tf.float32)
        y = tf.cast(y, tf.int32)
        x = tf.cast(x, tf.int32)

        def erase_one(img, y0, x0, eh, ew):
            mask = tf.ones([eh, ew, c], dtype=img.dtype)
            padding = [[y0, h - y0 - eh], [x0, w - x0 - ew], [0, 0]]
            mask_padded = tf.pad(mask, padding, "CONSTANT", constant_values=0.0)
            return img * (1.0 - mask_padded)  # zero out region

        return tf.map_fn(
            lambda args: erase_one(*args),
            (images, y, x, erase_h, erase_w),
            dtype=images.dtype
        )

data_aug_cutout = tf.keras.Sequential([
    RandomErasing(max_erase_area=0.3)
], name="cutout_augment")

D) Composite Augmentation (Catch‑All)


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.2),
    RandomErasing(max_erase_area=0.25),
], name="composite_augment")

# 👀 Visualize Augmented Batches

AUTOTUNE = tf.data.AUTOTUNE
train_ds_aug = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

for images, labels in train_ds_aug.take(1):
    show_batch(images, labels, class_names, title="Augmented images")
  
# 🧠 Build & Train a Simple Model

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = data_augmentation(inputs)     # augment only inside training graph
x = tf.keras.layers.Rescaling(1./255)(x)

# A small CNN
x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(128, 3, activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
# ✅ Best Practices
# Train‑only augmentation: Never augment validation or test sets.
# Keep labels plausible: Avoid transforms that alter semantics (e.g., flipping text direction on traffic signs).
# Moderation over extremes: Start with small factors (±10–20%) and inspect examples.
# Class balance: Augmentation doesn’t fix label imbalance by itself; consider class‑balanced sampling or focal loss.
# Performance: Use prefetch and num_parallel_calls=tf.data.AUTOTUNE.
# 🔧 Extras: Save Augmented Samples (Optional)

out_dir = "aug_samples"
os.makedirs(out_dir, exist_ok=True)

for images, labels in train_ds.take(1):
    aug_imgs = data_augmentation(images, training=True)
    aug_imgs = aug_imgs.numpy().astype("uint8")
    for i, img in enumerate(aug_imgs[:12]):
        Image.fromarray(img).save(os.path.join(out_dir, f"aug_{i}.png"))
