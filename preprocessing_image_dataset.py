 # 1) Load a dataset from folders

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = "data/train"  # contains 'cats/' and 'dogs/'

# Deterministic shuffling/reproducibility
SEED = 1234
BATCH_SIZE = 32
IMG_SIZE = (224, 224)  # VGG/ResNet-friendly

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",           # folder names -> labels
    label_mode="int",            # 0/1 integer labels
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,         # resize (Default: also center crop if aspect ratio differs when using tf.data+layers)
    shuffle=True,
    seed=SEED
)

# Inspect class names
print("Class names:", train_ds.class_names)


# 2) Optional: Center crop and additional resizing


# If you want an explicit center crop step followed by a final resize
preprocess_shape = (224, 224)

keras_pipeline = keras.Sequential([
    layers.CenterCrop(height=preprocess_shape[0], width=preprocess_shape[1]),
    layers.Rescaling(1./255.0)  # normalize to [0, 1]
])
# ✅ Note: For 8-bit images, pixel values are 0–255. Scaling by 1/255.0 maps values to [0, 1] (common for CNNs).

# 3) Apply preprocessing with map


def apply_preprocessing(images, labels):
    images = keras_pipeline(images)
    return images, labels

train_ds_pp = train_ds.map(apply_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
train_ds_pp = train_ds_pp.prefetch(buffer_size=tf.data.AUTOTUNE)

# 4) Visualize a batch


import matplotlib.pyplot as plt

def show_batch(ds, n=9):
    plt.figure(figsize=(8, 8))
    for images, labels in ds.take(1):
        for i in range(min(n, images.shape[0])):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(f"label={labels[i].numpy()}")
            plt.axis("off")
    plt.tight_layout()
    plt.show()

show_batch(train_ds_pp)

# 5) Verify normalization

import numpy as np

mins, maxs = [], []
for images, _ in train_ds_pp.take(5):  # sample a few batches
    mins.append(tf.reduce_min(images).numpy())
    maxs.append(tf.reduce_max(images).numpy())

print("Pixel min across sample batches:", float(np.min(mins)))
print("Pixel max across sample batches:", float(np.max(maxs)))
# Expect ~0.0 and ~1.0

# 🔥 PyTorch Pipeline
# 1) Define transforms (resize → center-crop → to tensor → normalize)


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

SEED = 1234
torch.manual_seed(SEED)

IMG_SIZE = 224
BATCH_SIZE = 32

# Mean/Std below are common for ImageNet-trained models; adjust if needed.
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),                 # resize shortest side to 256
    transforms.CenterCrop(IMG_SIZE),       # center crop to 224x224
    transforms.ToTensor(),                 # [0,1], CHW
    transforms.Normalize(imagenet_mean, imagenet_std)  # normalize
])

DATA_DIR = "data/train"

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

print("Classes:", dataset.classes)

# 2) Visualize a batch (denormalize for display)


import matplotlib.pyplot as plt
import numpy as np

def denormalize(img_tensor, mean, std):
    # img_tensor: CHW
    img = img_tensor.clone().cpu()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return img

def show_batch_pytorch(loader, n=9):
    images, labels = next(iter(loader))
    plt.figure(figsize=(8, 8))
    for i in range(min(n, images.size(0))):
        ax = plt.subplot(3, 3, i + 1)
        img = denormalize(images[i], imagenet_mean, imagenet_std)
        img = np.clip(img.permute(1, 2, 0).numpy(), 0, 1)
        plt.imshow(img)
        plt.title(f"label={labels[i].item()}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

show_batch_pytorch(loader)

# 3) Verify ranges (pre- and post-normalization)


# In PyTorch, after ToTensor() values are in [0,1], then Normalize() shifts/scales them.
# To inspect the raw tensor stats (post-normalization):
batch, _ = next(iter(loader))
print("Post-normalization: min =", float(batch.min()), "max =", float(batch.max()))
# Expect values roughly in a range centered around 0 (not limited to [0,1] anymore).

# 📎 Minimal End-to-End (Keras)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

train_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=(224, 224),
    batch_size=32,
    labels="inferred",
    label_mode="int",
    seed=1234,
    shuffle=True
)

model_input = keras.Input(shape=(224, 224, 3))
x = layers.Rescaling(1./255)(model_input)
x = layers.Conv2D(16, 3, activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
output = layers.Dense(2, activation="softmax")(x)

model = keras.Model(model_input, output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_ds, epochs=1)
# 📎 Minimal End-to-End (PyTorch)


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

dataset = datasets.ImageFolder("data/train", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(16, 2)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for images, labels in loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    print("batch loss:", float(loss.item()))
    break  # remove for full training



