# anime_real_vs_good_fake.py
# -*- coding: utf-8 -*-

"""
This script reproduces your DCGAN-style discriminator classification pipeline
for two classes:
  - real_images/
  - good_fake_images/   (bad_fake_images/ is removed)

It adds:
- Drive mounting check and folder walkthrough per your TODO recording
- Opening selected images from good_fake_images (im_55* and im_70* sorted desc)
- Robust visualization and logging
- Reproducible train/test split
- Misclassification visualization (scroll through mispredictions in the notebook)

It preserves your original architecture, transforms, and training recipe.
"""

import os
import glob
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------
# Global Settings
# ---------------------------

# Colab Drive dataset path
DATA_ROOT = "/content/drive/MyDrive/ai_workshop_dcgans/anime_classification/"

# Hyperparameters (preserving your choices)
BATCH_SIZE = 8
IMAGE_SIZE = 64
NC = 3            # number of channels
NDF = 64          # discriminator feature maps
LR = 0.0002
EPOCHS = 2
BETAS = (0.5, 0.999)
NUM_WORKERS = 2

# Reproducibility
SEED = 42


# ---------------------------
# Utilities (env, display)
# ---------------------------

def set_reproducible(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_env_info():
    device = get_device()
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")


def safe_imshow(img: np.ndarray, title: Optional[str] = None):
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def denormalize(img_tensor: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    if img_tensor.dim() == 4:
        mean_t = torch.tensor(mean, device=img_tensor.device).view(1, -1, 1, 1)
        std_t = torch.tensor(std, device=img_tensor.device).view(1, -1, 1, 1)
    else:
        mean_t = torch.tensor(mean, device=img_tensor.device).view(-1, 1, 1)
        std_t = torch.tensor(std, device=img_tensor.device).view(-1, 1, 1)
    return img_tensor * std_t + mean_t


def show_grid(tensor_batch: torch.Tensor, nrow: int = 8, title: Optional[str] = None):
    with torch.no_grad():
        grid = make_grid(tensor_batch.cpu(), nrow=nrow, padding=2, normalize=False)
        grid = denormalize(grid).clamp(0, 1)
        npimg = grid.permute(1, 2, 0).cpu().numpy()
    safe_imshow(npimg, title=title)


# ---------------------------
# Colab / Drive Steps (Recording TODOs)
# ---------------------------

def mount_colab_drive_if_needed():
    """
    In Colab, mount first:
        from google.colab import drive
        drive.mount('/content/drive')
    This function verifies accessibility afterward.
    """
    if os.path.exists("/content/drive/MyDrive"):
        print("✅ MyDrive is mounted and accessible.")
    else:
        print("⚠️ MyDrive not found. In Colab, run:")
        print("from google.colab import drive")
        print("drive.mount('/content/drive')")


def listdir_safe(path: str) -> List[str]:
    try:
        return sorted(os.listdir(path))
    except Exception as e:
        print(f"Could not list {path}: {e}")
        return []


def show_folder_walkthrough(base_path: str):
    """
    RECORDING TODOs:

    - Go to Drive
    - Show the folder anime_classification/
    - Click through and show the two subfolders (real_images, good_fake_images)
    - Click into good_fake_images:
        - open 2-3 images from im_55*
        - sort descending by name and open 2-3 images from im_70*
    """
    print("\n[Drive Walkthrough]")

    parent = os.path.dirname(base_path.rstrip("/"))
    parent_listing = listdir_safe(parent)
    print(f"\n📁 In: {parent}\nContains (sample): {parent_listing[:10]}")

    # anime_classification/
    print(f"\n📁 In: {base_path}")
    base_listing = listdir_safe(base_path)
    print(f"Contains: {base_listing}")

    # Expected subfolders
    classes_found = [d for d in base_listing if os.path.isdir(os.path.join(base_path, d))]
    print(f"\nExpected subfolders (classes): {classes_found}  (should be: real_images, good_fake_images)")

    # Go into good_fake_images and show specific series
    good_dir = os.path.join(base_path, "good_fake_images")
    if os.path.isdir(good_dir):
        # im_55* series (2-3 images)
        im55 = sorted(glob.glob(os.path.join(good_dir, "im_55*")))
        print(f"\n🖼 good_fake_images im_55* count: {len(im55)} (showing up to 3)")
        for p in im55[:3]:
            try:
                im = Image.open(p).convert("RGB")
                safe_imshow(np.array(im)/255.0, title=os.path.basename(p))
            except Exception as e:
                print(f"Failed to open {p}: {e}")

        # Sorted descending by name, then show im_70* (2-3 images)
        all_good_desc = sorted(glob.glob(os.path.join(good_dir, "*.*")), reverse=True)
        im70_desc = [p for p in all_good_desc if os.path.basename(p).startswith("im_70")]
        print(f"\n🖼 good_fake_images im_70* (sorted desc) count: {len(im70_desc)} (showing up to 3)")
        for p in im70_desc[:3]:
            try:
                im = Image.open(p).convert("RGB")
                safe_imshow(np.array(im)/255.0, title=os.path.basename(p))
            except Exception as e:
                print(f"Failed to open {p}: {e}")
    else:
        print("\n⚠️ good_fake_images/ folder not found.")


# ---------------------------
# Dataset & Dataloaders
# ---------------------------

def build_transforms(image_size: int = IMAGE_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def load_dataset(root: str, transform: T.Compose) -> dset.ImageFolder:
    dataset = dset.ImageFolder(root=root, transform=transform)
    return dataset


def split_dataset(
    dataset: data.Dataset,
    train_ratio: float = 0.80,
    seed: int = SEED
) -> Tuple[data.Subset, data.Subset]:
    total_len = len(dataset)
    train_len = int(round(total_len * train_ratio))
    test_len = total_len - train_len
    generator = torch.Generator().manual_seed(seed)
    return data.random_split(dataset, [train_len, test_len], generator=generator)


def make_dataloaders(
    train_set: data.Dataset,
    test_set: data.Dataset,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    test_full_batch: bool = False
) -> Tuple[data.DataLoader, data.DataLoader]:
    trainloader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    if test_full_batch and len(test_set) > 0:
        test_bs = len(test_set)  # whole test set in a single batch
    else:
        test_bs = batch_size
    testloader = data.DataLoader(
        test_set, batch_size=test_bs, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    return trainloader, testloader


# ---------------------------
# Model: Discriminator (as before)
# ---------------------------

class Discriminator(nn.Module):
    """
    DCGAN-style discriminator architecture with final Sigmoid (kept to match your code).
    """
    def __init__(self, nc: int = NC, ndf: int = NDF):
        super().__init__()
        self.net = nn.Sequential(
            # (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x 4 x 4 -> (1,1,1)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)  # (N,1,1,1)


def weights_init(m):
    """
    DCGAN-style init:
    - Conv/ConvTranspose: N(0, 0.02)
    - BatchNorm: weight ~ N(1,0.02), bias = 0
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ---------------------------
# Training / Evaluation
# ---------------------------

def train_one_epoch(
    net: nn.Module,
    trainloader: data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_every: int = 20,
    epoch_index: int = 0
):
    net.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device).float()  # BCE targets must be float

        optimizer.zero_grad()
        outputs = net(inputs)              # (N,1,1,1)
        loss = criterion(outputs.flatten(), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i % log_every) == 0 and i > 0:
            print(f"[Epoch {epoch_index+1} | Step {i:4d}] loss: {running_loss/log_every:.4f}")
            running_loss = 0.0


@torch.no_grad()
def evaluate(
    net: nn.Module,
    loader: data.DataLoader,
    device: torch.device,
    classes: List[str]
) -> Dict[str, float]:
    net.eval()
    correct = 0
    total = 0
    class_correct = {cls: 0 for cls in classes}
    class_total = {cls: 0 for cls in classes}

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        preds = torch.round(outputs.flatten()).long()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for lbl, pred in zip(labels, preds):
            cls_name = classes[int(lbl)]
            class_total[cls_name] += 1
            if pred == lbl:
                class_correct[cls_name] += 1

    overall_acc = 100.0 * correct / max(1, total)
    per_class_acc = {
        cls: 100.0 * class_correct[cls] / max(1, class_total[cls])
        for cls in classes
    }

    print(f"\nAccuracy on test images: {overall_acc:.1f}%")
    for cls, acc in per_class_acc.items():
        print(f"Accuracy for class {cls:>16s}: {acc:5.1f}%")

    metrics = {"overall_acc": overall_acc}
    metrics.update({f"acc_{cls}": acc for cls, acc in per_class_acc.items()})
    return metrics


# ---------------------------
# Misclassification Visualization
# ---------------------------

def show_single(image: torch.Tensor, title: str):
    plt.figure()
    # image is normalized; denorm for display
    img = denormalize(image.cpu()).permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title('\n\n{}'.format(title), fontdict={'size': 14})
    plt.show()


def collect_predictions(
    net: nn.Module,
    loader: data.DataLoader,
    device: torch.device
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Returns (all_labels_batches, all_predictions_batches)
    to keep parity with your original approach.
    """
    net.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            predictions = torch.round(outputs.flatten()).long()
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
    return all_labels, all_predictions


def show_misclassifications(
    net: nn.Module,
    loader: data.DataLoader,
    classes: List[str],
    device: torch.device
):
    """
    RECORDING TODO:
    - Scroll and show fake images categorized as real
    - and real images categorized as fake
    """
    all_labels, all_predictions = collect_predictions(net, loader, device)

    # Iterate again over the loader to have access to original images in the same batch order
    for i, (images, labels) in enumerate(loader):
        for j in range(len(all_predictions[i])):
            predicted_class = int(all_predictions[i][j])
            actual_class = int(all_labels[i][j])

            if predicted_class != actual_class:
                title = f"Model prediction {classes[predicted_class]} (class {predicted_class}), " \
                        f"actual category {classes[actual_class]} (class {actual_class})"
                show_single(images[j], title)


# ---------------------------
# Main Flow (Notebook-friendly)
# ---------------------------

def main():
    # === Runtime info (Recording: "Show the runtime is GPU") ===
    print_env_info()

    # === Colab Drive mount check ===
    # In Colab, run this in a cell first:
    # from google.colab import drive
    # drive.mount('/content/drive')
    mount_colab_drive_if_needed()

    # === Folder walkthrough per your updated TODO ===
    show_folder_walkthrough(DATA_ROOT)

    # === Dataset ===
    set_reproducible(SEED)
    transform = build_transforms(IMAGE_SIZE)
    anime_faces_dataset = load_dataset(DATA_ROOT, transform)
    print(f"\nDataset size: {len(anime_faces_dataset)}")

    classes = anime_faces_dataset.classes
    class_to_idx = anime_faces_dataset.class_to_idx
    print(f"Classes: {classes}")
    print(f"class_to_idx mapping: {class_to_idx}")

    # Quick per-class counts validation
    counts_by_class = {cls: 0 for cls in classes}
    for _, y in anime_faces_dataset.imgs:
        for cls, idx in class_to_idx.items():
            if y == idx:
                counts_by_class[cls] += 1
    print(f"Image counts by class: {counts_by_class}")

    # === Split (~80/20) ===
    train_set, test_set = split_dataset(anime_faces_dataset, train_ratio=0.80, seed=SEED)
    print(f"Train: {len(train_set)} | Test: {len(test_set)}")

    # === Dataloaders ===
    # If you want "whole test set is the batch size", set test_full_batch=True
    trainloader, testloader = make_dataloaders(train_set, test_set, BATCH_SIZE, NUM_WORKERS, test_full_batch=False)

    # === Check one training batch ===
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print(f"One train batch shape: {images.shape}")
    first8 = min(8, images.size(0))
    print("Labels (first batch):", " ".join(f"{classes[labels[j]]:5s}" for j in range(first8)))
    show_grid(images[:min(16, images.size(0))], nrow=8, title="Training batch (denormalized)")

    # === Model, Loss, Optimizer ===
    device = get_device()
    netD = Discriminator(nc=NC, ndf=NDF).to(device)
    netD.apply(weights_init)
    print("\nDiscriminator architecture:\n", netD)

    criterion = nn.BCELoss()  # Kept to match your Sigmoid output
    optimizer = optim.Adam(netD.parameters(), lr=LR, betas=BETAS)

    # === Train ===
    print("\n[Training]")
    for epoch in range(EPOCHS):
        train_one_epoch(netD, trainloader, optimizer, criterion, device, log_every=20, epoch_index=epoch)
    print("Finished Training.")

    # === Visual sanity check on a test batch ===
    dataiter = iter(testloader)
    test_images, test_labels = next(dataiter)
    print("\nGroundTruth:")
    gtn = min(8, test_images.size(0))
    print(" ".join(f"{classes[test_labels[j]]:5s}" for j in range(gtn)))
    show_grid(test_images[:min(16, test_images.size(0))], nrow=8, title="Test batch (denormalized)")

    with torch.no_grad():
        outputs = netD(test_images.to(device))
        predicted = torch.round(outputs.flatten()).long().cpu()
    print("\nPredicted:")
    print(" ".join(f"{classes[predicted[j]]:5s}" for j in range(gtn)))

    # === Evaluate full test set ===
    _ = evaluate(netD, testloader, device, classes)

    # === Misclassifications (Recording: scroll to show both types) ===
    show_misclassifications(netD, testloader, classes, device)


if __name__ == "__main__":
    main()
