# anime_real_vs_bad_fake.py
# -*- coding: utf-8 -*-

"""
This script reproduces and refactors your DCGAN-style discriminator classification
pipeline while adding:
- Colab Drive mounting checks
- Folder walkthrough for recording
- Image previews by pattern (im_5*, im_20*), sorted views
- Reproducible splitting
- Clean utilities for display and logging
- Optional, separate baseline (ResNet18) for comparison

It keeps your original architecture, preprocessing, and training logic intact.
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
# Global Settings / Defaults
# ---------------------------

# Colab Drive path (as in your original code)
DATA_ROOT = "/content/drive/MyDrive/ai_workshop_dcgans/anime_classification/"

# Hyperparameters (same as yours)
BATCH_SIZE = 16
IMAGE_SIZE = 64
NC = 3          # number of channels
NDF = 64        # feature map size of discriminator
LR = 0.0002     # learning rate
EPOCHS = 2
BETAS = (0.5, 0.999)

# For DataLoader workers (Colab usually ok with 2~4)
NUM_WORKERS = 2

# Random seeds for reproducibility
SEED = 42


# ---------------------------
# Environment / Utilities
# ---------------------------

def set_reproducible(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # For full determinism (slower):
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
    """Show an RGB image, clipping range to [0,1]."""
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def denormalize(img_tensor: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    Reverse the normalization: x*std + mean.
    Accepts a BCHW or CHW tensor.
    """
    if img_tensor.dim() == 4:
        # batch
        mean_t = torch.tensor(mean, device=img_tensor.device).view(1, -1, 1, 1)
        std_t = torch.tensor(std, device=img_tensor.device).view(1, -1, 1, 1)
    else:
        mean_t = torch.tensor(mean, device=img_tensor.device).view(-1, 1, 1)
        std_t = torch.tensor(std, device=img_tensor.device).view(-1, 1, 1)
    return img_tensor * std_t + mean_t


def show_grid(tensor_batch: torch.Tensor, nrow: int = 8, title: Optional[str] = None):
    """
    Display a grid from a batch (B, C, H, W) that has been normalized to mean=std=0.5.
    """
    with torch.no_grad():
        grid = make_grid(tensor_batch.cpu(), nrow=nrow, padding=2, normalize=False)
        grid = denormalize(grid).clamp(0, 1)
        npimg = grid.permute(1, 2, 0).cpu().numpy()
    safe_imshow(npimg, title=title)


# ---------------------------
# Colab / Drive Setup & Recording TODOs
# ---------------------------

def mount_colab_drive_if_needed():
    """
    In Colab, run:
        from google.colab import drive
        drive.mount('/content/drive')
    This function only verifies afterward.
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
    - Show folder ai_workshop_dcgans/
    - Click anime_classification/
    - Show the two subfolders
    - real_images: show 3-4 images
    - bad_fake_images: show 2-3 from im_5*; then sort desc and show im_20*
    """
    print("\n[Drive Walkthrough]")
    # 1) Show that 'ai_workshop_dcgans/' exists
    parent = os.path.dirname(base_path.rstrip("/"))
    parent_listing = listdir_safe(parent)
    print(f"\n📁 In: {parent}\nContains: {parent_listing}")

    # 2) Show that 'anime_classification/' exists
    print(f"\n📁 In: {base_path}")
    base_listing = listdir_safe(base_path)
    print(f"Contains: {base_listing}")

    # We expect two subfolders (class names)
    classes_found = [d for d in base_listing if os.path.isdir(os.path.join(base_path, d))]
    print(f"\nExpected subfolders (classes): {classes_found}")

    # 3) Browse into each class
    for cls in classes_found:
        cls_path = os.path.join(base_path, cls)
        files = listdir_safe(cls_path)
        print(f"\n📁 {cls_path} (num files: {len(files)}) sample: {files[:5]}")

    # 4) real_images: open 3-4 images
    real_dir = os.path.join(base_path, "real_images")
    if os.path.isdir(real_dir):
        real_imgs = sorted(glob.glob(os.path.join(real_dir, "*.*")))[:4]
        print(f"\n🖼 Showing {len(real_imgs)} sample real images")
        for p in real_imgs:
            try:
                im = Image.open(p).convert("RGB")
                safe_imshow(np.array(im)/255.0, title=os.path.basename(p))
            except Exception as e:
                print(f"Failed to open {p}: {e}")
    else:
        print("\n⚠️ real_images/ folder not found.")

    # 5) bad_fake_images: show im_5* (2-3 images)
    bad_dir = os.path.join(base_path, "bad_fake_images")
    if os.path.isdir(bad_dir):
        im5 = sorted(glob.glob(os.path.join(bad_dir, "im_5*")))
        print(f"\n🖼 bad_fake_images im_5* count: {len(im5)}")
        for p in im5[:3]:
            try:
                im = Image.open(p).convert("RGB")
                safe_imshow(np.array(im)/255.0, title=os.path.basename(p))
            except Exception as e:
                print(f"Failed to open {p}: {e}")

        # 6) Sort descending by name, show im_20* (2-3 images)
        all_bad = sorted(glob.glob(os.path.join(bad_dir, "*.*")), reverse=True)
        im20 = [p for p in all_bad if os.path.basename(p).startswith("im_20")]
        print(f"\n🖼 bad_fake_images im_20* (sorted desc) count: {len(im20)}")
        for p in im20[:3]:
            try:
                im = Image.open(p).convert("RGB")
                safe_imshow(np.array(im)/255.0, title=os.path.basename(p))
            except Exception as e:
                print(f"Failed to open {p}: {e}")
    else:
        print("\n⚠️ bad_fake_images/ folder not found.")


# ---------------------------
# Dataset & Dataloaders
# ---------------------------

def build_transforms(image_size: int = IMAGE_SIZE) -> T.Compose:
    """
    Matches your preprocessing:
    - Resize(image_size)
    - CenterCrop(image_size)
    - ToTensor()
    - Normalize(mean=0.5, std=0.5) per channel
    """
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
    """
    Split into train/test with approx 80/20 (like yours: 1640/408 for 2048 total).
    """
    total_len = len(dataset)
    train_len = int(round(total_len * train_ratio))
    test_len = total_len - train_len
    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = data.random_split(dataset, [train_len, test_len], generator=generator)
    return train_set, test_set


def make_dataloaders(
    train_set: data.Dataset,
    test_set: data.Dataset,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS
) -> Tuple[data.DataLoader, data.DataLoader]:
    trainloader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    testloader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    return trainloader, testloader


# ---------------------------
# Model: Discriminator (as in your code)
# ---------------------------

class Discriminator(nn.Module):
    """
    Mirrors your architecture exactly (ending in Sigmoid).
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
            # (ndf*8) x 4 x 4 -> 1 x 1 x 1 after final conv
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),  # keep your original
        )

    def forward(self, x):
        return self.net(x)  # (N, 1, 1, 1)


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
        labels = labels.to(device).float()  # BCE expects float targets

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
    """
    Computes overall accuracy and per-class accuracy.
    """
    net.eval()
    correct = 0
    total = 0
    class_correct = {cls: 0 for cls in classes}
    class_total = {cls: 0 for cls in classes}

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)                      # (N,1,1,1)
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
# Optional Baseline: ResNet18
# ---------------------------

def baseline_resnet18(
    train_set: data.Dataset,
    test_set: data.Dataset,
    epochs: int = 2,
    batch_size: int = 32,
):
    """
    Alternative scenario: Quick baseline using ResNet18 transfer learning.
    Useful to compare against the custom Discriminator.

    - Converts labels to 2-class cross-entropy (no sigmoid).
    - Uses ImageNet normalization for the model input.
    """
    import torchvision.models as models

    device = get_device()
    print("\n[Baseline] ResNet18 fine-tuning")

    # Build loaders (larger batch often OK for ResNet18)
    trainloader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    testloader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    # Model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace final layer for 2 classes
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # Loss/Opt
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train (very short run)
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"[ResNet18] Epoch {epoch+1}/{epochs} | loss: {running/max(1,len(trainloader)):.4f}")

    # Eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"[ResNet18] Test accuracy: {100.0*correct/max(1,total):.1f}%")

    return model


# ---------------------------
# Main Flow (Notebook-friendly)
# ---------------------------

def main():
    # === Runtime info (Recording: "Show the runtime is GPU") ===
    print_env_info()

    # === Colab: check Drive ===
    # In Colab notebook, run this in a cell before calling main():
    # from google.colab import drive; drive.mount('/content/drive')
    mount_colab_drive_if_needed()

    # === Folder walkthrough per your TODO recording ===
    show_folder_walkthrough(DATA_ROOT)

    # === Dataset ===
    set_reproducible(SEED)
    transform = build_transforms(IMAGE_SIZE)
    anime_faces_dataset = load_dataset(DATA_ROOT, transform)
    print(f"\nDataset size: {len(anime_faces_dataset)}")

    # Show classes and mapping
    classes = anime_faces_dataset.classes
    class_to_idx = anime_faces_dataset.class_to_idx
    print(f"Classes: {classes}")
    print(f"class_to_idx mapping: {class_to_idx}")

    # (Your note said each class has 1024 images; we’ll verify quickly)
    counts_by_class = {cls: 0 for cls in classes}
    for _, y in anime_faces_dataset.imgs:
        # imgs is (path, class_index)
        for cls, idx in class_to_idx.items():
            if y == idx:
                counts_by_class[cls] += 1
    print(f"Image counts by class: {counts_by_class}")

    # === Train/Test split (~80/20) ===
    train_set, test_set = split_dataset(anime_faces_dataset, train_ratio=0.80, seed=SEED)
    print(f"Train: {len(train_set)} | Test: {len(test_set)}")

    # === Dataloaders ===
    trainloader, testloader = make_dataloaders(train_set, test_set, BATCH_SIZE, NUM_WORKERS)

    # === Peek one batch ===
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print(f"One batch shape: {images.shape}")
    print("First 8 labels:", " ".join(f"{classes[labels[j]]:5s}" for j in range(min(8, len(labels)))))
    show_grid(images[:min(16, images.size(0))], nrow=8, title="Training batch (denormalized)")

    # === Model, Loss, Optimizer ===
    device = get_device()
    netD = Discriminator(nc=NC, ndf=NDF).to(device)
    netD.apply(weights_init)
    print("\nDiscriminator architecture:\n", netD)

    # Note: You used Sigmoid + BCELoss. We keep that:
    criterion = nn.BCELoss()
    optimizer = optim.Adam(netD.parameters(), lr=LR, betas=BETAS)

    # === Train ===
    print("\n[Training]")
    for epoch in range(EPOCHS):
        train_one_epoch(netD, trainloader, optimizer, criterion, device, log_every=20, epoch_index=epoch)
    print("Finished Training.")

    # === Quick visual sanity check on test batch ===
    dataiter = iter(testloader)
    test_images, test_labels = next(dataiter)
    print("\nGroundTruth:")
    print(" ".join(f"{classes[test_labels[j]]:5s}" for j in range(min(8, len(test_labels)))))
    show_grid(test_images[:min(16, test_images.size(0))], nrow=8, title="Test batch (denormalized)")

    # Predictions for the same batch
    with torch.no_grad():
        outputs = netD(test_images.to(device))      # (N,1,1,1)
        predicted = torch.round(outputs.flatten()).long().cpu()
    print("\nPredicted:")
    print(" ".join(f"{classes[predicted[j]]:5s}" for j in range(min(8, len(predicted)))))

    # === Full test evaluation ===
    metrics = evaluate(netD, testloader, device, classes)

    # === Save model checkpoint (optional) ===
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(netD.state_dict(), "./checkpoints/discriminator_anime_vs_badfake.pth")
    print("Saved: ./checkpoints/discriminator_anime_vs_badfake.pth")

    # === (Optional) Second scenario baseline ===
    # Uncomment to run a quick baseline using ResNet18 transfer learning:
    # baseline_resnet18(train_set, test_set, epochs=2, batch_size=32)


if __name__ == "__main__":
    main()
