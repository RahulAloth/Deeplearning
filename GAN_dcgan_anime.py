# dcgan_anime.py
# -*- coding: utf-8 -*-
"""
DCGAN trainer for anime faces (or any image folder) inspired by the PyTorch DCGAN tutorial.

This script converts the steps you described in your Colab notebook into a single
Python file suitable for a GitHub repository. It supports:

- (Optional) Google Drive mounting in Colab (guarded; safe to ignore elsewhere)
- Unzipping a dataset archive and cleaning the extra `__MACOSX` folder if present
- Setting up a PyTorch `ImageFolder` dataset and dataloader
- DCGAN Generator and Discriminator models with DCGAN-initialized weights
- Full training loop with periodic sample image grids
- Saving losses/accuracy plots and model checkpoints

Notes
-----
1) ImageFolder expects images inside at least one sub-folder. Your structure should be:

   images/
       anime_images/               # a folder acts as one class bucket
           img1.png
           img2.jpg
           ...

   If you have a ZIP, ensure it extracts into that shape (the code will do it if the zip
   contains an `anime_images/` directory at top-level).

2) DCGAN weight init follows the paper/tutorial convention: mean=0.0, std=0.02.

Usage examples
--------------
# Train on images under ./images (expects ./images/anime_images/*):
python dcgan_anime.py --images_root ./images --out_dir ./runs/anime_dcgan --epochs 50 --batch_size 64

# If your data is in a ZIP (contains folder anime_images/ at top-level):
python dcgan_anime.py --zip_path ./anime_images.zip --images_root ./images --out_dir ./runs/anime_dcgan

# In Google Colab, if your ZIP is in Drive, you may mount first (optional flag):
python dcgan_anime.py --mount_drive --zip_path /content/drive/MyDrive/ai_workshop_dcgans/anime_classification/anime_images.zip \
    --images_root /content/images --out_dir /content/runs/anime_dcgan

Artifacts
---------
- Sample image grids saved during training under: <out_dir>/samples/
- Loss and score plots saved as: <out_dir>/losses.png and <out_dir>/scores.png
- Model checkpoints (each epoch): <out_dir>/checkpoints/netG_epoch_XXX.pt, netD_epoch_XXX.pt

"""

from __future__ import annotations
import argparse
import os
import random
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision.utils import save_image

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    # Fallback if tqdm is unavailable
    def tqdm(x, *args, **kwargs):
        return x


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def maybe_mount_drive() -> None:
    """Attempt to mount Google Drive if running in Colab and flag provided."""
    try:
        from google.colab import drive  # type: ignore
        print("[Info] Detected Colab; mounting Google Drive at /content/drive ...")
        drive.mount("/content/drive")
    except Exception as e:
        print("[Warn] google.colab not available or mount failed. Proceeding without drive mount.")
        print(f"       Details: {e}")


def unzip_dataset(zip_path: str, extract_to: str) -> None:
    """Unzip archive into extract_to and remove __MACOSX if present."""
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    ensure_dir(extract_to)
    print(f"[Info] Extracting '{zip_path}' -> '{extract_to}' ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)

    macosx_dir = os.path.join(extract_to, "__MACOSX")
    if os.path.isdir(macosx_dir):
        print("[Info] Removing extra __MACOSX folder ...")
        import shutil
        shutil.rmtree(macosx_dir, ignore_errors=True)


# -----------------------------
# Models and initialization
# -----------------------------

def weights_init(m: nn.Module) -> None:
    """DCGAN weight initialization: normal with mean=0.0, std=0.02."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # input Z: (nz) x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # output in [-1, 1]
            # (nc) x 64 x 64
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, ndf: int = 64, nc: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # input: (nc) x 64 x 64
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

            # (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Data
# -----------------------------

def get_dataloader(images_root: str, image_size: int, batch_size: int, num_workers: int = 2,
                    drop_last: bool = True) -> torch.utils.data.DataLoader:
    """Create ImageFolder dataset and dataloader.

    Expect structure like images_root/<one_or_more_subfolders>/*.jpg
    """
    tfms = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = dset.ImageFolder(root=images_root, transform=tfms)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No images found under '{images_root}'. Ensure there is at least one subfolder "
            f"(e.g., '{images_root}/anime_images/') containing images.")

    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
    )
    return dl


# -----------------------------
# Training
# -----------------------------

def denorm(images: torch.Tensor) -> torch.Tensor:
    """Map from [-1,1] back to [0,1] for saving."""
    return (images.clamp(-1, 1) + 1) / 2


def save_grid(tensor: torch.Tensor, path: str, nrow: int = 8) -> None:
    ensure_dir(os.path.dirname(path))
    tensor = denorm(tensor)
    save_image(tensor, path, nrow=nrow)


def plot_training(G_losses, D_losses, real_scores, fake_scores, out_dir: str) -> None:
    ensure_dir(out_dir)

    # Losses
    plt.figure(figsize=(10, 6))
    plt.plot(G_losses, label='Generator loss', alpha=0.7)
    plt.plot(D_losses, label='Discriminator loss', alpha=0.7)
    plt.title('Training Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'losses.png'))
    plt.close()

    # Scores
    plt.figure(figsize=(10, 6))
    plt.plot(real_scores, label='Real score D(x)', alpha=0.7)
    plt.plot(fake_scores, label='Fake score D(G(z)) before D update', alpha=0.7)
    plt.title('Accuracy Scores')
    plt.xlabel('Iteration')
    plt.ylabel('Score (probability)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'scores.png'))
    plt.close()


def train(
    images_root: str,
    out_dir: str,
    *,
    image_size: int = 64,
    batch_size: int = 64,
    nz: int = 100,
    ngf: int = 64,
    ndf: int = 64,
    lr: float = 2e-4,
    beta1: float = 0.5,
    epochs: int = 50,
    sample_every: int = 200,
    nrow: int = 8,
) -> None:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Using device: {device}")

    dl = get_dataloader(images_root, image_size, batch_size)

    # Models
    netG = Generator(nz=nz, ngf=ngf, nc=3).to(device)
    netD = Discriminator(ndf=ndf, nc=3).to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    print("[Info] Generator architecture:\n", netG)
    print("[Info] Discriminator architecture:\n", netD)

    # Loss & Optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Fixed noise for tracking progress (always 64)
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Tracking
    G_losses, D_losses, real_scores, fake_scores = [], [], [], []
    iters = 0

    # Output dirs
    samples_dir = os.path.join(out_dir, 'samples')
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    ensure_dir(samples_dir)
    ensure_dir(ckpt_dir)

    for epoch in tqdm(range(epochs), desc="Epoch"):
        for i, (real_images, _) in enumerate(dl):
            # -----------------------------
            # (1) Update D: maximize log(D(x)) + log(1 - D(G(z)))
            # -----------------------------
            netD.zero_grad()

            real_images = real_images.to(device)
            bsz = real_images.size(0)
            real_labels = torch.full((bsz,), 1.0, dtype=torch.float, device=device)
            fake_labels = torch.full((bsz,), 0.0, dtype=torch.float, device=device)

            # Real pass
            output_real = netD(real_images).view(-1)
            errD_real = criterion(output_real, real_labels)
            errD_real.backward()
            D_x = output_real.mean().item()

            # Fake pass
            noise = torch.randn(bsz, nz, 1, 1, device=device)
            fake_images = netG(noise)
            output_fake = netD(fake_images.detach()).view(-1)
            errD_fake = criterion(output_fake, fake_labels)
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # -----------------------------
            # (2) Update G: maximize log(D(G(z)))
            # -----------------------------
            netG.zero_grad()
            output = netD(fake_images).view(-1)
            errG = criterion(output, real_labels)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Logging & sampling
            if (i % sample_every == 0) or (i == len(dl) - 1):
                print(
                    f"Epoch {epoch:03d} | Step {i:04d} | "
                    f"D_real: {errD_real.item():.3f} | D_fake: {errD_fake.item():.3f} | "
                    f"D_total: {errD.item():.3f} | G: {errG.item():.3f} | "
                    f"Real_score: {D_x:.3f} | Fake_score: {D_G_z1:.3f} | Fake_after_D: {D_G_z2:.3f}"
                )

                with torch.no_grad():
                    fake_grid = netG(fixed_noise).detach().cpu()
                grid_path = os.path.join(samples_dir, f"epoch{epoch:03d}_step{i:04d}.png")
                save_grid(fake_grid, grid_path, nrow=nrow)

            # Track
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            real_scores.append(D_x)
            fake_scores.append(D_G_z1)

            iters += 1

        # Save checkpoints each epoch
        torch.save(netG.state_dict(), os.path.join(ckpt_dir, f"netG_epoch_{epoch:03d}.pt"))
        torch.save(netD.state_dict(), os.path.join(ckpt_dir, f"netD_epoch_{epoch:03d}.pt"))

    # Final plots
    plot_training(G_losses, D_losses, real_scores, fake_scores, out_dir)
    print(f"[Done] Training complete. Artifacts saved under: {out_dir}")


# -----------------------------
# Entrypoint / CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DCGAN on an ImageFolder dataset (anime faces).")

    # Data
    p.add_argument('--images_root', type=str, default='./images',
                   help='Root folder containing at least one subfolder with images (e.g., ./images/anime_images)')
    p.add_argument('--zip_path', type=str, default=None,
                   help='Optional path to a ZIP file to extract into images_root')

    # Output
    p.add_argument('--out_dir', type=str, default='./runs/dcgan_run',
                   help='Directory to store samples, checkpoints, and plots')

    # Model/Train
    p.add_argument('--image_size', type=int, default=64, help='Spatial size of training images (default: 64)')
    p.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    p.add_argument('--nz', type=int, default=100, help='Size of latent vector (default: 100)')
    p.add_argument('--ngf', type=int, default=64, help='Generator feature map size (default: 64)')
    p.add_argument('--ndf', type=int, default=64, help='Discriminator feature map size (default: 64)')
    p.add_argument('--lr', type=float, default=2e-4, help='Learning rate (default: 2e-4)')
    p.add_argument('--beta1', type=float, default=0.5, help='Adam beta1 (default: 0.5)')
    p.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    p.add_argument('--sample_every', type=int, default=200, help='Steps between sample grids (default: 200)')
    p.add_argument('--nrow', type=int, default=8, help='Grid columns for sample images (default: 8)')

    # Misc
    p.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    p.add_argument('--mount_drive', action='store_true', help='Attempt to mount Google Drive (Colab only)')

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Optionally mount drive (for Colab)
    if args.mount_drive:
        maybe_mount_drive()

    # Optionally unzip into images_root
    if args.zip_path is not None:
        unzip_dataset(args.zip_path, args.images_root)
        print("[Info] After extraction, contents under images_root:")
        for root, dirs, files in os.walk(args.images_root):
            depth = root.replace(args.images_root, '').count(os.sep)
            indent = '  ' * max(depth, 0)
            print(f"{indent}{os.path.basename(root) or args.images_root}/")
            for d in dirs:
                print(f"{indent}  {d}/")
            # show a few files per folder
            for f in files[:5]:
                print(f"{indent}  {f}")

    ensure_dir(args.out_dir)

    train(
        images_root=args.images_root,
        out_dir=args.out_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        nz=args.nz,
        ngf=args.ngf,
        ndf=args.ndf,
        lr=args.lr,
        beta1=args.beta1,
        epochs=args.epochs,
        sample_every=args.sample_every,
        nrow=args.nrow,
    )


if __name__ == '__main__':
    main()
