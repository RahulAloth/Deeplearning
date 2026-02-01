#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a simple GAN (Generator + Discriminator) using dense neural networks (MLPs)
on the Fashion-MNIST dataset (28x28 grayscale).

- Downloads data to ./data
- Saves sample outputs to ./outputs
- Uses non-saturating loss (BCE with logits)
- Normalizes images to [-1, 1] (mean=0.5, std=0.5) so Generator can use Tanh.

Run:
    python train_gan_fashion_mnist.py \
        --epochs 20 --batch-size 16 --lr 2e-4 --z-dim 100 --save-every 1

For quick smoke test:
    python train_gan_fashion_mnist.py --epochs 1 --batch-size 16
"""

import os
import math
import time
import random
import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image

from tqdm import tqdm


# ----------------------------
# Reproducibility (optional)
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (slower):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Models: Dense Generator/Discriminator
# ----------------------------
class Generator(nn.Module):
    """
    MLP Generator: z -> 784 (1*28*28) with Tanh output in [-1, 1]
    """
    def __init__(self, z_dim: int = 100, img_dim: int = 28 * 28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, img_dim),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        x = x.view(x.size(0), 1, 28, 28)
        return x


class Discriminator(nn.Module):
    """
    MLP Discriminator: 784 -> 1 (logit)
    """
    def __init__(self, img_dim: int = 28 * 28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 1)  # logits (use with BCEWithLogitsLoss)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


# ----------------------------
# Weights initialization (DCGAN-ish init; safe for MLPs too)
# ----------------------------
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight, mean=1.0, std=0.02)
        nn.init.constant_(m.bias, 0.0)


# ----------------------------
# Data: Dataset & DataLoader
# ----------------------------
def get_dataloader(
    data_dir: Path,
    batch_size: int = 16,
    num_workers: int = 2,
    drop_last: bool = True,
    download: bool = True
) -> Tuple[DataLoader, int]:
    """
    Build Fashion-MNIST training DataLoader with normalization to [-1,1].
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                # [0,1], CxHxW
        transforms.Normalize((0.5,), (0.5,))  # -> roughly [-1, 1]
    ])
    train_ds = datasets.FashionMNIST(
        root=str(data_dir),
        train=True,
        transform=transform,
        download=download
    )

    # Windows note: if you hit issues, set num_workers=0
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return loader, len(train_ds)


# ----------------------------
# Utility: Save sample grid
# ----------------------------
@torch.no_grad()
def save_samples(
    generator: nn.Module,
    z_fixed: torch.Tensor,
    out_dir: Path,
    step_tag: str,
    value_range: Tuple[float, float] = (-1, 1)
):
    generator.eval()
    fake = generator(z_fixed).cpu()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Normalize=True + value_range ensures proper visualization from [-1,1]
    save_path = out_dir / f"samples_{step_tag}.png"
    save_image(fake, str(save_path), nrow=8, normalize=True, value_range=value_range)
    generator.train()
    return save_path


# ----------------------------
# Training Loop
# ----------------------------
def train(
    epochs: int,
    batch_size: int,
    z_dim: int,
    lr: float,
    betas: Tuple[float, float],
    data_dir: Path,
    out_dir: Path,
    device: torch.device,
    save_every: int = 1,
    num_workers: int = 2,
):
    # Data
    loader, n_train = get_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        download=True
    )
    n_batches = len(loader)
    print(f"Training samples: {n_train} | Batch size: {batch_size} | Batches/epoch: {n_batches}")

    # Models
    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator().to(device)
    G.apply(init_weights)
    D.apply(init_weights)

    # Loss & optimizers
    criterion = nn.BCEWithLogitsLoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)

    # Fixed noise for monitoring progress
    z_fixed = torch.randn(64, z_dim, device=device)

    # Training
    global_step = 0
    for epoch in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for real, _ in pbar:
            real = real.to(device)
            bsz = real.size(0)

            # ======================
            #  Train Discriminator
            # ======================
            opt_D.zero_grad(set_to_none=True)

            # Real images -> label=1
            logits_real = D(real)
            labels_real = torch.ones(bsz, 1, device=device)
            d_loss_real = criterion(logits_real, labels_real)

            # Fake images (no grad to G) -> label=0
            z = torch.randn(bsz, z_dim, device=device)
            with torch.no_grad():
                fake = G(z)
            logits_fake = D(fake)
            labels_fake = torch.zeros(bsz, 1, device=device)
            d_loss_fake = criterion(logits_fake, labels_fake)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_D.step()

            # ======================
            #  Train Generator (non-saturating)
            # ======================
            opt_G.zero_grad(set_to_none=True)

            z = torch.randn(bsz, z_dim, device=device)
            fake = G(z)
            logits_fake_for_G = D(fake)
            # Non-saturating loss: maximize log(D(G(z))) == minimize BCE(logits, 1)
            g_loss = criterion(logits_fake_for_G, labels_real)  # reuse labels_real=1 shape

            g_loss.backward()
            opt_G.step()

            # Logging
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            global_step += 1
            pbar.set_postfix({
                "D_loss": f"{d_loss.item():.4f}",
                "G_loss": f"{g_loss.item():.4f}"
            })

        # Epoch summary
        d_avg = epoch_d_loss / n_batches
        g_avg = epoch_g_loss / n_batches
        print(f"[Epoch {epoch:03d}] D_loss: {d_avg:.4f} | G_loss: {g_avg:.4f}")

        # Save samples
        if (epoch % save_every) == 0:
            sample_path = save_samples(G, z_fixed, out_dir, step_tag=f"epoch_{epoch:03d}")
            print(f"Saved samples to: {sample_path}")

    # Final snapshot
    final_path = save_samples(G, z_fixed, out_dir, step_tag="final")
    print(f"Training complete. Final samples: {final_path}")

    # Save model weights
    torch.save({
        "generator": G.state_dict(),
        "discriminator": D.state_dict(),
        "z_dim": z_dim
    }, out_dir / "gan_mlp_fashionmnist.pth")
    print(f"Saved model weights to: {out_dir / 'gan_mlp_fashionmnist.pth'}")


# ----------------------------
# Main
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple MLP GAN on Fashion-MNIST.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--z-dim", type=int, default=100, help="Dimension of the latent noise vector.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for Adam.")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.5, 0.999), help="Adam betas.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (for reproducibility).")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory to store/download dataset.")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Directory to save generated samples & weights.")
    parser.add_argument("--save-every", type=int, default=1, help="Save samples every N epochs.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (use 0 on Windows if needed).")
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    # Windows tip: if you encounter DataLoader issues, try num_workers=0
    if os.name == "nt" and args.num_workers > 0:
        print("On Windows? If you see DataLoader errors, re-run with --num-workers 0")

    start = time.time()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        z_dim=args.z_dim,
        lr=args.lr,
        betas=tuple(args.betas),
        data_dir=data_dir,
        out_dir=out_dir,
        device=device,
        save_every=args.save_every,
        num_workers=args.num_workers
    )
    elapsed = time.time() - start
    print(f"Total time: {elapsed/60:.2f} min")
