# filename: pytorch_lightning_demo.py
"""
Educational example: A small neural network using PyTorch Lightning
------------------------------------------------------------------
This script builds, trains, and evaluates a simple feed-forward neural network
on a synthetic classification dataset. The code is deliberately verbose and
well-commented to connect concepts such as tensors, autograd/gradients,
activation functions, loss minimization, and the Lightning training loop.

Copyright & usage note:
- This file is original, written for educational purposes. It does not copy
  text or code from third-party copyrighted sources.
- You are free to adapt and reuse it in your learning or projects. If you
  publish derivative work, please attribute appropriately and verify license
  compatibility for any external libraries you add.

How to run:
1) Ensure PyTorch and PyTorch Lightning are installed in your environment.
   (Lightning >= 2.x recommended.)
2) `python module4_pytorch_lightning_demo.py`
3) Observe training logs; accuracy should improve over epochs.
"""

import math
import random
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# PyTorch Lightning provides a higher-level interface for training.
import lightning as L
from lightning.pytorch.loggers import CSVLogger

# -----------------------------
# 1) Reproducibility utilities
# -----------------------------
def set_seed(seed: int = 42) -> None:
    """Set seeds for Python, PyTorch, and CUDA (if available) for repeatability.

    Note: Deterministic behavior can slightly slow down training, but it helps
    when you want to reproduce results and experiments consistently.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism you could also set torch.use_deterministic_algorithms(True),
    # but some ops may not have deterministic kernels.

# -----------------------------
# 2) Create a synthetic dataset
# -----------------------------
def make_synthetic_classification(n_samples: int = 10_000,
                                  n_features: int = 20,
                                  n_classes: int = 2,
                                  class_sep: float = 2.0,
                                  seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a simple, linearly-separable-ish dataset.

    We create features by sampling from Gaussian distributions and offsets
    (shifts) per class to make them separable. This keeps the example self-contained
    without relying on internet downloads or external datasets.
    """
    g = torch.Generator().manual_seed(seed)

    # Half the samples per class (assuming balanced classes)
    per_class = n_samples // n_classes
    features = []
    labels = []

    for c in range(n_classes):
        # Mean vector for class c: move means apart with class_sep
        mean = torch.zeros(n_features)
        mean[c % n_features] = class_sep  # shift one dimension per class

        # Sample from N(mean, I)
        x = torch.randn(per_class, n_features, generator=g) + mean
        y = torch.full((per_class,), c, dtype=torch.long)
        features.append(x)
        labels.append(y)

    X = torch.vstack(features)
    y = torch.cat(labels)

    # Shuffle the dataset to mix classes
    idx = torch.randperm(X.size(0), generator=g)
    X = X[idx]
    y = y[idx]
    return X, y

# -------------------------------------------
# 3) Define a LightningModule (the model, loss,
#    optimizer, and training/validation steps)
# -------------------------------------------
class SimpleClassifier(L.LightningModule):
    """A small feed-forward neural network for tabular data.

    Key ideas reflected here:
    - Tensors: `forward` consumes torch.Tensor inputs and returns logits.
    - Neurons/weights/biases: created via nn.Linear layers (affine transforms).
    - Activation functions: use ReLU to introduce non-linearities.
    - Loss: cross-entropy measures how far predictions are from true labels.
    - Autograd: Lightning/PyTorch track operations; `loss.backward()` (handled
      by Trainer) computes gradients w.r.t. parameters.
    - Optimizer: stochastic gradient descent variant (Adam) updates parameters
      to minimize the loss.
    """

    def __init__(self, n_features: int, n_classes: int, hidden_sizes=(64, 64), lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))  # final layer produces logits
        self.net = nn.Sequential(*layers)

        # CrossEntropyLoss expects raw logits and integer class labels.
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute logits for each class.

        The computation graph is built dynamically as these operations run.
        """
        return self.net(x)

    def configure_optimizers(self):
        """Create the optimizer. Adam is a convenient default.

        Lightning will call this to attach the optimizer and handle steps.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        """One training iteration over a mini-batch.

        1) Compute logits via forward pass.
        2) Compute loss versus ground-truth labels.
        3) Log metrics for monitoring.
        (Backward pass & optimizer step are handled by Lightning's Trainer.)
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Compute accuracy for logging (not used for optimization).
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Evaluate on validation data without updating weights."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

# -----------------------------
# 4) Data preparation utilities
# -----------------------------
def make_dataloaders(batch_size: int = 128,
                     n_samples: int = 10_000,
                     n_features: int = 20,
                     n_classes: int = 2,
                     val_fraction: float = 0.2,
                     seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders from synthetic data.

    Lightning prefers DataLoaders, which batch and optionally shuffle data.
    """
    X, y = make_synthetic_classification(n_samples=n_samples,
                                         n_features=n_features,
                                         n_classes=n_classes,
                                         seed=seed)
    dataset = TensorDataset(X, y)

    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader

# -----------------------------
# 5) Main: wire everything up
# -----------------------------
def main():
    set_seed(42)

    # Hyperparameters (feel free to tweak)
    n_features = 20
    n_classes = 2
    batch_size = 128
    lr = 1e-3
    max_epochs = 10

    train_loader, val_loader = make_dataloaders(batch_size=batch_size,
                                                n_samples=10_000,
                                                n_features=n_features,
                                                n_classes=n_classes,
                                                val_fraction=0.2,
                                                seed=42)

    model = SimpleClassifier(n_features=n_features,
                             n_classes=n_classes,
                             hidden_sizes=(64, 64),
                             lr=lr)

    # The Trainer orchestrates the training loop, validation, logging, and device placement.
    logger = CSVLogger("logs", name="lightning_synthetic_tabular")
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",  # uses GPU if available; otherwise CPU
        logger=logger,
        deterministic=False,  # set True for strict reproducibility
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)

    # After training, run a quick evaluation on the validation set.
    val_metrics = trainer.validate(model, dataloaders=val_loader, verbose=True)
    print("Final validation metrics:", val_metrics)

if __name__ == "__main__":
    main()
