# filename: pytorch_lightning_classification.py
"""
Educational example: A small multi-class classification model using PyTorch Lightning
------------------------------------------------------------------------------------
This script builds, trains, and evaluates a simple feed-forward neural network (MLP)
for **classification** on a synthetic dataset. It mirrors the structure used in the
regression example and highlights workshop concepts: tensors, activations, autograd,
loss minimization, and Lightning's high-level training loop.

Copyright & usage note:
- Original educational code (no third-party copyrighted code copied).
- You may adapt and reuse it. If you add external assets later, check licenses and
  include attribution where required.

How to run:
1) Install dependencies (PyTorch + Lightning 2.x):
   `pip install torch lightning`
2) Execute: `python module4_pytorch_lightning_classification.py`
3) Observe training logs; loss should decrease and accuracy/F1 should improve.
"""

from typing import Tuple
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

import lightning as L
from lightning.pytorch.loggers import CSVLogger

# -----------------------------
# 1) Reproducibility utilities
# -----------------------------

def set_seed(seed: int = 7) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# 2) Create a synthetic multi-class dataset
# -----------------------------

def make_synthetic_multiclass(n_samples: int = 12_000,
                              n_features: int = 20,
                              n_classes: int = 4,
                              class_sep: float = 2.5,
                              seed: int = 7) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate Gaussian clusters per class, separated by class_sep.

    We avoid external datasets to keep this example self-contained
    (no download/licensing constraints) while illustrating classification.
    """
    g = torch.Generator().manual_seed(seed)

    per_class = n_samples // n_classes
    X_list, y_list = [], []

    # Create class-specific means in different dimensions
    base_mean = torch.zeros(n_features)
    for c in range(n_classes):
        mean = base_mean.clone()
        mean[c % n_features] = class_sep  # shift a unique dimension per class
        cov = torch.eye(n_features)  # identity covariance for simplicity
        # Sample from N(mean, I)
        Xc = torch.randn(per_class, n_features, generator=g) @ cov + mean
        yc = torch.full((per_class,), c, dtype=torch.long)
        X_list.append(Xc)
        y_list.append(yc)

    X = torch.vstack(X_list)
    y = torch.cat(y_list)

    # Shuffle dataset
    idx = torch.randperm(X.size(0), generator=g)
    X = X[idx]
    y = y[idx]
    return X, y


# -------------------------------------------
# 3) LightningModule: model, loss, optimizer,
#    and training/validation logic
# -------------------------------------------

class Classifier(L.LightningModule):
    """A simple MLP classifier for tabular data.

    Concepts mapped to the workshop:
    - Tensors flow through Linear + ReLU layers (neurons/weights/biases).
    - Autograd tracks ops; Lightning handles backward() + optimizer steps.
    - Loss is cross-entropy for classification; we log Accuracy and macro-F1.
    """

    def __init__(self, n_features: int, n_classes: int, hidden_sizes=(128, 64), lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))  # logits
        self.net = nn.Sequential(*layers)

        self.loss_fn = nn.CrossEntropyLoss()

        # Accumulators for epoch-level metrics via confusion matrix
        self.n_classes = n_classes
        self.val_cm = None  # will be tensor [C, C] where rows=true, cols=pred

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # -------- Training --------
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # -------- Validation with macro-F1 --------
    def on_validation_epoch_start(self) -> None:
        device = self.device
        self.val_cm = torch.zeros(self.n_classes, self.n_classes, dtype=torch.long, device=device)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        # Update confusion matrix
        for t, p in zip(y.view(-1), preds.view(-1)):
            self.val_cm[t, p] += 1
        acc = (preds == y).float().mean()
        self.log("val_loss_step", loss, on_step=True, on_epoch=False)
        self.log("val_acc_step", acc, on_step=True, on_epoch=False)

    def on_validation_epoch_end(self) -> None:
        # Compute accuracy, precision/recall per class, and macro-F1
        cm = self.val_cm
        # True positives per class = diagonal
        tp = cm.diag().float()
        # Predicted positives per class = column sums
        pred_pos = cm.sum(dim=0).float().clamp(min=1)
        # Actual positives per class = row sums
        actual_pos = cm.sum(dim=1).float().clamp(min=1)

        precision = tp / pred_pos
        recall = tp / actual_pos
        f1 = 2 * (precision * recall) / (precision + recall).clamp(min=1e-12)
        macro_f1 = f1.mean()
        accuracy = tp.sum() / cm.sum().clamp(min=1)

        self.log("val_acc", accuracy, prog_bar=True, on_epoch=True)
        self.log("val_macro_f1", macro_f1, prog_bar=True, on_epoch=True)


# -----------------------------
# 4) Data preparation utilities
# -----------------------------

def make_dataloaders(batch_size: int = 256,
                     n_samples: int = 12_000,
                     n_features: int = 20,
                     n_classes: int = 4,
                     val_fraction: float = 0.2,
                     class_sep: float = 2.5,
                     seed: int = 7) -> Tuple[DataLoader, DataLoader]:
    X, y = make_synthetic_multiclass(n_samples=n_samples,
                                     n_features=n_features,
                                     n_classes=n_classes,
                                     class_sep=class_sep,
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
    set_seed(7)

    n_features = 20
    n_classes = 4
    batch_size = 256
    lr = 1e-3
    max_epochs = 15

    train_loader, val_loader = make_dataloaders(batch_size=batch_size,
                                                n_samples=12_000,
                                                n_features=n_features,
                                                n_classes=n_classes,
                                                val_fraction=0.2,
                                                class_sep=2.5,
                                                seed=7)

    model = Classifier(n_features=n_features, n_classes=n_classes, hidden_sizes=(128, 64), lr=lr)

    logger = CSVLogger("logs", name="lightning_synthetic_classification")
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        logger=logger,
        deterministic=False,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)

    # Final validation run
    trainer.validate(model, dataloaders=val_loader, verbose=True)


if __name__ == "__main__":
    main()
