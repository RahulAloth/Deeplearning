# filename: pytorch_lightning_regression.py
"""
Educational example: A small regression model using PyTorch Lightning
--------------------------------------------------------------------
This script builds, trains, and evaluates a simple feed-forward neural network
(MLP) for **regression** on a synthetic dataset. It mirrors the structure of the
classification example: well-commented code that connects tensors, autograd,
activation functions, loss minimization, metrics, and Lightning's higher-level
training loop.

Copyright & usage note:
- This file is original, written for educational purposes. It does not copy
  text or code from third-party copyrighted sources.
- You may adapt and reuse it. If you publish derivative work, attribute where
  appropriate and observe licenses for any external assets you add.

How to run:
1) Install dependencies (PyTorch + Lightning 2.x):
   `pip install torch lightning`
2) Execute: `python module4_pytorch_lightning_regression.py`
3) Check the console logs for loss/metrics; you should see MSE and MAE improve,
   and R^2 approach 1.0 on this synthetic problem.
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

def set_seed(seed: int = 123) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# 2) Create a synthetic regression dataset
# -----------------------------

def make_synthetic_regression(n_samples: int = 10_000,
                              n_features: int = 20,
                              noise_std: float = 0.5,
                              non_linear: bool = True,
                              seed: int = 123) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate y from X with known parameters plus noise.

    y = X @ w_true + b + optional_nonlinearity + Gaussian noise

    Keeping the data synthetic avoids external downloads and licensing issues and
    makes the example runnable in constrained environments.
    """
    g = torch.Generator().manual_seed(seed)

    # Features sampled from a standard normal distribution
    X = torch.randn(n_samples, n_features, generator=g)

    # Ground-truth linear weights and bias
    w_true = torch.linspace(1.0, 2.0, n_features)  # shaped (n_features,)
    b_true = 0.75

    y = X @ w_true + b_true

    if non_linear:
        # Add a mild non-linear term to make the task slightly richer
        y = y + 0.5 * torch.sin(X[:, 0]) + 0.25 * (X[:, 1] ** 2)

    # Additive Gaussian noise
    y = y + noise_std * torch.randn(n_samples, generator=g)

    # Reshape y to (N, 1) for regression heads that return shape (N, 1)
    y = y.unsqueeze(1)
    return X, y


# -------------------------------------------
# 3) LightningModule: model, loss, optimizer,
#    and training/validation logic
# -------------------------------------------

class Regressor(L.LightningModule):
    """A simple MLP regressor for tabular data.

    Concepts mapped to the workshop:
    - Tensors flow through Linear + ReLU layers (neurons/weights/biases).
    - Autograd tracks ops; Lightning handles backward() + optimizer steps.
    - Loss is MSE for regression; we also log MAE and R^2.
    """

    def __init__(self, n_features: int, hidden_sizes=(128, 64), lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # scalar regression output
        self.net = nn.Sequential(*layers)

        self.loss_fn = nn.MSELoss()

        # Accumulators for epoch-level validation metrics (for a correct R^2)
        self.val_count = 0
        self.val_sum_y = 0.0
        self.val_sum_y2 = 0.0
        self.val_sse = 0.0  # sum of squared errors
        self.val_mae_sum = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # -------- Training --------
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        mae = torch.mean(torch.abs(preds - y))
        self.log("train_mse", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # -------- Validation (epoch-level R^2) --------
    def on_validation_epoch_start(self) -> None:
        # Reset accumulators (store as Python floats for numerical stability)
        self.val_count = 0
        self.val_sum_y = 0.0
        self.val_sum_y2 = 0.0
        self.val_sse = 0.0
        self.val_mae_sum = 0.0

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)

        # Update accumulators
        with torch.no_grad():
            n = y.numel()
            self.val_count += n
            self.val_sum_y += float(y.sum())
            self.val_sum_y2 += float((y ** 2).sum())
            self.val_sse += float(((y - preds) ** 2).sum())
            self.val_mae_sum += float(torch.abs(y - preds).sum())

        # Log per-step MSE (Lightning will average across steps)
        self.log("val_mse_step", loss, on_step=True, on_epoch=False, prog_bar=False)

    def on_validation_epoch_end(self) -> None:
        if self.val_count > 0:
            mean_y = self.val_sum_y / self.val_count
            sst = self.val_sum_y2 - self.val_count * (mean_y ** 2)
            # Guard against the (rare) degenerate case of zero variance
            r2 = 1.0 - (self.val_sse / sst) if sst > 1e-12 else 0.0
            mse = self.val_sse / self.val_count
            mae = self.val_mae_sum / self.val_count

            self.log("val_mse", mse, prog_bar=True, on_epoch=True)
            self.log("val_mae", mae, prog_bar=True, on_epoch=True)
            self.log("val_r2", r2, prog_bar=True, on_epoch=True)


# -----------------------------
# 4) Data preparation utilities
# -----------------------------

def make_dataloaders(batch_size: int = 256,
                     n_samples: int = 20_000,
                     n_features: int = 20,
                     val_fraction: float = 0.2,
                     noise_std: float = 0.5,
                     seed: int = 123) -> Tuple[DataLoader, DataLoader]:
    X, y = make_synthetic_regression(n_samples=n_samples,
                                     n_features=n_features,
                                     noise_std=noise_std,
                                     non_linear=True,
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
    set_seed(123)

    # Hyperparameters (feel free to tweak)
    n_features = 20
    batch_size = 256
    lr = 1e-3
    max_epochs = 15

    train_loader, val_loader = make_dataloaders(batch_size=batch_size,
                                                n_samples=20_000,
                                                n_features=n_features,
                                                val_fraction=0.2,
                                                noise_std=0.5,
                                                seed=123)

    model = Regressor(n_features=n_features, hidden_sizes=(128, 64), lr=lr)

    logger = CSVLogger("logs", name="lightning_synthetic_regression")
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",  # uses GPU if available
        logger=logger,
        deterministic=False,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)

    # Final validation run
    trainer.validate(model, dataloaders=val_loader, verbose=True)


if __name__ == "__main__":
    main()
