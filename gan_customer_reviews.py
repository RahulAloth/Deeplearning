# python gan_customer_reviews.py --epochs 1000 --batch-size 64 --outdir runs/gan_reviews

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build & Evaluate a GAN: Synthetic Customer Reviews (Structured Data)

This script trains a simple GAN on simulated, tabular "customer review" features:
  - rating (1..5)
  - review_length (approx. sentence count proxy)
  - sentiment_score (0..1-ish)

Pipeline:
  1) Simulate "real" structured data and min-max normalize
  2) Define Generator and Discriminator (Keras / TensorFlow)
  3) Assemble GAN and train with BCE, label smoothing, and optional label noise
  4) Plot smoothed loss curves and estimate a convergence point
  5) Visualize real vs. synthetic distributions
  6) Convert synthetic rows to human-readable review snippets

Outputs:
  - Trained models: generator.h5, discriminator.h5
  - Plots: loss_curves.png, distributions.png
  - Samples: synthetic_samples.csv (with decoded review text)

Author: (c) 2026 — You
License: MIT
"""

from __future__ import annotations

import os
import argparse
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers, backend as K


# ---------------------------
# Utilities & Reproducibility
# ---------------------------

def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------
# Data: Simulate "real" table
# ---------------------------

def simulate_review_data(n_samples: int = 3000,
                         rating_low: int = 1,
                         rating_high: int = 5,
                         review_len_mu: float = 50.0,
                         review_len_sigma: float = 10.0,
                         sentiment_mu: float = 0.5,
                         sentiment_sigma: float = 0.15,
                         seed: int = 42) -> pd.DataFrame:
    """
    Create a small synthetic "real" dataset with 3 fields:
      rating ∈ {1..5}, review_length ~ N(50,10), sentiment_score ~ N(0.5,0.15)
    """
    rng = np.random.default_rng(seed)
    ratings = rng.integers(rating_low, rating_high + 1, size=n_samples)
    review_length = rng.normal(loc=review_len_mu, scale=review_len_sigma, size=n_samples)
    sentiment_score = rng.normal(loc=sentiment_mu, scale=sentiment_sigma, size=n_samples)

    df = pd.DataFrame({
        "rating": ratings,
        "review_length": review_length,
        "sentiment_score": sentiment_score
    })
    return df


def normalize_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df.values)
    return X, scaler


# ---------------------------
# Models: Generator & Discriminator
# ---------------------------

def build_generator(latent_dim: int, output_dim: int) -> tf.keras.Model:
    """
    MLP generator that maps z ~ N(0, I) to normalized feature vector x ∈ [0,1]^output_dim
    """
    model = models.Sequential(name="generator")
    model.add(layers.Input(shape=(latent_dim,), name="z"))

    model.add(layers.Dense(128, kernel_initializer="he_normal"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization(momentum=0.8))

    model.add(layers.Dense(128, kernel_initializer="he_normal"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization(momentum=0.8))

    model.add(layers.Dense(64, kernel_initializer="he_normal"))
    model.add(layers.LeakyReLU(0.2))

    # Data were min-max normalized → sigmoid to confine to [0,1]
    model.add(layers.Dense(output_dim, activation="sigmoid", name="x_fake"))

    return model


def build_discriminator(input_dim: int, l2: float = 1e-4, dropout: float = 0.3) -> tf.keras.Model:
    """
    MLP discriminator that outputs P(real | x) with sigmoid.
    Regularization to avoid early overpowering and overfitting.
    """
    model = models.Sequential(name="discriminator")
    model.add(layers.Input(shape=(input_dim,), name="x_in"))

    model.add(layers.Dense(128, kernel_initializer="he_normal",
                           kernel_regularizer=regularizers.l2(l2)))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(64, kernel_initializer="he_normal",
                           kernel_regularizer=regularizers.l2(l2)))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1, activation="sigmoid", name="p_real"))
    return model


def assemble_gan(generator: tf.keras.Model,
                 discriminator: tf.keras.Model,
                 lr: float = 2e-4,
                 beta1: float = 0.5,
                 beta2: float = 0.999) -> Tuple[tf.keras.Model, tf.keras.optimizers.Optimizer]:
    """
    Freeze D, connect z -> G(z) -> D, optimize G with BCE.
    """
    # Compile D standalone (for its own updates)
    d_opt = optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)
    discriminator.compile(optimizer=d_opt, loss="binary_crossentropy", metrics=["accuracy"])

    # Combined model for G updates
    discriminator.trainable = False
    z = layers.Input(shape=(generator.input_shape[1],), name="z_in")
    validity = discriminator(generator(z))
    gan = models.Model(z, validity, name="gan")
    g_opt = optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)
    gan.compile(optimizer=g_opt, loss="binary_crossentropy")

    return gan, d_opt


# ---------------------------
# Training
# ---------------------------

def sample_latent(batch_size: int, latent_dim: int) -> np.ndarray:
    return np.random.normal(0.0, 1.0, size=(batch_size, latent_dim))


def train_gan(X: np.ndarray,
              generator: tf.keras.Model,
              discriminator: tf.keras.Model,
              gan: tf.keras.Model,
              epochs: int = 1000,
              batch_size: int = 64,
              latent_dim: int = 32,
              label_smooth_real: float = 0.9,
              flip_labels_prob: float = 0.0,
              verbose_every: int = 100) -> Dict[str, list]:
    """
    Train loop with:
      - label smoothing for real labels (e.g., 0.9 instead of 1.0)
      - optional label flipping noise
      - one D step + one G step per epoch (simple vanilla schedule)
    """
    n = X.shape[0]
    y_real = np.ones((batch_size, 1)) * label_smooth_real
    y_fake = np.zeros((batch_size, 1))

    history = {
        "d_loss": [],
        "d_acc": [],
        "g_loss": [],
    }

    for epoch in range(1, epochs + 1):
        # ------------------
        # 1) Train Discriminator
        # ------------------
        idx = np.random.randint(0, n, batch_size)
        real_batch = X[idx]

        z = sample_latent(batch_size, latent_dim)
        fake_batch = generator.predict(z, verbose=0)

        # Optional label flipping noise
        y_real_noisy = y_real.copy()
        y_fake_noisy = y_fake.copy()
        if flip_labels_prob > 0.0:
            flip_r = np.random.rand(batch_size, 1) < flip_labels_prob
            flip_f = np.random.rand(batch_size, 1) < flip_labels_prob
            y_real_noisy[flip_r] = 0.0
            y_fake_noisy[flip_f] = label_smooth_real

        # Enable D updates
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_batch, y_real_noisy)
        d_loss_fake = discriminator.train_on_batch(fake_batch, y_fake_noisy)

        # Keras can return scalar or [loss, acc]; normalize:
        if isinstance(d_loss_real, (list, tuple)):
            d_loss = 0.5 * (float(d_loss_real[0]) + float(d_loss_fake[0]))
            d_acc = 0.5 * (float(d_loss_real[1]) + float(d_loss_fake[1]))
        else:
            d_loss = 0.5 * (float(d_loss_real) + float(d_loss_fake))
            d_acc = np.nan

        # ------------------
        # 2) Train Generator (via GAN)
        # ------------------
        z = sample_latent(batch_size, latent_dim)
        target = np.ones((batch_size, 1)) * label_smooth_real  # want D to predict "real"
        discriminator.trainable = False
        g_loss = gan.train_on_batch(z, target)

        # Record
        history["d_loss"].append(d_loss)
        history["d_acc"].append(d_acc)
        history["g_loss"].append(float(g_loss) if not isinstance(g_loss, (list, tuple)) else float(g_loss[0]))

        if (epoch % verbose_every) == 0 or epoch == 1:
            if np.isnan(d_acc):
                print(f"Epoch {epoch:4d}/{epochs} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")
            else:
                print(f"Epoch {epoch:4d}/{epochs} | D Loss: {d_loss:.4f} (acc {d_acc:.3f}) | G Loss: {g_loss:.4f}")

    return history


# ---------------------------
# Evaluation & Visualization
# ---------------------------

def smooth_curve(values: list[float], sigma: float = 3.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return gaussian_filter1d(arr, sigma=sigma)


def plot_losses(history: Dict[str, list], outpath: str) -> int:
    d_s = smooth_curve(history["d_loss"], sigma=5)
    g_s = smooth_curve(history["g_loss"], sigma=5)

    # Simple convergence proxy: point of minimal absolute difference
    diff = np.abs(d_s - g_s)
    conv_epoch = int(np.argmin(diff)) + 1

    plt.figure(figsize=(10, 5))
    plt.plot(d_s, label="Discriminator Loss (smoothed)", color="tab:blue")
    plt.plot(g_s, label="Generator Loss (smoothed)", color="tab:orange")
    plt.axvline(conv_epoch, color="tab:green", ls="--", label=f"≈ Convergence @ epoch {conv_epoch}")
    plt.title("GAN Training — Smoothed Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return conv_epoch


def synth_to_text(row: pd.Series) -> str:
    """
    Convert structured fields into a short, friendly review sentence.
    Assumes row contains: rating, review_length, sentiment_score
    """
    # Rating-based quality
    r = int(round(row["rating"]))
    if r >= 5:
        quality = "outstanding"
    elif r == 4:
        quality = "very good"
    elif r == 3:
        quality = "okay"
    elif r == 2:
        quality = "below average"
    else:
        quality = "poor"

    # Sentiment cue
    s = float(row["sentiment_score"])
    if s >= 0.75:
        sentiment = "Loved it."
    elif s >= 0.5:
        sentiment = "It was fine."
    else:
        sentiment = "Not a fan."

    # Length hint
    L = float(row["review_length"])
    if L >= 60:
        prefix = "In a longer take: "
    elif L < 40:
        prefix = "Quick note: "
    else:
        prefix = ""

    return f"{prefix}Overall, the product was {quality}. {sentiment}"


def plot_distributions(real_df: pd.DataFrame, synth_df: pd.DataFrame, outpath: str) -> None:
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cols = real_df.columns.tolist()
    for i, col in enumerate(cols):
        sns.kdeplot(real_df[col], label="Real", ax=axes[i], fill=True, color="tab:blue", alpha=0.3)
        sns.kdeplot(synth_df[col], label="Synthetic", ax=axes[i], fill=True, color="tab:orange", alpha=0.3)
        axes[i].set_title(f"{col} — Distribution")
        axes[i].legend()
    plt.suptitle("Real vs. Synthetic Feature Distributions")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ---------------------------
# Main
# ---------------------------

def main(args: argparse.Namespace) -> None:
    set_seeds(args.seed)
    ensure_dir(args.outdir)

    # 1) Data
    df_real = simulate_review_data(n_samples=args.n_samples, seed=args.seed)
    X, scaler = normalize_dataframe(df_real)
    input_dim = X.shape[1]
    latent_dim = args.latent_dim

    # 2) Models
    generator = build_generator(latent_dim=latent_dim, output_dim=input_dim)
    discriminator = build_discriminator(input_dim=input_dim, l2=1e-4, dropout=0.3)
    gan, _ = assemble_gan(generator, discriminator, lr=args.lr, beta1=args.beta1, beta2=args.beta2)

    print("\nModel summaries:\n")
    generator.summary()
    discriminator.summary()
    gan.summary()

    # 3) Training
    history = train_gan(
        X=X,
        generator=generator,
        discriminator=discriminator,
        gan=gan,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=latent_dim,
        label_smooth_real=args.label_smooth,
        flip_labels_prob=args.flip_prob,
        verbose_every=max(1, args.verbose_every),
    )

    # 4) Loss plots & convergence
    loss_plot_path = os.path.join(args.outdir, "loss_curves.png")
    conv_epoch = plot_losses(history, outpath=loss_plot_path)
    print(f"\nSaved smoothed loss curves → {loss_plot_path}")
    print(f"Approximate convergence epoch: {conv_epoch}")

    # 5) Sample synthetic data and invert scaling
    n_gen = args.eval_samples
    z = sample_latent(n_gen, latent_dim)
    synth_norm = generator.predict(z, verbose=0)
    synth_df = pd.DataFrame(scaler.inverse_transform(synth_norm), columns=df_real.columns)

    # Post-processing: clamp and round rating into [1,5]
    synth_df["rating"] = synth_df["rating"].clip(1, 5).round().astype(int)

    # 6) Distributions plot
    dist_plot_path = os.path.join(args.outdir, "distributions.png")
    plot_distributions(df_real, synth_df, outpath=dist_plot_path)
    print(f"Saved distributions plot → {dist_plot_path}")

    # 7) Decode to text reviews
    synth_df["synthetic_review"] = synth_df.apply(synth_to_text, axis=1)

    # 8) Save artifacts
    csv_path = os.path.join(args.outdir, "synthetic_samples.csv")
    synth_df.to_csv(csv_path, index=False)
    print(f"Saved synthetic samples → {csv_path}")

    gen_path = os.path.join(args.outdir, "generator.h5")
    disc_path = os.path.join(args.outdir, "discriminator.h5")
    generator.save(gen_path)
    discriminator.save(disc_path)
    print(f"Saved models → {gen_path}, {disc_path}")

    # Show a few rows inline (headless-safe)
    print("\nPreview of generated samples:")
    preview_cols = ["rating", "review_length", "sentiment_score", "synthetic_review"]
    with pd.option_context('display.max_colwidth', None):
        print(synth_df[preview_cols].head(10))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a simple GAN to generate structured customer review features.")
    p.add_argument("--n-samples", type=int, default=3000, help="Number of real (simulated) samples.")
    p.add_argument("--latent-dim", type=int, default=32, help="Dimensionality of latent noise z.")
    p.add_argument("--epochs", type=int, default=1000, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate for Adam.")
    p.add_argument("--beta1", type=float, default=0.5, help="Adam beta1.")
    p.add_argument("--beta2", type=float, default=0.999, help="Adam beta2.")
    p.add_argument("--label-smooth", type=float, default=0.9, help="Label smoothing value for 'real' labels.")
    p.add_argument("--flip-prob", type=float, default=0.0, help="Probability of flipping labels (regularization).")
    p.add_argument("--eval-samples", type=int, default=1000, help="How many synthetic samples to evaluate/plot.")
    p.add_argument("--outdir", type=str, default="runs/gan_customer_reviews", help="Output directory for artifacts.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--verbose-every", type=int, default=100, help="Print metrics every N epochs.")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(args)



