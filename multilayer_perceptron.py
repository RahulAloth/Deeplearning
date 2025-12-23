
"""
Multilayer Perceptron (MLP) for House Price Regression — Manual Backprop (NumPy)
---------------------------------------------------------------------------------
Architecture: 2 -> 32 -> 16 -> 1 (ReLU hidden layers, linear output)
Loss optimized: MSE (report RMSE)
Optimizer: Mini-batch SGD

This script shows *how* learning happens in an MLP:
- Forward pass through multiple layers
- Loss computation (MSE) and RMSE metric
- Backward pass: manual gradients for Linear+ReLU layers
- Parameter updates with SGD
"""

import numpy as np

# ---- Reproducibility ----
rng = np.random.default_rng(7)

# ---- Synthetic dataset: price ~ 3000 * size + 20000 * rooms + noise ----
N = 2000
size_m2 = rng.uniform(30, 200, size=N)         # house size in m^2
rooms   = rng.integers(1, 7, size=N)           # number of rooms
X = np.stack([size_m2, rooms], axis=1).astype(np.float32)  # (N, 2)

true_W = np.array([3000.0, 20000.0], dtype=np.float32)
true_b = np.float32(50000.0)
noise  = rng.normal(0, 20000, size=N).astype(np.float32)
y = (X @ true_W + true_b + noise).astype(np.float32)       # (N,)

# ---- Standardize features (helps optimization) ----
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0) + 1e-8
X_stdized = (X - X_mean) / X_std

# ---- Train/validation split ----
idx = rng.permutation(N)
train_idx, val_idx = idx[:1600], idx[1600:]
Xtr, ytr = X_stdized[train_idx], y[train_idx]
Xval, yval = X_stdized[val_idx], y[val_idx]

# ---- MLP parameters (He-like init for ReLU) ----
def he_init(fan_in, fan_out):
    # Kaiming/He uniform init scaled for ReLU
    limit = np.sqrt(6.0 / fan_in)
    return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)

W1 = he_init(2, 32);   b1 = np.zeros(32, dtype=np.float32)
W2 = he_init(32, 16);  b2 = np.zeros(16, dtype=np.float32)
W3 = he_init(16, 1);   b3 = np.zeros(1, dtype=np.float32)

# ---- Training hyperparameters ----
epochs = 200
batch_size = 64
lr = 0.03
weight_decay = 1e-4  # L2 regularization on weights (not biases)

# ---- Utilities ----
def relu(x):  # element-wise
    return np.maximum(x, 0.0)

def relu_grad(x):  # derivative w.r.t. pre-activation
    return (x > 0).astype(np.float32)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def batch_iter(X, y, batch_size):
    n = X.shape[0]
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = order[start:end]
        yield X[idx], y[idx]

# ---- Training loop ----
for epoch in range(1, epochs + 1):
    # Mini-batch SGD
    for Xb, yb in batch_iter(Xtr, ytr, batch_size):
        # Forward pass
        Z1 = Xb @ W1 + b1        # (B, 32)
        A1 = relu(Z1)            # (B, 32)

        Z2 = A1 @ W2 + b2        # (B, 16)
        A2 = relu(Z2)            # (B, 16)

        Z3 = A2 @ W3 + b3        # (B, 1)
        y_hat = Z3.squeeze(-1)   # (B,)

        # Loss (MSE)
        diff = (y_hat - yb)                      # (B,)
        mse  = np.mean(diff**2)

        # Add L2 on weights (not biases)
        l2 = weight_decay * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        loss = mse + l2

        # Backward pass
        B = Xb.shape[0]
        d_yhat = (2.0 / B) * diff                # dMSE/dy_hat

        # Output layer (linear)
        dZ3 = d_yhat[:, None]                    # (B, 1)
        grad_W3 = A2.T @ dZ3 + 2 * weight_decay * W3
        grad_b3 = np.sum(dZ3, axis=0)

        # Propagate to A2
        dA2 = dZ3 @ W3.T                         # (B, 16)
        dZ2 = dA2 * relu_grad(Z2)                # (B, 16)
        grad_W2 = A1.T @ dZ2 + 2 * weight_decay * W2
        grad_b2 = np.sum(dZ2, axis=0)

        # Propagate to A1
        dA1 = dZ2 @ W2.T                         # (B, 32)
        dZ1 = dA1 * relu_grad(Z1)                # (B, 32)
        grad_W1 = Xb.T @ dZ1 + 2 * weight_decay * W1
        grad_b1 = np.sum(dZ1, axis=0)

        # Parameter update (SGD)
        W3 -= lr * grad_W3;  b3 -= lr * grad_b3
        W2 -= lr * grad_W2;  b2 -= lr * grad_b2
        W1 -= lr * grad_W1;  b1 -= lr * grad_b1

    # ---- Epoch metrics ----
    # Train RMSE
    Z1 = Xtr @ W1 + b1; A1 = relu(Z1)
    Z2 = A1 @ W2 + b2;  A2 = relu(Z2)
    Z3 = A2 @ W3 + b3;  ytr_hat = Z3.squeeze(-1)
    train_rmse = rmse(ytr, ytr_hat)

    # Val RMSE
    Z1 = Xval @ W1 + b1; A1 = relu(Z1)
    Z2 = A1 @ W2 + b2;  A2 = relu(Z2)
    Z3 = A2 @ W3 + b3;  yval_hat = Z3.squeeze(-1)
    val_rmse = rmse(yval, yval_hat)

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | train RMSE: {train_rmse:,.2f} | val RMSE: {val_rmse:,.2f}")

print("\nLearned shapes:")
print("W1:", W1.shape, "b1:", b1.shape)
print("W2:", W2.shape, "b2:", b2.shape)
print("W3:", W3.shape, "b3:", b3.shape)
