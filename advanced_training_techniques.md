# Batch Normalization — Study Notes

## Overview
Batch normalization is a technique used in deep learning to stabilize and speed up training. It reduces changes in the distribution of activations as the network learns, helping each layer receive inputs with a more consistent scale.

---

## Why It’s Needed
During training, the inputs to each layer may shift as earlier layers update their weights. This phenomenon, often referred to as **internal covariate shift**, can slow learning and make optimization more difficult. Batch normalization reduces these shifts by standardizing inputs within each mini-batch.

---

## How Batch Normalization Works

### 1. Compute Batch Statistics
For each feature in a mini-batch:
- Calculate the **mean**.
- Calculate the **variance**.

These values describe how the inputs are distributed within that batch.

### 2. Normalize the Inputs
Each input is standardized using the batch statistics:

\[
\hat{x}_i = \frac{x_i - \mu_B}{\sigma_B}
\]

Where:
- \( x_i \) = original input  
- \( \mu_B \) = batch mean  
- \( \sigma_B \) = batch standard deviation  
- \( \hat{x}_i \) = normalized input  

This ensures the inputs have zero mean and unit variance.

### 3. Scale and Shift
To allow the network to learn an optimal representation, two trainable parameters are applied:

\[
y_i = \gamma \hat{x}_i + \beta
\]

Where:
- \( \gamma \) = learnable scale parameter  
- \( \beta \) = learnable shift parameter  
- \( y_i \) = final output after batch normalization  

---

## Advantages

- **Faster Training:** Helps stabilize gradients, allows larger learning rates.
- **Better Generalization:** Acts like a regularizer, reducing overfitting.
- **Reduced Sensitivity:** Less dependence on weight initialization.
- **Supports Deeper Networks:** Mitigates vanishing and exploding gradients.
- **Simplifies Hyperparameter Tuning:** More forgiving training dynamics.

---

## Limitations

- **Batch Size Dependence:** Small batches may produce poor estimates of mean and variance.
- **Additional Computation:** Requires extra operations per batch.
- **Less Effective for Certain Models:** Not ideal for tasks with very small batches or for certain recurrent sequence models.

---

## Summary
Batch normalization standardizes inputs within each mini-batch, then applies learnable scaling and shifting. This stabilizes training, makes deeper architectures more feasible, and often improves overall model performance, although it comes with trade-offs related to batch size and computation.

# Applying Batch Normalization to a Deep Learning Model — Study Notes

## Overview
Batch normalization can be added to neural networks to help stabilize training, improve convergence, and make deeper architectures easier to train. It works by normalizing the outputs of one layer before they are passed to the next. In practice, applying it in modern deep learning frameworks is straightforward.

---

## Model Structure in This Example
A typical setup might include:

- **Input layer:** size 784 (common for flattened 28×28 images)
- **Hidden layer 1:** Dense layer with 512 units + ReLU activation  
- **Batch Normalization**
- **Hidden layer 2:** Dense layer with 128 units + ReLU activation  
- **Batch Normalization**
- **Output layer:** Dense layer with 10 units (e.g., for classification)

Batch normalization layers are inserted **between** Dense layers to normalize each layer’s activations before they are passed forward.

---

## Required Imports
To build such a model, the following components are typically imported:

```python
from keras.layers import Input, Dense, BatchNormalization
from keras import Sequential
```

## Applying Batch Normalization in a Sequential Model
### A simplified example of adding batch normalization between layers:
```python
model = Sequential([
    Input(shape=(784,)),

    Dense(512, activation="relu"),
    BatchNormalization(),

    Dense(128, activation="relu"),
    BatchNormalization(),

    Dense(10, activation="softmax")
])
```

## Key idea:
- You directly insert a BatchNormalization() layer after a Dense layer and before the next layer. This automatically performs normalization during training and uses stored statistics during inference.
- 
## Workflow Summary
- Prepare data
  - Load and preprocess input data before building the model.
- Select runtime environment
  - Make sure the correct Python kernel or environment is active (e.g., Python 3.10).
- Build the model
  - Add input, Dense layers, and BatchNormalization layers in the desired order.
- Run the model definition
  - Some frameworks may display informational warnings—for example, about missing GPU hardware—but these do not affect CPU execution.
- Proceed to training
  - After initialization, the model can be compiled and trained, benefiting from more stable learning dynamics.

## Practical Notes
- Batch normalization often leads to smoother and faster convergence.
- It can reduce sensitivity to initialization and learning‑rate choices.
- CPU-based execution works fine for demonstration or small datasets.
- This technique is commonly used before applying more advanced training tricks such as:
  - Gradient clipping
  - Early stopping
  - Learning-rate scheduling
## Conclusion
-  To apply batch normalization in a deep learning model, you simply insert normalization layers between Dense layers. This enhances training stability and performance with minimal code changes. After setting up this foundation, additional optimization techniques can be integrated in the workflow.

-  # Gradient Clipping — Study Notes

## Overview
Gradient clipping is a method used during neural network training to prevent the gradients from becoming excessively large. Extremely large gradients can cause unstable training behavior, numerical errors, or prevent the model from converging effectively.

This issue is known as the **exploding gradients problem**, and it occurs most often in deep networks or recurrent architectures where gradients must propagate through many layers or time steps.

---

## Purpose of Gradient Clipping
Gradient clipping limits how large the gradients are allowed to grow. By restricting gradient magnitude, the training process becomes more stable, updates remain controlled, and optimization progresses more smoothly.

---

## Two Main Types of Gradient Clipping

### 1. Clipping by Value
Each gradient component is forced to stay within a specified range:

- Gradients larger than a positive threshold are capped at the threshold.
- Gradients smaller than a negative threshold are raised to the negative threshold.

Example workflow:
- Let the gradient vector be:  
  `G = [2, -6, 8, -3, 5]`
- Let the clipping threshold be:  
  `C = 4`  
- Components above +4 become +4, and those below –4 become –4.

Result after clipping:  
`[2, -4, 4, -3, 4]`

This method is simple and limits abrupt updates to specific parameters.

---

### 2. Clipping by Norm
Instead of clipping individual values, the **entire gradient vector** is rescaled if its overall size (norm) exceeds a threshold.

Steps:
1. Compute the L2 norm of the gradient vector.  
2. If the norm is greater than a selected threshold `C`, scale the entire vector so that its norm becomes exactly `C`.

Example:
- Gradient vector: `G = [2, -6, 8, -3, 5]`
- Threshold: `C = 6`
- L2 norm ≈ `11.75` (which exceeds the threshold)
- Scaling factor: `6 / 11.75 ≈ 0.51`
- Multiply each component by `0.51`  
  Result: `[1.02, -3.06, 4.08, -1.53, 2.55]`

Clipping by norm maintains the direction of the gradient while reducing its magnitude.

---

## When to Use Each Method

### Use clipping by value when:
- You want a very simple approach.
- Individual gradient components must be kept under strict bounds.
- You don’t need to preserve the exact direction of the gradient vector.

### Use clipping by norm when:
- You want to preserve the gradient’s direction.
- You want consistent scaling across all parameters.
- You are training deep or recurrent models where exploding gradients are more common.

---

## Benefits of Gradient Clipping
- **Stabilized training:** Prevents sudden large updates to model weights.
- **Helps with convergence:** Keeps learning on track even in deep architectures.
- **Especially useful for RNNs:** Recurrent networks often suffer from exploding gradients due to long backpropagation chains.

---

## Limitations
- Does not fix the root causes of exploding gradients, such as:
  - Poor initialization
  - Inappropriate model design
- Selecting a suitable threshold can be challenging and often requires experimentation.
- Can act as a temporary solution rather than addressing deeper architectural problems.

---

## Summary
Gradient clipping is a practical technique that controls gradient size during training, ensuring stable and predictable updates. It comes in two common forms—clipping by value and clipping by norm—each suited to different situations. While not a complete solution to training instability, it plays an important role in deep learning optimization.

``

# Applying Gradient Clipping to a Deep Learning Model — Study Notes

## Overview
Gradient clipping is applied at **optimization time** to prevent excessively large gradient updates. In practice, you either cap each gradient component (**clip by value**) or rescale the whole gradient vector if its norm exceeds a threshold (**clip by norm**). Most deep learning libraries expose simple switches to enable this during training.

---

## Why Apply It During Training?
- **Stabilizes updates:** Prevents large, erratic parameter jumps.
- **Aids convergence:** Especially helpful for deep and recurrent networks.
- **Easy to enable:** Usually a one-line change in the optimizer or training step.

---

## Typical Workflow (High-Level)
1. **Build or load your model** (e.g., after adding BatchNorm if applicable).
2. **Choose an optimizer** (e.g., Adam).
3. **Enable gradient clipping** via:
   - A **norm threshold** (recommended default approach).
   - A **value cap** (simpler but may distort gradient direction).
4. **Compile and train** the model as usual.
5. **Tune thresholds** (`clipnorm` or `clipvalue`) based on stability and performance.

---

## Keras / TensorFlow Example (Clip by Norm)

```python
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Input
from keras.optimizers import Adam

# Example architecture (e.g., for 28x28 images flattened to 784)
model = Sequential([
    Input(shape=(784,)),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dense(10, activation="softmax")
])

# Apply gradient clipping by L2 norm (e.g., threshold = 1.0)
optimizer = Adam(learning_rate=1e-3, clipnorm=1.0)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=..., batch_size=...)
```

## Notes
- clipnorm=1.0 limits the L2 norm of gradients to 1.0 per update.
- Start with 1.0 and adjust if training remains unstable (lower) or too slow (slightly higher).
## Keras / TensorFlow Example (Clip by Value)

from keras.optimizers import Adam

# Each gradient component is clipped into [-0.5, 0.5]
```python
optimizer = Adam(learning_rate=1e-3, clipvalue=0.5)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

## When to prefer:
- If specific parameters exhibit spikes you want to hard-cap.
- When simplicity is more important than preserving gradient direction.
## PyTorch Example (Clip by Norm)
```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Linear(128, 10)
)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

max_norm = 1.0  # L2 norm threshold

for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # Clip the gradients by norm before stepping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        optimizer.step()
```
## PyTorch Example (Clip by Value)
- torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
- Use this after loss.backward() and before optimizer.step().
Choosing Thresholds

## Starting points:
```python
clipnorm in [0.5, 5.0] (common defaults: 1.0 or 2.0).
clipvalue in [0.1, 1.0] depending on observed gradient magnitudes.
```

## Tune with validation:
- If loss oscillates or diverges → lower the threshold or learning rate.
- If training is stable but slow → consider slightly higher threshold.

## Practical Tips
- Combine with other stabilizers:
- Batch Normalization / Layer Normalization
- Reasonable learning rates
- Proper initialization

## Monitor:
- Training/validation loss curves
- Gradient norms (optional logging) to verify clipping is engaged.
- CPU-only environments are fine for demonstration; GPU warnings about availability can be ignored if you don’t need GPU acceleration.

## Summary
- To apply gradient clipping:
- Keras: set clipnorm or clipvalue in the optimizer (e.g., Adam(clipnorm=1.0)).
- PyTorch: call clip_grad_norm_ or clip_grad_value_ after backward() and before step().
