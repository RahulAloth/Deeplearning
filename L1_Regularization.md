
# Applying L1 Regularization (Lasso) in Deep Learning

## Overview
L1 regularization, commonly known as **Lasso Regularization**, is a powerful technique used in deep learning to reduce **overfitting** and improve **model generalization**. It works by adding a penalty to the loss function that is proportional to the **absolute value of the weights**. This penalty encourages the model to learn **sparse weight representations**, meaning that many weights are driven to zero.

In this study note, we explore:
- Why L1 regularization is needed  
- How it helps combat overfitting  
- How to apply it in a Keras deep learning model  
- How to interpret training results  
- Additional learning insights and best practices  

---

## Detecting Overfitting
A common indicator of overfitting is a **divergence between training and validation loss curves**:
- Training loss continues to decrease
- Validation loss stagnates or increases

This pattern indicates that the model is memorizing training data instead of learning generalizable patterns.

> ✅ **Goal:** Reduce this divergence to improve generalization to unseen data.

---

## What is L1 Regularization?
L1 regularization modifies the loss function as follows:

\[
\text{Loss} = \text{Original Loss} + \lambda \sum |w_i|
\]

Where:
- \( w_i \) are the model weights
- \( \lambda \) (alpha) is the regularization strength

### Key Properties
- Encourages **sparse models** (many weights become exactly zero)
- Acts as an implicit **feature selector**
- Helps reduce variance and overfitting

---

## Applying L1 Regularization in Keras

### Step 1: Import the Regularizer
```python
from tensorflow.keras.regularizers import l1
```

### Step 2: Define the Regularized Model
Apply the regularization to the kernel (weights) of each hidden layer.

```python
model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=l1(0.001)),
    Dense(64, activation='relu', kernel_regularizer=l1(0.001)),
    Dense(1, activation='sigmoid')
])
```

#### Explanation:
- kernel_regularizer=l1(0.001) adds an L1 penalty
- 0.001 controls the strength of regularization
- Larger values increase sparsity but may cause underfitting

### Step 3: Compile the Model

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### Step 4: Train the Model

```python
history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1
)
```

### Training Settings:
- epochs = 15: Allows sufficient learning
- batch_size = 128: Balances speed and stability
- validation_split = 0.1: Monitors generalization

### Evaluating the Results
- After training, plot:
- Training loss
- Validation loss
### Observations
- ✅ Both curves decrease at a similar rate
- ✅ Validation loss no longer diverges
- ✅ Model generalizes better

- This indicates that L1 regularization is effective.
- Why L1 Regularization Works
- Penalizes large weights
- Forces the network to rely on only the most important features
- Simplifies the model representation
- Reduces sensitivity to noise in the training data

### Additional Learning Points
- L1 vs L2 Regularization

| Aspect             | L1 (Lasso)            | L2 (Ridge)           |
|--------------------|-----------------------|----------------------|
| Weight behavior    | Many weights → 0      | Small but non-zero   |
| Feature selection  | ✅ Yes                | ❌ No                |
| Sparsity           | High                  | Low                  |
| Stability          | Less stable           | More stable          |


### Best Practices
- ⚖️ Tune the regularization parameter (λ) carefully
- 🧪 Combine with cross-validation
- 🔀 Often used together with Dropout or Early Stopping
- 🧠 Useful when dataset has many irrelevant features

### When to Use L1 Regularization
- ✅ High-dimensional data
- ✅ Need interpretability
- ✅ Feature selection is important
- ✅ Strong signs of overfitting
- 🚫 Avoid if:
      - Dataset is very small
      - Important features should not be eliminated

### Key Takeaways
- L1 regularization helps prevent overfitting by enforcing sparsity
- Easy to implement using kernel_regularizer=l1()
- Improves generalization and model robustness
- Useful as part of a broader regularization strategy
  

  
