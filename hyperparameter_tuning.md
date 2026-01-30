# 📘 Parameters vs. Hyperparameters — Study Notes

## Overview
In machine learning and deep learning, **parameters** and **hyperparameters** play crucial yet distinct roles in how models are constructed, trained, and optimized.

---

## 🧠 Model Parameters

### What Are Parameters?
- **Parameters are the internal variables learned directly from training data.**
- They are **not set manually**.
- The learning algorithm updates them to **minimize the loss function**, which measures the difference between predictions and actual values.

### Key Traits
- Learned during training  
- Adjusted automatically via optimization (e.g., gradient descent)  
- Define how the model represents patterns in data  

---

## 📊 Example: Linear Regression

A simple house-price model:

\[
Y = W \cdot X + B
\]

- **W (weight)**: slope  
- **B (bias)**: y-intercept  
- **Parameters**: `W` and `B`  
- These are adjusted during training to minimize prediction error.

---

## 🤖 Example: Neural Network for Handwritten Digit Recognition

### Network Architecture
- **Input Layer**: 784 neurons (28×28 pixel image)  
- **Hidden Layer 1**: 512 neurons  
- **Hidden Layer 2**: 128 neurons  
- **Output Layer**: 10 neurons (digit classes 0–9)

### Parameters in the Network
#### Weights
- Input → Hidden1: `784 × 512`  
- Hidden1 → Hidden2: `512 × 128`  
- Hidden2 → Output: `128 × 10`

#### Biases
- Hidden Layer 1: 512 biases  
- Hidden Layer 2: 128 biases  
- Output Layer: 10 biases  

All weights and biases = **parameters learned during training** via optimization algorithms such as **stochastic gradient descent**.

---

## ⚙️ Hyperparameters

### What Are Hyperparameters?
Hyperparameters are **external configuration choices** set *before* training begins.

They:
- Govern the **behavior of the training algorithm**
- Define the **model architecture**
- Are **not learned from data**
- Require experimentation and tuning

### House-building Analogy
- **Parameters = construction materials** (bricks, wood, cement)  
- **Hyperparameters = architectural blueprint** (number of rooms, layout, design choices)

---

## 🌟 Examples of Hyperparameters

### 1. **Learning Rate**
- Controls the size of weight updates during training  
- Too high → unstable  
- Too low → slow learning  

### 2. **Batch Size**
- How many samples are used per gradient update  

### 3. **Number of Epochs**
- Number of times the full dataset is passed through the model during training  

---

## 📝 Summary

| Concept | Parameters | Hyperparameters |
|--------|------------|-----------------|
| Learned during training? | ✔️ Yes | ❌ No |
| Set manually? | ❌ No | ✔️ Yes |
| Examples | Weights, biases | Learning rate, batch size, epochs |
| Role | Define model behavior | Define training process & model design |

---

These notes provide a clear differentiation between parameters and hyperparameters, along with examples to deepen understanding.


# 📘 Key Hyperparameters in Deep Learning — Study Notes

Hyperparameters are **external settings** that define how a neural network is **structured** and **trained**.  
Unlike model parameters (weights and biases), **hyperparameters are set before training** and have a major influence on learning efficiency, convergence, and generalization.

---

## 🔑 Why Hyperparameters Matter
- They shape the **learning process** and **model architecture**.
- Good hyperparameter choices lead to:
  - Efficient training  
  - Stable optimization  
  - Better generalization to unseen data  
- Poorly chosen hyperparameters can cause:
  - Overfitting  
  - Underfitting  
  - Slow or unstable convergence  

---

# 1️⃣ Learning Rate

### What It Does
The **learning rate** controls how big a step the model takes when updating weights during each iteration.

### Effects of Poor Choice
- **Too high** → overshooting, divergence, unstable training  
- **Too low** → slow learning, risk of getting stuck in suboptimal minima  

### Best Practices
- A common baseline: **0.001** (especially with Adam).  
- Use **learning rate schedulers**:
  - Step decay  
  - Cosine annealing  
  - Warmup schedules  
- Perform a quick scan of values between **0.0001 to 0.1** to find a good range.

---

# 2️⃣ Batch Size

### What It Does
Batch size = number of samples processed before updating model parameters.

### Trade-offs
- **Smaller batch sizes**:
  - Noisier gradients → can escape shallow minima  
  - Faster initial convergence  
  - Useful for limited GPU memory  
- **Larger batch sizes**:
  - More stable gradients  
  - Faster computation on modern GPUs  
  - Might require a higher learning rate  

### Best Practices
- Start between **32 and 256**.  
- Adjust based on:
  - Hardware memory  
  - Model stability  
  - Learning rate behavior  

---

# 3️⃣ Number of Epochs

### What It Does
Epochs = number of full passes through the training dataset.

### Pitfalls
- **Too few** → underfitting  
- **Too many** → overfitting  

### Best Practices
- Typical starting range: **10 to 50 epochs**  
- Use **early stopping** based on:
  - Validation loss  
  - Validation accuracy  
- Stop training when validation performance stops improving.

---

# 4️⃣ Model Architecture Hyperparameters

These define the **design of the neural network**, such as:
- Number of layers  
- Number of neurons per layer  

### Best Practices
- Start simple → shallow model  
- If underfitting → increase depth or width  
- For well-studied tasks, start with known architectures:
  - **ResNet** (image classification)  
  - **BERT variants** (NLP)  

### Regularization for Stability
As models grow:
- Add **dropout**  
- Use **weight decay (L2 regularization)**  
- Add **batch normalization**  

---

# 5️⃣ Dropout Rate

### What It Does
Dropout randomly “turns off” a fraction of neurons during training.  
Prevents over-reliance on specific nodes → improves generalization.

### Typical Range
- **0.1 to 0.5**, depending on:
  - Layer type  
  - Model complexity  

### Behavior
- If model underfits → **reduce** dropout  
- If model overfits → **increase** dropout  

---

# 6️⃣ L1 / L2 Regularization Coefficients

Help prevent overfitting by penalizing large weights.

### Best Practices
- Start with **small** coefficients  
- Increase only if signs of overfitting appear  
- Decrease if underfitting

---

# 7️⃣ Weight Initialization Method

Initialization determines how weights start before training.

### Why It Matters
Good initialization:
- Keeps gradients stable  
- Ensures signals propagate properly  
Poor initialization:
- Causes vanishing or exploding gradients  

### Recommended Methods
- **Xavier Initialization**  
- **He Initialization**  

These are widely used for deep networks.

---

# 8️⃣ Optimizer Choice

Controls how gradients update weights during backpropagation.

### Common Optimizers
- **SGD (Stochastic Gradient Descent)**
  - Simple, effective  
  - Often improved with **momentum**  
- **Adam**
  - Adaptive learning rate per parameter  
  - Robust across many tasks  
  - Good default choice  

### Best Practices
- Start with proven defaults (Adam or SGD+momentum).  
- Only adjust optimizer settings if:
  - Training divergence occurs  
  - Convergence is unusually slow  

---

# 🎯 Hyperparameter Tuning Strategy

### Key Guidelines
- **Start with known, reliable baselines**  
- Adjust **one hyperparameter at a time**  
- Monitor:
  - Validation loss  
  - Validation accuracy  
- Iterate gradually to develop intuition over time.

Hyperparameter tuning is a **balancing act**, but with systematic testing, patterns and best practices become clearer.

---
# 📘 Methods for Hyperparameter Tuning — Study Notes

Hyperparameter tuning is the process of finding the **best hyperparameter settings** to optimize a deep learning model’s training performance, convergence, and generalization. Unlike model parameters (weights, biases), **hyperparameters are set before training** and are not learned automatically.

Because the hyperparameter search space can be large and complex, several tuning strategies are commonly used.

---

# 1️⃣ Grid Search

## 🔍 What It Is
Grid search evaluates **every possible combination** of predefined hyperparameter values.

Example:
- Learning rates tested: 0.00001 → 0.1 (5 values)  
- Batch sizes tested: 32 → 512 (5 values)  
- Total combinations = **5 × 5 = 25 models**

## ✅ Advantages
- **Exhaustive and thorough** — guarantees testing every point in the grid  
- **Simple, intuitive, and widely supported**  
- Ensures no defined configuration is missed  

## ❌ Disadvantages
- **Computationally expensive** — grows exponentially with more hyperparameters  
- **Rigid** — must evaluate *every* combination, even poor candidates  
- Not ideal for large or continuous search spaces  

---

# 2️⃣ Random Search

## 🔍 What It Is
Instead of testing all combinations, random search **samples random values** from specified ranges.

Example:
- Learning rate randomly chosen from: 0.0001 → 0.1  
- Batch size randomly chosen from: 32 → 512  

## ✅ Advantages
- **More efficient** than grid search  
- Quickly finds good configurations  
- Scales well — add more trials as needed  
- Focuses on potentially useful regions of the search space  

## ❌ Disadvantages
- No guarantee of exploring all important regions  
- Might miss high‑performing configurations  
- Less systematic coverage than grid search  

---

# 3️⃣ Bayesian Optimization

## 🔍 What It Is
Bayesian optimization builds a **surrogate model** to predict how hyperparameter settings affect performance.

After each evaluation, it updates its internal model to choose the next hyperparameters by balancing:
- **Exploration** (try new regions)
- **Exploitation** (refine promising regions)

Think of it like having a “smart assistant” that gets better at guessing where good hyperparameters are.

## ✅ Advantages
- **Intelligent and efficient** search  
- Requires **fewer evaluations** than grid or random search  
- Learns from previous trials  
- Great for expensive or complex models  

## ❌ Disadvantages
- **More complex** to implement  
- Requires maintaining a sophisticated surrogate model  
- Computational overhead for the optimization process itself  

---

# 🧠 Choosing the Right Method

### ✔️ Use **Grid Search** when:
- The search space is **small**
- Hyperparameters have only a few discrete options
- You need complete coverage

### ✔️ Use **Random Search** when:
- The search space is **large**
- Hyperparameters vary over continuous ranges
- You want faster, more flexible testing

### ✔️ Use **Bayesian Optimization** when:
- Training models is **expensive**
- You need efficient exploration  
- The search space is complex and multi-dimensional  

---

# 🎯 Best Practices for Hyperparameter Tuning
- Start with **baseline or default** values  
- Tune **one hyperparameter at a time** initially  
- Use **domain knowledge** when narrowing ranges  
- Use **validation metrics** to guide decisions  
- Iterate gradually — hyperparameter tuning is a **process of refinement**

---

Hyperparameter tuning ensures that a model reaches its **best possible performance** without unnecessary computation. By carefully selecting tuning methods and combining them with practical experience, you can efficiently explore the search space and achieve optimal results.

# 📘 Defining a Tunable Deep Learning Model in Keras — Study Notes

In this lesson, we learn how to define a **tunable** deep learning model in Keras, preparing it for hyperparameter tuning using **Keras Tuner**. This includes creating a function that acts as the *model blueprint*, specifying which hyperparameters should be searched and how they influence model structure and training.

---

## 🔧 Prerequisites
Before defining the tunable model:
- Import and preprocess data  
- Build and train a baseline model  
- Import necessary layers and optimizers  

These steps should already be completed earlier in the notebook.

---

# 1️⃣ Import Required Components

To support hyperparameter tuning, we import:

- **Dropout** layer  
- **Adam** optimizer

2️⃣ Defining the Tunable Model Function

Keras Tuner repeatedly calls this function, each time with different hyperparameter values.
Function Name: build_model(hp)

hp is a hyperparameter object that defines which hyperparameters to search.
The function returns a compiled model ready for evaluation.

3️⃣ Model Architecture with Hyperparameters
a. Model Initialization
model = keras.Sequential()
b. Input Layer

Input shape: 784 (flattened 28×28 image)

Hidden Layer 1 (Tunable)
Use hp.Int to tune the number of neurons:
hp.Int("hidden1", min_value=32, max_value=512, step=32)
Dropout Layer 1 (Tunable)
Use hp.Float to tune dropout rate:
hp.Float("dropout1", min_value=0.1, max_value=0.5, step=0.1)
Hidden Layer 2 (Tunable)
Number of neurons chosen with:
hp.Int("hidden2", min_value=16, max_value=128, step=16)

 Dropout Layer 2 (Tunable)
Same dropout range as earlier:
hp.Float("dropout2", min_value=0.1, max_value=0.5, step=0.1)
4️⃣ Output Layer (Not Tuned)
Output layer has a fixed number of units:

10 units for digit classification (0–9)
Dense(10, activation="softmax")

5️⃣ Learning Rate (Tunable with hp.Choice)
We evaluate a set of discrete learning rates:

hp.Choice("learning_rate", values=[0.0001, 0.001, 0.01])

6️⃣ Compiling the Model
The optimizer uses the selected learning rate.

  model.compile(
    optimizer=Adam(learning_rate=hp_learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

 Purpose of the Tunable Function
The function:

Acts as the architectural blueprint
Defines all hyperparameters to explore:

Hidden layer sizes
Dropout rates
Learning rate


Is repeatedly invoked during the hyperparameter search

This allows Keras Tuner to systematically explore the hyperparameter space and find the configuration that maximizes model performance.

Please see the code : ./run_hparam_tuning.py
