# AI Workshop: Build a Neural Network with PyTorch Lightning

## 📌 Overview
This workshop focuses on building neural networks using **PyTorch** and **PyTorch Lightning**. It emphasizes writing **clean, modular, and reusable code** while leveraging Lightning’s high-level abstractions for training.

---

## ✅ What is PyTorch?
- **Definition:** An optimized tensor library for deep learning using GPUs and CPUs.
- **Key Features:**
  - Dynamic computation graphs for flexibility.
  - Integrated with **NumPy** for easy data conversion.
  - Native GPU support for accelerated training.
  - **Autograd** for automatic differentiation during backpropagation.
- **Use Case:** Ideal for building deep learning models with granular control.

---

## ✅ Why PyTorch Lightning?
- **Purpose:** A lightweight wrapper on PyTorch that simplifies training and reduces boilerplate code.
- **Advantages:**
  - Cleaner, modular code.
  - Built-in support for **experiment logging** and **distributed training**.
  - Callback system for custom functionality.
  - Interoperable with PyTorch for flexibility.
- **Trainer Class:** Abstracts away complex training loops.

---

## 🔍 Prerequisites
- Basic understanding of **machine learning** (regression, classification).
- Familiarity with **neural networks**.
- Comfortable with **Python programming**.

---

## 🧠 Quick Neural Network Overview
- **Architecture:** Layers connected in a directed acyclic graph.
- **Components:**
  - **Input Layer:** Accepts data.
  - **Hidden Layers:** Transform data and extract features.
  - **Output Layer:** Produces predictions.
- **Neuron Function:**  
  - Computes `Wx + b` (linear relationships).
  - Applies **activation function** (non-linear relationships).
- **Activation Functions:** ReLU, Sigmoid, etc.
- **Trainable Parameters:** Weights and biases updated during training.

---

## 🔄 Training Process
1. Feed **batches of data** through the network.
2. Compute predictions and **loss**.
3. Calculate **gradients** (partial derivatives of loss w.r.t parameters).
4. Perform **backpropagation** to update weights.
5. Repeat until convergence using **gradient descent**.

---

## ⚡ Why PyTorch Lightning for Training?
- Reduces repetitive code.
- Encourages modular design for better reproducibility.
- Simplifies **distributed training** and device management.
- Provides **callbacks** and **logging** for experiments.

---

### ✅ Key Takeaways
- PyTorch gives flexibility and control.
- PyTorch Lightning accelerates development with cleaner abstractions.
- Both can be combined for maximum efficiency.

---

## 📚 References
- [PyTorch Documentation- PyTorch Lightning

---
