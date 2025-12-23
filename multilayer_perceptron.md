# Multilayer Perceptron (MLP) 

## 1. What is a Multilayer Perceptron?
A **Multilayer Perceptron (MLP)** is a type of **artificial neural network** that consists of multiple layers of nodes (also called neurons). It is widely used for tasks like **classification** and **regression**.

### Key Layers:
- **Input Layer**: Receives the raw data.
- **Hidden Layers**: Perform computations and extract patterns.
- **Output Layer**: Produces the final prediction or classification.

---

## 2. Structure
- **Shallow Network**: Has one hidden layer.
- **Deep Network**: Has multiple hidden layers.

Each neuron (except in the input layer) applies a **non-linear activation function** to its input.

---

## 3. Characteristics of MLP
- **Feed-forward architecture**: Data flows in one direction (input → hidden → output).
- **Fully connected**: Each neuron in one layer connects to every neuron in the next layer.
- **Learns complex patterns**: Can model both **linear** and **non-linear** relationships.

---

## 4. Why Use MLP?
- Handles **large and complex datasets**.
- Works for both:
  - **Regression** (e.g., predicting house prices).
  - **Classification** (e.g., determining if a house has a garage).

---

## 5. Activation Functions
Activation functions introduce **non-linearity**, enabling the network to learn complex patterns. Common examples:
- Sigmoid
- ReLU
- Tanh

---

## 6. Learning Process
- MLP uses **backpropagation** and **gradient descent** to adjust weights.
- It does **not require labeled data** for feature extraction, but supervised learning is common for prediction tasks.

---

## Summary
- **MLP = Input Layer + Hidden Layers + Output Layer**
- **Feed-forward, fully connected**
- Learns **linear and non-linear relationships**
- Suitable for **classification and regression**

---

