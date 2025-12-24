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
# Neural Network Layers: Input, Hidden, and Output

## 1. What Are Neural Network Layers?
Neural network layers allow the network to solve **complex non-linear problems**. Layers are composed of **nodes (neurons)** stacked vertically and connected from input to output.

---

## 2. Types of Layers
### **Input Layer**
- The first layer in the network.
- Receives raw data or features.
- **Passive**: Performs no computation, only passes data to the next layer.

### **Hidden Layer**
- Located between input and output layers.
- Performs computations using **weighted inputs** and **activation functions**.
- You can add multiple hidden layers to handle complex data (deep networks).

### **Output Layer**
- The final layer that produces predictions or classifications.
- May have one or multiple nodes depending on the task.

---

## 3. Key Characteristics
- **Fully Connected**: Each node in one layer connects to every node in the next layer.
- **Feedforward**: Data flows in one direction (input → hidden → output).
- **No loops**: Unlike recurrent networks, feedforward networks have no cycles.

---

## 4. Workflow
1. **Input Layer**: Accepts features.
2. **Hidden Layers**: Apply weights and activation functions to learn patterns.
3. **Output Layer**: Produces the final result (prediction or classification).

---


# How Neural Networks Learn

## 🧠 Overview
Neural networks learn by identifying patterns in data and adjusting internal parameters (weights and biases) to minimize prediction errors. This process is iterative and relies on mathematical optimization.

---

## ✅ Example Use Case
**Predicting house prices**  
- **Type of problem:** Regression (predicting continuous values, not categories)
- **Goal:** Estimate house price based on features like size, location, etc.

---

## 🔍 Key Concepts
- **Predicted Value (Y_hat):** Output from the network
- **Actual Value (Y):** Ground truth from dataset
- **Transfer Function:** Weighted sum of inputs passed through an activation function
- **Batch Size:** Number of samples processed in one forward pass

---

## 📐 Learning Process
### 1. Forward Pass
- Input data flows through layers → produces predictions (Y_hat)
- Compare predictions to actual values → compute error

### 2. Loss Function
For regression, commonly use **Root Mean Squared Error (RMSE):**

``
RMSE = sqrt( (1/n) * Σ (Y - Y_hat)^2 )

**Steps to compute RMSE:**
1. Calculate errors for each prediction: `error = Y - Y_hat`
2. Square each error
3. Compute the mean of squared errors
4. Take the square root of the mean

---

### 3. Backpropagation
- Compute gradients of error with respect to weights
- Adjust weights to reduce error
- Repeat until convergence (error cannot decrease further)

---

## 🔄 Intuition
Think of it as a feedback loop:
- Forward pass predicts → backward pass corrects
- Like a game of tennis: back and forth until the model stabilizes

---

## 🏁 Convergence
When further weight updates no longer reduce error significantly, the network has **learned** the optimal parameters for the given data.

---

### ✅ Summary
Neural networks learn by:
- Passing data forward
- Measuring error
- Adjusting weights backward
- Iterating until the cost function is minimized

---


**Tip:** Always visualize the process with a graph of predicted vs actual values to understand how close your model is to the ideal line of best fit.
- I have detailed this as python code in the chapter [multilayer_perceptron.py](./multilayer_perceptron.py)



# Transfer and Activation Functions in Neural Networks

## 🧠 Overview
A neuron in a neural network performs two key operations:
1. **Transfer (Weighted Sum):** Combines inputs and weights into a single value.
2. **Activation:** Applies a non-linear function to the weighted sum to introduce non-linearity.

Although often shown as separate steps, both occur inside a single computational node.

---

## ✅ Components of a Node
- **Inputs:** Features from the dataset (e.g., zip code, ocean proximity).
- **Weights (w1, w2, ...):** Learnable parameters that scale inputs.
- **Bias (b):** A constant added to the weighted sum to shift the activation function left or right.
- **Transfer Function:** Computes:
- z = (x1 * w1) + (x2 * w2) + ... + b
- - **Activation Function:** Applies a non-linear function `f(z)` to produce the node’s output.

---

## 🔍 Why Activation Functions?
- Real-world data is often **non-linear**.
- Without activation functions, the network would only learn **linear relationships**.
- Activation functions allow the network to learn **complex decision boundaries**.

---

## 📐 Common Activation Functions
### 1. **ReLU (Rectified Linear Unit)**
f(x) = max(0, x)
- Outputs zero for negative inputs, linear for positive inputs.
- Popular for deep networks due to efficiency and reduced vanishing gradient issues.

### 2. **Sigmoid**
f(x) = 1 / (1 + e^(-x))

- Outputs values between 0 and 1.
- Often used for probabilities in binary classification.

### 3. **Tanh**
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

- Outputs values between -1 and 1.
- Good for centered data; often used in recurrent networks.

---

## ✅ Summary
- **Transfer function:** Weighted sum + bias.
- **Activation function:** Adds non-linearity.
- **Why important?** Enables learning of complex patterns beyond linear relationships.

---

**Tip:** Choose activation functions based on the task:
- ReLU for hidden layers in deep networks.
- Sigmoid for binary outputs.
- Tanh for outputs needing negative and positive ranges.

# How Neural Networks Learn

## Overview
Neural networks learn by identifying patterns in data and adjusting internal parameters (weights and biases) to minimize prediction errors. This process is iterative and relies on concepts like **forward pass**, **loss calculation**, and **backpropagation**.

---

## Example Use Case
**Predicting house prices**  
- **Type of problem**: Regression (predicting a continuous value, not classifying into categories).

---

## Key Concepts
### 1. **Predicted Value (`Y_hat`)**
- Output of the network after a forward pass.
- Compared against the actual value (`Y`) from the dataset.

### 2. **Weighted Sum**
`z = sum(x_i * w_i) + b`
- Combines inputs (\(x_i\)) with weights (\(w_i\)) and bias (\(b\)).

### 3. **Activation Function**
`Y_hat = f(z)`
- Applies non-linearity to the weighted sum.

---

## Learning Process
1. **Forward Pass**
   - Input data flows through the network.
   - Produces predictions (`Y_hat`).

2. **Error Calculation**
   - Difference between actual and predicted values.
   - For regression, a common metric is **Root Mean Squared Error (RMSE)**:
RMSE = sqrt( (1/n) * sum( (Y_i - Y_hat_i)^2 ) )

3. **Backpropagation**
   - Uses calculus to compute gradients.
   - Adjusts weights and biases to reduce error.
   - Iterates until convergence (error cannot decrease further).

---

## Batch Size
- Number of training examples processed in one forward pass.
- Larger batches = faster computation but higher memory usage.

---

## Why Backpropagation Matters
- Enables the network to learn by **minimizing the loss function**.
- Updates weights in the opposite direction of the gradient (gradient descent).

---

## Convergence
- The point where further training does not significantly reduce error.
- Indicates the model has learned the best possible weights for the given data.

---

### Summary
Neural networks learn by:
- Passing data forward through layers.
- Measuring prediction error.
- Adjusting weights using backpropagation until the error is minimized.

---
