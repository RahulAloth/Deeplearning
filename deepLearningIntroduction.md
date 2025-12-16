## Getting Started with Deep Learning

Deep learning is a key technology driving AI and machine learning innovations.
The course introduces basic concepts with simple explanations and examples.
Focus areas:

* Steps to build deep learning models.
* Using Keras and TensorFlow for implementation.
* Hands-on exercises for skill assessment

## Prerequisites

Familiarity with:
* Machine learning concepts and technologies.
* Python programming and Jupyter Notebooks.

## What is Deep Learning?
* Definition: A subfield of machine learning focused on building and using neural network models.
* Structure: Neural networks with more than three layers are typically considered deep learning networks.
* Inspiration: Neural networks mimic the human brain, organized like brain cells, and imitate how humans process data and make decisions.
Growth Factors:
* Algorithms have existed for years, but recent advances in:
  * Large-scale data processing
  * Inference technologies (e.g., GPUs)
These advances have driven real-world adoption.

## Popular Applications:
* Natural Language Processing (NLP) – ideal for unstructured data.
* Speech recognition and synthesis.
* Image recognition.
* Self-driving cars – cutting-edge technology powered by deep learning.
  
Other Domains:
* Customer experience
* Healthcare
* Robotics

## Linear Regression
* Definition: A basic statistical concept used in machine learning and forms a foundation for deep learning.
* Purpose: Explains the relationship between two or more variables.
* Components:
  * Dependent variable: y
  * Independent variable(s): x (or x₁, x₂, …, xₙ for multiple variables)    
* Equation:
    y=a⋅x+b
    * a = slope
    * b = intercept
* Nature: Provides a linear relationship between y and x.
* Reality: Predictions may have errors because real-world relationships are not perfectly linear.
* Use Case: Regression problems to predict continuous variables.
* Multiple Variables:
    * y=a1x1+a2x2+⋯+anxn+by = a_1 x_1 + a_2 x_2 + \dots + a_n x_n + by=a1​x1​+a2​x2​+⋯+an​xn​+b

## Building a Linear Regression Model
* Goal: Determine slope (a) and intercept (b) using known values of x and y.
* For one variable: Use two equations and substitution to find a and b.
* For multiple variables: Becomes more complex.

## Logistic Regression (Related Technique)

* Used for binary classification (output y = 0 or 1).
* Formula similar to linear regression but includes an activation function f:
* y=f(a⋅x+b)y = f(a \cdot x + b)y=f(a⋅x+b)
* Converts continuous output into a boolean (0 or 1).
* Can also extend to multiple independent variables.

# Deep Learning Basics

- **Definition**  
  - A subset of machine learning focused on **neural networks** with multiple layers (typically more than 3).
  - Designed to mimic the **human brain’s structure and decision-making process**.

- **Core Idea**  
  - Uses **layers of neurons** to process data and learn patterns.
  - Each layer transforms input data into more abstract representations.

- **Key Components**
  - **Input Layer**: Receives raw data.
  - **Hidden Layers**: Perform computations and extract features.
  - **Output Layer**: Produces predictions or classifications.
  - **Weights & Biases**: Parameters adjusted during training.
  - **Activation Functions**: Introduce non-linearity (e.g., ReLU, Sigmoid).

- **Training Process**
  - **Forward Propagation**: Data flows through layers to produce output.
  - **Loss Function**: Measures prediction error.
  - **Backpropagation**: Adjusts weights using gradients to minimize error.
  - **Optimization**: Algorithms like **Gradient Descent** improve model performance.

- **Why Deep Learning Works Well**
  - Handles **large-scale data** efficiently.
  - Learns **complex patterns** without manual feature engineering.
  - Scales with **GPU acceleration** and big data.

- **Popular Architectures**
  - **CNN (Convolutional Neural Networks)** – for images.
  - **RNN (Recurrent Neural Networks)** – for sequences like text or speech.
  - **Transformers** – for NLP tasks.

- **Applications**
  - **Natural Language Processing (NLP)** – chatbots, translation.
  - **Computer Vision** – image recognition, self-driving cars.
  - **Speech Recognition** – voice assistants.
  - **Healthcare, Robotics, Customer Experience

# An Analogy for Deep Learning

- **Purpose of Analogy**  
  - Helps understand how deep learning works through a simple example.
  - Deep learning is an **iterative process** that uses trial and error to optimize parameters.

- **Key Idea**  
  - Starts with **random initialization** of model parameters.
  - Gradually adjusts parameters to minimize error and reach optimal values.

- **Linear Regression Example**  
  - Formula: `10 = 3A + B`
  - Goal: Find values of **A** and **B** using trial and error.
  - Steps:
    - Initialize A = 1, B = 1 → Result = 4 → Error = +6.
    - Adjust A = 4, B = 3 → Result = 15 → Error = -5.
    - Adjust A = 2, B = 2 → Result = 8 → Error = +2.
    - Adjust A = 3 → Result = 11 → Error = -1.
    - Adjust A = 2, B = 3 → Result = 9 → Error = +1.
    - Adjust B = 4 → Result = 10 → Error = 0.
  - Final values: **A = 2, B = 4**.

- **Conceptual Takeaways**  
  - Each iteration reduces error progressively.
  - Similar to **gradient descent** in deep learning.
  - Works well for small problems, but becomes complex with many variables.
  - Goal is not always zero error, but **minimizing error to an acceptable level**.

- **Connection to Deep Learning**  
  - Deep learning uses the same principle:
    - Start with random weights.
    - Use **loss function** to calculate    - Use **loss function** to calculate error.

# The Perceptron

- **Definition**
  - The perceptron is the basic **learning unit** in an artificial neural network.
  - Represents the algorithm for **supervised learning** in neural networks.
  - Resembles a **human brain cell**.

- **Function**
  - Takes **multiple inputs**, performs computations, and outputs a **boolean value** (0 or 1).
  - Represents a **single node** in a neural network.

- **Foundation**
  - Built on **logistic regression** principles.
  - Formula derived from logistic regression:
    - Replace slope (**a**) with **weight (w)**.
    - Replace intercept (**b**) with **bias (b)**.
    - Apply an **activation function (f)** to produce output.

- **Components**
  - Inputs: \( x_1, x_2, \dots, x_n \).
  - Each input multiplied by a corresponding **weight**.
  - Add a constant input (1) multiplied by **bias**.
  - Sum all weighted inputs + bias.
  - Apply **activation function** → Output \( y \) (0 or 1).

- **Role in Neural Networks**
  - Neural networks are built by **connecting multiple perceptrons**.
  - Perceptron formula

# Artificial Neural Networks (ANNs)

- **Definition**
  - An ANN is a **network of perceptrons (nodes)**.
  - Mimics the structure of the **human brain**, which is a network of cells.

- **Structure**
  - Nodes (perceptrons) are organized into **layers**:
    - **Input Layer**: One node per independent variable.
    - **Hidden Layers**: One or more layers for feature extraction.
    - **Output Layer**: Produces predictions; number of nodes depends on the task.
  - A **deep neural network** usually has **three or more layers**.
  - Each node has:
    - **Weights**
    - **Bias**
    - **Activation function**
  - Nodes in one layer connect  - Nodes in one layer connect to **all nodes in the next layer** (dense network).
  - Nodes within the same layer are **not connected** (except in advanced cases).

- **Architecture**
  - Defined by:
    - Number of layers.
    - Number of nodes in each layer.
  - Determined by **experience and experimentation**.

- **Working Process**
  - Inputs (independent variables) enter the **input layer**.
  - Data may be **pre-processed** before feeding into the network.
  - Each node applies its formula:
    - Weighted sum of inputs + bias → Activation function → Output.
  - Outputs from one layer become inputs for the next layer.

# Training an Artificial Neural Network (ANN)

- **What is Training?**
  - Training an ANN means finding the **optimal values** for:
    - **Parameters**: Weights and biases for all nodes.
    - **Hyperparameters**: Number of layers, number of nodes per layer, learning rate, etc.
  - Goal: **Maximize prediction accuracy** (sometimes trade-off with performance).

- **Inputs and Parameters**
  - Inputs, weights, and biases can be **n-dimensional arrays** (e.g., image pixels).
  - Model complexity depends on the use case.

- **Training Process**
  1. **Start with architecture** (layers and nodes) based on intuition or prior experience.
  2. **Initialize weights and biases randomly**.
  3. Use **training data** (where both inputs and outputs are known).
  4. Perform **forward propagation**:
     - Apply weights and biases to inputs.
     - Compute outputs and **error** using a loss function.
  5. **Backpropagation**:
     - Adjust weights and biases to reduce error.
     - Repeat iterations until error is minimized to an acceptable level.
  6. **Fine-tune hyperparameters** to improve speed and reduce iterations.
  7. **Save the trained model** (parameters + hyperparameters) for predictions.

- **Analogy**
  - Similar to the earlier example of trial-and-error with A and B in linear regression.
  - Here, **weights and biases** replace A and B.
  - Iterative process reduces error progressively
