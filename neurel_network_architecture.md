# Input Layer in Artificial Neural Networks (ANN)

## 1. Vectors as Input
- A **vector** is an ordered list of numeric values.
- In deep learning, input data is usually represented as a vector (often using NumPy arrays).
- Vectors represent **features** (independent variables) used for training and prediction.

## 2. Samples and Features
- **Sample** = One instance of real-world data (like a record in a database).
- **Features** = Attributes of a sample (e.g., age, salary, service).

### Examples:
- Employee dataset → Each employee = sample; age, salary = features.
- Text → Each document = sample; numeric representation = features.
- Image → Each image = sample; pixel values = features.
- Speech → Represented as a time series of numbers.

## 3. Preprocessing Requirements
- Input data must be **numeric** before feeding into a neural network.
- Common preprocessing techniques:
  - **Normalization**: Center and scale values to standard ranges.
  - **Categorical Encoding**: Integer encoding or One-Hot encoding.
  - **Text Representation**:
    - TF-IDF (Term Frequency-Inverse Document Frequency).
    - **Embeddings** (popular in deep learning).
  - **Images**: Represented as pixel value vectors.
  - **Speech**: Converted into numeric time-series.

## 4. Example: Employee Data
- Features: Age, Salary, Service → Represented as `x1`, `x2`, `x3`.
- Normalize values (center and scale).
- Optionally transpose so each sample is a column.

## 5. Final Step
- Once preprocessed, the data is ready to be passed into the **input layer** of the neural network.

# Hidden Layers in Artificial Neural Networks (ANN)

## 1. Role of Hidden Layers
- Hidden layers form the **brain** of a neural network where knowledge is acquired and used.
- An ANN can have **one or more hidden layers**.
- More layers → deeper network.

## 2. Structure and Node Configuration
- Each hidden layer can have **one or more nodes**.
- Node count is typically configured in powers of 2:
  - Examples: 8, 16, 32, 64, 128, etc.
- The architecture of a neural network is defined by:
  - Number of layers.
  - Number of nodes in each layer.

## 3. Connectivity Between Layers
- Output of each node in the **previous layer** becomes input for **every node** in the current layer.
- Similarly, output of each node in the current layer is passed to every node in the next layer.

### Example:
- First hidden layer: 4 nodes → 4 outputs from activation functions.
- Second hidden layer: 5 nodes → Each node receives all 4 outputs from the previous layer.

## 4. Determining the Right Architecture
- Each node learns relationships between **feature variables** and **target variables**.
- Knowledge is stored in **weights and biases**.
- More nodes and layers often improve accuracy, but:
  - Not always true.
  - More layers = more compute resources and longer training/inference time.

## 5. Best Practices
- Start with **small numbers** of layers and nodes.
- Gradually increase until acceptable accuracy is achieved.
- Final architecture is determined by **experimentation**.


# Weights and Biases in Neural Networks

## Overview
Weights and biases form the foundation for deep learning algorithms. They are **trainable parameters** in a neural network, adjusted during training to provide accurate predictions.

- **Weights**: Numeric values associated with each input for each node.
- **Bias**: A single numeric value associated with each node.

At the layer level, weights and biases are handled as **arrays (matrices)**.

## Structure in Neural Networks
- Each input for a node has an associated **weight**.
- Each node has **one bias value**.
- For example:
  - First hidden layer: 3 inputs, 4 nodes
    - Weights: \(3 \times 4 = 12\)
    - Biases: \(4\)
- Total for the network:
  - **Weights**: 53
  - **Biases**: 14
  - **Total Parameters**: 67

## Computation Details
- Training computes hidden layers together for optimization.
- Weights and biases for each layer are maintained as **matrices**.
- Input and output are also represented as matrices.

### Example: Hidden Layer 2
- Inputs: 4
- Nodes: 5
- Input matrix: \(1 \times 4\)
- Weight matrix: \(4 \times 5\)
- Bias matrix: \(1 \times 5\)
- Output:
  - Matrix multiplication: \( (1 \times 4) \cdot (4 \times 5) = (1 \times 5) \)
  - Add bias: \( (1 \times 5) + (1 \times 5) = (1 \times 5) \)
The output matrix is then passed to the next layer.

## Key Takeaways
- Weights and biases are essential for neural network learning.
- Computations rely heavily on **matrix operations**.
- Deep learning frameworks handle these computations internally.

**Recommended Reading**: Matrix multiplication for deeper understanding.

# Activation Functions in Neural Networks

## Overview
An **activation function** plays a crucial role in determining the output of a node in a neural network. It takes the matrix output of the node and decides **if and how the node will propagate its information to the next layer**.

### Key Roles:
- Acts as a **filter** to reduce noise.
- **Normalizes output**, which can become large due to matrix multiplications.
- Converts output to a **nonlinear value**, enabling the network to learn complex patterns.


## Why Activation Functions Matter
- They help neural networks learn **specific patterns** in data.
- Implemented as part of the model configuration in deep learning libraries.
- Output dimensions remain the same as input dimensions.


## Popular Activation Functions
### 1. **Sigmoid**
- Output range: **0 to 1**
- Interpretation:
  - Value near 0 → Node does not pass its learnings forward.
- Common in binary classification.

### 2. **Tanh**
- Output range: **-1 to +1**
- Normalizes output around zero.

### 3. **ReLU (Rectified Linear Unit)**
- If input < 0 → Output = 0
- Else → Output = input
- Widely used for hidden layers due to simplicity and efficiency.

### 4. **Softmax**
- Used in **classification problems**.
- Produces a **vector of probabilities** for each class.
- Sum of probabilities = **1**.
- Class with highest probability is chosen for prediction.


## Key Takeaways
- Activation functions introduce **nonlinearity**, essential for deep learning.
- Each function has **advantages, shortcomings, and specific applications**.
- Deep learning frameworks handle implementation; you just specify them as **hyperparameters**.


# The Output Layer in Neural Networks

## Overview
The **output layer** is the final layer in a neural network where desired predictions are obtained. It produces the **final prediction** after applying its own set of **weights and biases**.

## Key Characteristics
- There is **only one output layer** in a neural network.
- The activation function for the output layer may differ from hidden layers depending on the problem.
  - Example: **Softmax** activation is commonly used for classification problems.

## Output Details
- The output is a **vector of values** that may require **post-processing** to map them to business-related values.
  - Example: In classification, the output is a set of probabilities mapped to corresponding classes.

## Determining Number of Nodes
- Depends on the type of problem:
  - **Binary Classification**: 1 node (probability of positive outcome).
  - **Multi-class Classification (n classes)**: n nodes (each node gives probability for one class).
  - **Regression**: 1 node (produces the numeric output).

## Key Takeaways
- The output layer completes the neural network structure.
- Activation function choice is **problem-specific**.
- Post-processing may be required to convert raw outputs into meaningful predictions.
## Overall Takeaway
- In a dataset, each row represents one observation or record, and in machine learning terminology, this is called a sample (or sometimes an instance).
- In the input dataset, each record or row becomes a sample.
- Each column represents a feature (or attribute) of that sample.
- A node in a neural network always has one bias value, regardless of how many inputs it receives.
- Each node in a hidden layer receives the output of all nodes from the previous layer, not the next layer.
- Each node in a hidden layer receives the outputs of all nodes from the previous layer.
- Neural networks are feed-forward (in most cases), meaning data flows from input → hidden layers → output layer.

# Node Output Formula in Layer L

The output of a node in layer **L** is computed as:

\[
\text{Output} = f\Bigg( \sum_{i=1}^{n} w_i \cdot a_i^{(L-1)} + b \Bigg)
\]

Where:
- \( f \) = Activation function
- \( w_i \) = Weight for input \( i \)
- \( a_i^{(L-1)} \) = Output from node \( i \) in the previous layer
- \( b \) = Bias term for the node
- \( n \) = Number of inputs to the node
- The activation function most commonly used in the output layer for classification problems is:  Softmax

- Why Softmax?
- Purpose: Converts raw scores (logits) into probabilities for each class.
- Output: A vector where:
- Each element represents the probability of a class.
- All probabilities sum to 1.
- Use Case: Multi-class classification problems.
- Example: Predicting whether an image is a cat, dog, or bird
- Other case:
- Binary classification: Often uses Sigmoid in the output layer because it outputs a value between 0 and 1 (interpreted as probability of positive class).
- Multi-class classification: Uses Softmax because it handles multiple classes and normalizes outputs into probabilities.
- 
