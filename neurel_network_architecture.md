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

