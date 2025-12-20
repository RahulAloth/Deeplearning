# Deep Learning Model Training: Setup and Initialization

## 1. Overview
Before training a deep learning model, several steps are required to prepare the data and configure the model. This section covers **setup and initialization** for the training process.


## 2. Data Preparation
- **Convert samples into numeric vectors** using preprocessing techniques.
- **Transpose input vectors** (optional).
- Apply similar transformations to **target variables**.
- **Split data** into:
  - **Training set**: Used to fit parameters (weights and biases).
  - **Validation set**: Used to check accuracy and error rates, refine the model.
  - **Test set**: Used to measure final model performance.
- **Typical split**: `80% Training | 10% Validation | 10% Test`.

## 3. Model Initialization
- **Parameters and Hyperparameters**:
  - Number of layers.
  - Number of nodes per layer.
  - Activation functions for each layer.
- **Hyperparameters**:
  - Epochs.
  - Batch size.
  - Error (loss) function.
- **Selection criteria**:
  - Intuition and experience.
  - Best practices and references.
  - Suitability for the specific problem.
- **Refinement**:
  - Adjust parameters if results are not acceptable.
  - Retrain the model after adjustments.

## 4. Weight and Bias Initialization
- **Purpose**: Start with initial values; neural network learns optimal values during training.
- **Techniques**:
  - **Zero Initialization**: All weights and biases set to zero (not preferred).
  - **Random Initialization**: Preferred method.
    - Values drawn from a **standard normal distribution**:
      - Mean = 0
      - Standard deviation = 1.

## 5. Next Steps
Once setup and initialization are complete, the model is ready for **training**.

# Forward Propagation in Neural Networks

## 1. Overview
Forward propagation is the process of passing input data through the neural network to generate predictions. It is the same as performing an actual prediction using the trained model.


## 2. Pre-requisites
- Input data is organized as **samples and features**.
- Data is **split into training, validation, and test sets**.
- For the **training set**:
  - Each sample has:
    - **Target value (y)**: Actual value in the training set.
    - **Predicted value (ŷ)**: Value predicted by the network during forward propagation.


## 3. Steps in Forward Propagation
1. **Send inputs through the neural network**:
   - For each sample, inputs are fed into the network layer by layer.
2. **Compute outputs for each node**:
   - Use the **perceptron formula**:
     $$ z = \sum (w_i \cdot x_i) + b $$
     $$ a = \text{activation}(z) $$
3. **Pass outputs to the next layer** until the final layer.
4. **Obtain predicted value (ŷ)** at the output layer.
5. Repeat for **all samples in the training set**.


## 4. Collect Predictions
- Gather all predicted values (ŷ) for the training samples.
- Compare **ŷ vs y** to compute **error rates** (covered in the next step of training).


## 5. Key Points
- Forward propagation is **essential for prediction and error calculation**.
- It is repeated for every sample in the training dataset.

# Measuring Accuracy and Error in Neural Networks

## 1. Overview
Accuracy and error represent the gap between **predicted values (ŷ)** and **actual target values (y)**. After forward propagation, we compute this gap using specific functions.

---

## 2. Key Concepts
- **ŷ (y-hat)**: Predicted value from the neural network.
- **y**: Actual target value from the dataset.
- **Goal**: Minimize the difference between ŷ and y.

---

## 3. Functions Used
### **Loss Function**
- Measures **prediction error for a single sample**.

### **Cost Function**
- Measures **error across a set of samples**.
- Provides an **averaging effect** over all errors in the training dataset.
- Terms **loss function** and **cost function** are often used interchangeably.

---

## 4. Popular Cost Functions
- **Regression Problems**:
  - **Mean Square Error (MSE)**:
    $$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$
  - **Root Mean Square Error (RMSE)**:
    $$ \text{RMSE} = \sqrt{\text{MSE}} $$
    - Preferred because it keeps error in the same scale as target variables.

- **Classification Problems**:
  - **Binary Cross Entropy**: For binary classification.
  - **Categorical Cross Entropy**: For multi-class classification.

---

## 5. Measuring Accuracy
1. Send a set of samples through the ANN for **forward propagation**.
2. Predict outcomes (ŷ).
3. Compute **prediction error** using a cost function.
4. Use **backward propagation** to adjust weights and biases based on the error.

---

## 6. Next Steps
Backward propagation will be discussed in the next section.


# Back Propagation in Neural Networks

## 1. Overview
Back propagation is the process of **adjusting weights and biases** in a neural network to reduce prediction error. It works in the reverse direction of forward propagation.

---

## 2. Purpose
- After forward propagation, we compute the **overall prediction error**.
- Each node contributes to this error based on its **weights and biases**.
- Goal: Adjust weights and biases to **minimize error contribution** from each node.

---

## 3. How Back Propagation Works
1. **Start from the output layer**:
   - Compute a **delta value** based on the overall error.
   - Apply this delta to update weights and biases in the output layer.
2. **Move to the previous layer**:
   - Compute a new delta based on updated values in the current layer.
   - Apply delta to update weights and biases in the previous layer.
3. **Repeat the process**:
   - Continue computing deltas and updating weights layer by layer.
   - Stop when the input layer is reached.

---

## 4. Key Steps
- Compute **deltas (D1, D2, …)** for each layer.
- Apply deltas to **weights and biases**.
- Propagate deltas backward to influence previous layers.

---

## 5. Mathematical Details
- Back propagation involves **partial derivatives** of the cost function with respect to weights and biases.
- Deep learning libraries handle these computations internally.

---

## 6. Outcome
- Updated weights and biases that **reduce overall prediction error**.
- Prepares the network for the next iteration of training.

---

## 7. Next Steps
To further reduce error and improve accuracy, we repeat **forward propagation + back propagation** for multiple epochs.

