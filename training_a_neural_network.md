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

