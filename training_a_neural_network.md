# Deep Learning Model Training: Setup and Initialization

## 1. Overview
Before training a deep learning model, several steps are required to prepare the data and configure the model. This section covers **setup and initialization** for the training process.

---

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
