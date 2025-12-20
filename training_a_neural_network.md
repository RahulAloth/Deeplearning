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

# Gradient Descent in Neural Networks

## 1. Overview
Gradient Descent is an **optimization algorithm** used to minimize the error (or cost) in a neural network by iteratively adjusting weights and biases. It is the backbone of training deep learning models.

---

## 2. Why Gradient Descent?
- After **forward propagation**, we compute predictions (ŷ).
- We calculate **error** using a cost function (e.g., MSE, Cross-Entropy).
- **Back propagation** computes gradients (partial derivatives) of the cost function with respect to weights and biases.
- Gradient Descent uses these gradients to **update weights and biases** in the direction that reduces error.

---

## 3. Core Idea
- The gradient indicates the **direction of steepest increase** in error.
- To minimize error, we move **in the opposite direction of the gradient**.
- Update rule:
  $$ w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial \text{Cost}}{\partial w} $$
  $$ b_{\text{new}} = b_{\text{old}} - \eta \cdot \frac{\partial \text{Cost}}{\partial b} $$
  Where:
  - \( \eta \) = **learning rate** (controls step size)
  - \( \frac{\partial \text{Cost}}{\partial w} \) = gradient of cost w.r.t weight

---

## 4. The Process
1. **Forward Propagation**:
   - Predict outputs using current weights and biases.
2. **Compute Error**:
   - Use a cost function to measure prediction error.
3. **Back Propagation**:
   - Calculate gradients for each weight and bias.
4. **Update Parameters**:
   - Apply gradient descent update rule.
5. **Repeat**:
   - Iterate over multiple epochs until error converges or stops improving.

---

## 5. Behavior of Gradient Descent
- As iterations progress:
  - Error **oscillates** but generally moves closer to zero.
  - We approach a **minimum of the cost function**.
- **Goal**: Find the global minimum (or a good local minimum) of the cost function.

---

## 6. Key Hyperparameters
- **Learning Rate (η)**:
  - Too high → overshooting, divergence.
  - Too low → slow convergence.
- **Number of Epochs**:
  - More epochs → better convergence (up to a point).
- **Batch Size**:
  - Full Batch Gradient Descent: Uses entire dataset per update.
  - Mini-Batch Gradient Descent: Uses small batches (common in deep learning).
  - Stochastic Gradient Descent (SGD): Uses one sample per update.

---

## 7. Variants of Gradient Descent
- **SGD (Stochastic Gradient Descent)**:
  - Faster updates, introduces randomness.
- **Momentum**:
  - Adds velocity term to smooth updates.
- **Adam Optimizer**:
  - Combines momentum and adaptive learning rates.
- **RMSProp**:
  - Adjusts learning rate based on recent gradients.

---

## 8. Challenges & Insights
- **Local Minima**:
  - Cost function may have multiple minima.
- **Vanishing/Exploding Gradients**:
  - Common in deep networks; mitigated by proper initialization and normalization.
- **Learning Rate Scheduling**:
  - Dynamic adjustment of learning rate improves convergence.

---

## 9. Intuition Behind Gradient Descent
Think of gradient descent as **rolling down a hill**:
- The hill = cost function surface.
- The ball = current weights.
- The slope = gradient.
- The step size = learning rate.
- Goal: Reach the lowest point (minimum error).

---

## 10. Summary
Gradient Descent is **not a single step**, but a **repeated cycle of learning**:
- Forward Propagation → Compute Error → Back Propagation → Update Weights.
- Repeat until the model achieves acceptable accuracy.

---

# Gradient Descent in Neural Networks

## 1. Overview
Gradient Descent is an **optimization algorithm** used to minimize the error (or cost) in a neural network by iteratively adjusting weights and biases. It is the backbone of training deep learning models.

---

## 2. Why Gradient Descent?
- After **forward propagation**, we compute predictions (ŷ).
- We calculate **error** using a cost function (e.g., MSE, Cross-Entropy).
- **Back propagation** computes gradients (partial derivatives) of the cost function with respect to weights and biases.
- Gradient Descent uses these gradients to **update weights and biases** in the direction that reduces error.

---

## 3. Core Idea
- The gradient indicates the **direction of steepest increase** in error.
- To minimize error, we move **in the opposite direction of the gradient**.
- Update rule:
  $$ w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial \text{Cost}}{\partial w} $$
  $$ b_{\text{new}} = b_{\text{old}} - \eta \cdot \frac{\partial \text{Cost}}{\partial b} $$
  Where:
  - \( \eta \) = **learning rate** (controls step size)
  - \( \frac{\partial \text{Cost}}{\partial w} \) = gradient of cost w.r.t weight

---

## 4. The Process
1. **Forward Propagation**:
   - Predict outputs using current weights and biases.
2. **Compute Error**:
   - Use a cost function to measure prediction error.
3. **Back Propagation**:
   - Calculate gradients for each weight and bias.
4. **Update Parameters**:
   - Apply gradient descent update rule.
5. **Repeat**:
   - Iterate over multiple epochs until error converges or stops improving.

---

## 5. Behavior of Gradient Descent
- As iterations progress:
  - Error **oscillates** but generally moves closer to zero.
  - We approach a **minimum of the cost function**.
- **Goal**: Find the global minimum (or a good local minimum) of the cost function.

---

## 6. Key Hyperparameters
- **Learning Rate (η)**:
  - Too high → overshooting, divergence.
  - Too low → slow convergence.
- **Number of Epochs**:
  - More epochs → better convergence (up to a point).
- **Batch Size**:
  - Full Batch Gradient Descent: Uses entire dataset per update.
  - Mini-Batch Gradient Descent: Uses small batches (common in deep learning).
  - Stochastic Gradient Descent (SGD): Uses one sample per update.

---

## 7. Variants of Gradient Descent
- **SGD (Stochastic Gradient Descent)**:
  - Faster updates, introduces randomness.
- **Momentum**:
  - Adds velocity term to smooth updates.
- **Adam Optimizer**:
  - Combines momentum and adaptive learning rates.
- **RMSProp**:
  - Adjusts learning rate based on recent gradients.

---

## 8. Challenges & Insights
- **Local Minima**:
  - Cost function may have multiple minima.
- **Vanishing/Exploding Gradients**:
  - Common in deep networks; mitigated by proper initialization and normalization.
- **Learning Rate Scheduling**:
  - Dynamic adjustment of learning rate improves convergence.

---

## 9. Intuition Behind Gradient Descent
Think of gradient descent as **rolling down a hill**:
- The hill = cost function surface.
- The ball = current weights.
- The slope = gradient.
- The step size = learning rate.
- Goal: Reach the lowest point (minimum error).

---

## 10. Summary
Gradient Descent is **not a single step**, but a **repeated cycle of learning**:
- Forward Propagation → Compute Error → Back Propagation → Update Weights.
- Repeat until the model achieves acceptable accuracy.

---


# Gradient Descent in Neural Networks

## 1. Overview
Gradient Descent is an **optimization algorithm** used to minimize the error (or cost) in a neural network by iteratively adjusting weights and biases. It is the backbone of training deep learning models.

---

## 2. Why Gradient Descent?
- After **forward propagation**, we compute predictions (ŷ).
- We calculate **error** using a cost function (e.g., MSE, Cross-Entropy).
- **Back propagation** computes gradients (partial derivatives) of the cost function with respect to weights and biases.
- Gradient Descent uses these gradients to **update weights and biases** in the direction that reduces error.

---

## 3. Core Idea
- The gradient indicates the **direction of steepest increase** in error.
- To minimize error, we move **in the opposite direction of the gradient**.
- Update rule:
  $$ w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial \text{Cost}}{\partial w} $$
  $$ b_{\text{new}} = b_{\text{old}} - \eta \cdot \frac{\partial \text{Cost}}{\partial b} $$
  Where:
  - \( \eta \) = **learning rate** (controls step size)
  - \( \frac{\partial \text{Cost}}{\partial w} \) = gradient of cost w.r.t weight

---

## 4. The Process
1. **Forward Propagation**:
   - Predict outputs using current weights and biases.
2. **Compute Error**:
   - Use a cost function to measure prediction error.
3. **Back Propagation**:
   - Calculate gradients for each weight and bias.
4. **Update Parameters**:
   - Apply gradient descent update rule.
5. **Repeat**:
   - Iterate over multiple epochs until error converges or stops improving.

---

## 5. Behavior of Gradient Descent
- As iterations progress:
  - Error **oscillates** but generally moves closer to zero.
  - We approach a **minimum of the cost function**.
- **Goal**: Find the global minimum (or a good local minimum) of the cost function.

---

## 6. Key Hyperparameters
- **Learning Rate (η)**:
  - Too high → overshooting, divergence.
  - Too low → slow convergence.
- **Number of Epochs**:
  - More epochs → better convergence (up to a point).
- **Batch Size**:
  - Full Batch Gradient Descent: Uses entire dataset per update.
  - Mini-Batch Gradient Descent: Uses small batches (common in deep learning).
  - Stochastic Gradient Descent (SGD): Uses one sample per update.

---

## 7. Variants of Gradient Descent
- **SGD (Stochastic Gradient Descent)**:
  - Faster updates, introduces randomness.
- **Momentum**:
  - Adds velocity term to smooth updates.
- **Adam Optimizer**:
  - Combines momentum and adaptive learning rates.
- **RMSProp**:
  - Adjusts learning rate based on recent gradients.

---

## 8. Challenges & Insights
- **Local Minima**:
  - Cost function may have multiple minima.
- **Vanishing/Exploding Gradients**:
  - Common in deep networks; mitigated by proper initialization and normalization.
- **Learning Rate Scheduling**:
  - Dynamic adjustment of learning rate improves convergence.

---

## 9. Intuition Behind Gradient Descent
Think of gradient descent as **rolling down a hill**:
- The hill = cost function surface.
- The ball = current weights.
- The slope = gradient.
- The step size = learning rate.
- Goal: Reach the lowest point (minimum error).

---

## 10. Summary
Gradient Descent is **not a single step**, but a **repeated cycle of learning**:
- Forward Propagation → Compute Error → Back Propagation → Update Weights.
- Repeat until the model achieves acceptable accuracy.

---


## 1. Overview
Gradient Descent is an **optimization algorithm** used to minimize the error (or cost) in a neural network by iteratively adjusting weights and biases. It is the backbone of training deep learning models.

---

## 2. Why Gradient Descent?
- After **forward propagation**, we compute predictions (ŷ).
- We calculate **error** using a cost function (e.g., MSE, Cross-Entropy).
- **Back propagation** computes gradients (partial derivatives) of the cost function with respect to weights and biases.
- Gradient Descent uses these gradients to **update weights and biases** in the direction that reduces error.

---

## 3. Core Idea
- The gradient indicates the **direction of steepest increase** in error.
- To minimize error, we move **in the opposite direction of the gradient**.

### Update rules (display math):
$$
 w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial \text{Cost}}{\partial w}
$$
$$
 b_{\text{new}} = b_{\text{old}} - \eta \cdot \frac{\partial \text{Cost}}{\partial b}
$$

Where: \( \eta \) = **learning rate** (step size), and \( \frac{\partial \text{Cost}}{\partial w} \) is the gradient of the cost with respect to the weight.

---

## 4. The Process
1. **Forward Propagation**: Predict outputs using current weights and biases.
2. **Compute Error**: Use a cost function to measure prediction error.
3. **Back Propagation**: Calculate gradients for each weight and bias.
4. **Update Parameters**: Apply gradient descent update rule.
5. **Repeat**: Iterate over multiple epochs until error converges or stops improving.

---

## 5. Behavior of Gradient Descent
- As iterations progress:
  - Error **oscillates** but generally moves closer to zero.
  - We approach a **minimum of the cost function**.
- **Goal**: Find the global minimum (or a good local minimum) of the cost function.

---

## 6. Key Hyperparameters
- **Learning Rate (\(\eta\))**:
  - Too high → overshooting, divergence.
  - Too low → slow convergence.
- **Number of Epochs**: More epochs → better convergence (up to a point).
- **Batch Size**:
  - Full Batch Gradient Descent: Uses entire dataset per update.
  - Mini-Batch Gradient Descent: Uses small batches (common in deep learning).
  - Stochastic Gradient Descent (SGD): Uses one sample per update.

---

## 7. Variants of Gradient Descent
- **SGD (Stochastic Gradient Descent)**: Faster updates, introduces randomness.
- **Momentum**: Adds velocity term to smooth updates.
- **Adam Optimizer**: Combines momentum and adaptive learning rates.
- **RMSProp**: Adjusts learning rate based on recent gradients.

---

## 8. Challenges & Insights
- **Local Minima**: Cost function may have multiple minima.
- **Vanishing/Exploding Gradients**: Common in deep networks; mitigated by proper initialization and normalization.
- **Learning Rate Scheduling**: Dynamic adjustment of learning rate improves convergence.

---

## 9. Intuition Behind Gradient Descent
Think of gradient descent as **rolling down a hill**:
- The hill = cost function surface.
- The ball = current weights.
- The slope = gradient.
- The step size = learning rate.
- Goal: Reach the lowest point (minimum error).

---

## 10. Summary
Gradient Descent is **not a single step**, but a **repeated cycle of learning**:
- Forward Propagation → Compute Error → Back Propagation → Update Weights.
- Repeat until the model achieves acceptable accuracy.

---



# Validation and Testing in Neural Networks

## 1. Overview
Validation and testing are critical steps in building robust neural network models. They ensure that the model performs well **not only on training data** but also on **unseen data**, reducing the risk of overfitting.

---

## 2. Why Do We Need Validation and Testing?
- **Training error (in-sample error)**:
  - Computed during training using the training dataset.
  - Does not guarantee performance on new data.
- **Out-of-sample error**:
  - Measured using independent datasets (validation and test sets).
  - Indicates how well the model generalizes.

---

## 3. Validation
- **Purpose**:
  - Monitor model performance during training.
  - Detect overfitting early.
- **Process**:
  1. After each epoch (or batch), use the model to predict on the **validation set**.
  2. Compute **accuracy and loss** for validation data.
  3. Compare validation error with training error:
     - If validation error increases while training error decreases → **Overfitting**.
- **Action**:
  - Fine-tune hyperparameters (learning rate, batch size, regularization).
  - Apply techniques like **early stopping** if validation loss stops improving.

---

## 4. Testing
- **Purpose**:
  - Final evaluation of the model after all tuning is complete.
- **Process**:
  - Use the **test set** (never seen during training or validation).
  - Compute final metrics: accuracy, error rates, precision, recall, F1-score (depending on problem type).
- **Key Point**:
  - Testing is done **only once** at the end to measure true generalization.

---

## 5. Best Practices
- **Data Split**:
  - Typical ratio: `80% Training | 10% Validation | 10% Test`.
- **Avoid Data Leakage**:
  - Ensure validation and test sets are completely independent.
- **Monitor Metrics**:
  - Track both training and validation curves to detect overfitting.
- **Use Cross-Validation**:
  - For small datasets, k-fold cross-validation improves reliability.

---

## 6. Insights
- Validation helps **guide training**; testing provides **final judgment**.
- A model with very low training error but high validation error is **overfitted**.
- A model with similar training and validation errors but poor test performance may suffer from **data mismatch** or **underfitting**.

---

## 7. Summary
- **Validation**: Continuous monitoring during training for tuning and early stopping.
- **Testing**: One-time evaluation after training to measure real-world performance.

---
# 🧠 Understanding an ANN Model

## ✅ What is an ANN Model?
An **Artificial Neural Network (ANN)** model is a computational structure inspired by biological neural networks. It is primarily used for tasks like classification, regression, and pattern recognition.

### **Key Components**
1. **Parameters**
   - **Weights**: Numerical values that determine the strength of connections between neurons.
   - **Biases**: Values added to the weighted sum to shift activation.
   - These are learned during **training**.
   - When someone says a model has *X parameters*, they refer to the total count of weights and biases.

2. **Hyperparameters**
   - **Architecture**: Number of layers and nodes per layer.
   - **Activation Functions**: Functions applied at each node (e.g., ReLU, Sigmoid).
   - **Cost Function**: Measures prediction error.
   - **Optimizer**: Algorithm for updating weights (e.g., SGD, Adam).
   - **Learning Rate**: Controls step size during optimization.
   - **Batch Size & Epochs**: Define how data is processed during training.

> **Insight:** Hyperparameters are not learned; they are chosen before training and significantly impact model performance.

---

## 📂 Model Representation
- A saved model file typically contains:
  - Learned **weights and biases**.
  - Chosen **hyperparameters**.
- Models can be:
  - **Saved** for reuse.
  - **Shared** across systems.
  - **Loaded** into other applications.

---

## 🔮 Prediction Process
- **Forward Propagation** is used for prediction:
  1. Preprocess input features.
  2. Pass inputs through layers using final weights and biases.
  3. Compute outputs at each node.
  4. Derive final predictions.
- **Post-processing** may be needed to convert raw outputs into business-friendly formats (e.g., probabilities → labels).

> **Insight:** Prediction is computationally lighter than training since it skips backpropagation.

---

## 💡 Practical Takeaways
- ANN models are **parameter-heavy**, so memory and compute requirements matter.
- **Hyperparameter tuning** is critical for achieving optimal performance.
- **Model portability** (saving/loading) enables deployment in real-world applications.
- Prediction pipelines often include **data preprocessing** and **post-processing** for usability.

---

### ✅ Next Steps
Now that you understand the fundamentals, the next chapter typically involves **building a real ANN model** using frameworks like TensorFlow or PyTorch.
