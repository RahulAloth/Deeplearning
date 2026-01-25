
# 📘 Study Note: Common Loss Functions in Deep Learning

In deep learning, a **loss function** measures how far a model’s predictions deviate from the true target values. During training, optimization algorithms (like SGD or Adam) use the loss as feedback to adjust model parameters (weights and biases). Choosing the right loss function is essential because it directly influences how effectively a model learns for a given task.

---

## 🔍 What Is a Loss Function?

A loss function:

- Quantifies the error between predictions and true labels.  
- Guides the optimizer during backpropagation.  
- Helps the model gradually improve by minimizing this error.

Different tasks require different loss functions. The most common categories are:

- **Regression** (predicting continuous values)  
- **Binary classification** (two classes)  
- **Multiclass classification** (three or more classes)

---

# 1. 📈 Loss Functions for Regression

Regression tasks involve predicting continuous numeric values (e.g., house prices, temperatures).

---

## **1.1 Mean Squared Error (MSE)**

One of the most widely used regression losses.

### **Formula**
\[
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
\]

### **Key Characteristics**
- Penalizes large errors more strongly due to squaring  
- Always non‑negative  
- Sensitive to outliers  

### **Typical Use Cases**
- Stock price prediction  
- Forecasting  
- Low‑noise regression tasks  

---

## **1.2 Mean Absolute Error (MAE)**

### **Formula**
\[
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
\]

### **Key Characteristics**
- More robust to outliers than MSE  
- Penalizes deviations linearly  
- Converges slower because gradient is constant and non‑smooth at zero  

### **Typical Use Cases**
- Noisy datasets  
- When large deviations should not be heavily penalized  

---

# 2. ⚖️ Loss Functions for Binary Classification

Binary classification predicts one of two possible classes, usually encoded as `0` or `1`.

---

## **2.1 Binary Cross‑Entropy (BCE)**  
Also known as **log loss**.

### **Formula**
\[
\text{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right]
\]

### **Key Characteristics**
- Measures closeness of predicted probabilities to true labels  
- Encourages confident and correct predictions  

### **Typical Applications**
- Spam detection  
- Fraud detection  
- Medical diagnosis  
- Any yes/no classification task  

---

# 3. 🎨 Loss Functions for Multiclass Classification

Multiclass classification predicts one class out of many possible categories.

---

## **3.1 Categorical Cross‑Entropy (CCE)**  
Used when labels are **one‑hot encoded**.

### **Formula**
\[
\text{CCE} = -\sum_{i=1}^{N} \sum_{j=1}^{K} y_{ij} \log(\hat{y}_{ij})
\]

### **Key Characteristics**
- Compares predicted probability distribution with the true one‑hot encoding  
- Penalizes misclassification proportionally to predicted probability  

### **Typical Applications**
- Image classification (CIFAR‑10, MNIST)  
- Text classification  
- Audio classification  

---

## **3.2 Sparse Categorical Cross‑Entropy**

### **When to Use**
- Labels are integer encoded (`0–9`)  
- Avoiding one‑hot encoding for efficiency  

Same mathematical idea as CCE, but suitable for integer labels.

---

# 4. 🧩 Specialized Loss Functions in Advanced Deep Learning

Some tasks require domain‑specific loss functions tailored to unique data structures.

---

## **4.1 Intersection over Union (IoU) Loss**
Used for:
- Object detection  
- Semantic segmentation  

Measures overlap between predicted and true regions.

---

## **4.2 Dice Loss**
Used for:
- Medical image segmentation  
- Imbalanced segmentation datasets  

Optimizes overlap between predicted and actual masks.

---

## **4.3 Sequence Loss**
Used for:
- Machine translation  
- Text generation  
- Speech recognition  

Handles variable‑length sequence outputs.

---

# 🧠 Summary

Choosing the right loss function is crucial for model performance:

| Task Type | Recommended Loss Function |
|----------|----------------------------|
| Regression | MSE, MAE |
| Binary Classification | Binary Cross‑Entropy |
| Multiclass Classification | Categorical Cross‑Entropy / Sparse Categorical Cross‑Entropy |
| Object Detection | IoU Loss |
| Segmentation | Dice Loss |
| Sequence Modeling | Sequence Loss |

The loss function is the core driver of training—guiding the optimizer to reduce error and improve the model’s predictive accuracy.
