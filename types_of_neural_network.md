# Neural Network Architectures: CNN, RNN, and Transformers

---

## 1. RNN vs Feedforward Neural Networks

- **Recurrent Neural Networks (RNNs)**  
  - Have feedback connections, allowing them to remember previous states.
  - Best suited for **sequential data** tasks such as:
    - Natural Language Processing (NLP)
    - Speech recognition
    - Time-series prediction

- **Feedforward Neural Networks**  
  - No feedback connections; cannot retain past information.
  - Best suited for **non-sequential data** tasks such as:
    - Image classification (with CNNs)
    - Tabular data predictions

---

## 2. Transformer Architecture for NLP Tasks

- **Why Transformers?**
  - Ideal for machine translation, text summarization, and question answering.
  - **Parallelizable**: Can run on multiple processors for faster training.
  - Uses **Attention Mechanism**:
    - Learns relationships between different parts of a sequence.
    - More powerful than recurrent connections in RNNs.

- **Core Components**
  - **Encoder**: Encodes input sequence into representations.
  - **Decoder**: Decodes representations into output sequence.
  - **Attention**: Captures dependencies across tokens without sequential processing.

- **Key Difference from RNNs**
  - Transformers do not rely on sequential order for processing.
  - Easier to parallelize → more efficient training.

---

## 3. Convolutional Neural Networks (CNNs)

- **Purpose**
  - Designed for **image recognition and classification** tasks.
  - Extracts features from images using convolution and pooling.

- **Layers in CNN**
  1. **Input Layer**: Accepts image as 2D or 3D tensor.
  2. **Convolutional Layers**: Apply filters to extract features.
  3. **Pooling Layers**: Reduce feature map size to lower complexity and prevent overfitting.
  4. **Fully Connected Layers**: Combine extracted features for prediction.
  5. **Output Layer**: Produces final classification (e.g., Softmax).

---

### Summary Table

| Architecture   | Best For                          | Key Feature                  |
|---------------|-----------------------------------|-----------------------------|
| **Feedforward** | Non-sequential data              | Simple forward connections  |
| **RNN**        | Sequential data (text, audio)    | Feedback loops for memory   |
| **Transformer**| NLP tasks, parallel processing   | Attention mechanism         |
| **CNN**        | Image recognition                | Convolution + pooling       |

---
