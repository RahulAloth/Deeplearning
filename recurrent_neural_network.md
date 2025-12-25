
# Recurrent Neural Networks (RNNs)

Recurrent Neural Networks (RNNs) are designed to process **sequential data**, where the order of elements matters. Unlike feed-forward networks, RNNs maintain a memory of previous inputs, making them ideal for tasks involving context and time dependencies.

---

## Why RNNs?
- Feed-forward networks cannot handle sequences because they treat inputs independently.
- RNNs introduce **loops** in the architecture, allowing information to persist across time steps.

---

## Key Use Cases
- **Natural Language Processing (NLP)**: Language translation, text generation, sentiment analysis.
- **Speech Recognition**: Converting audio signals into text.
- **Time-Series Prediction**: Forecasting stock prices, weather patterns.
- **Video and Audio Analysis**: Sequential frame or sound processing.

---

## How RNNs Work
- Each time step processes:
  - Current input (\(x_t\))
  - Previous hidden state (\(h_{t-1}\))
- Produces:
  - New hidden state (\(h_t\))
  - Optional output (\(y_t\))
- This structure enables the network to “remember” past information.

---

## Types of RNN Architectures
1. **One-to-One**  
   - Single input → Single output  
   - Example: Simple classification.

2. **One-to-Many**  
   - Single input → Sequence of outputs  
   - Example: Music generation.

3. **Many-to-One**  
   - Sequence of inputs → Single output  
   - Example: Sentiment analysis.

4. **Many-to-Many (Equal)**  
   - Sequence of inputs → Sequence of outputs (same length)  
   - Example: Machine translation.

5. **Many-to-Many (Unequal)**  
   - Sequence of inputs → Sequence of outputs (different lengths)  
   - Example: Video classification.

---

## Advantages
- Handles sequential and time-dependent data.
- Maintains context across inputs.
- Suitable for tasks where previous information influences future predictions.

---

## Challenges
- **Vanishing/Exploding Gradients** during training.
- Difficulty in capturing long-term dependencies (addressed by LSTM and GRU variants).

---

## Applications
- Pattern detection
- Speech and voice recognition
- Language modeling and translation
- Time-series forecasting
- Image captioning
