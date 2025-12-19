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
