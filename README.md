# Multilayer Perceptron (MLP) - Breast Cancer Classification

## Overview

This project implements a **Multilayer Perceptron (MLP)** from scratch to classify breast tumors as:

- **M** (Malignant)
- **B** (Benign)

The model is trained on the **Wisconsin Breast Cancer dataset**, using only Python and basic numerical operations — **no machine learning libraries**.

---

## Objectives

- Understand and implement core ML concepts:
  - Feedforward
  - Backpropagation
  - Gradient Descent
- Build a modular neural network from scratch
- Evaluate model performance rigorously
- Compare multiple architectures
- Ensure robustness using cross-validation

---

## Dataset

Each sample contains:
- An ID
- A label (`M` or `B`)
- 30 numerical features describing cell nuclei

Example:

```
842302,M,17.99,10.38,...
```

---

## Preprocessing

- Label encoding:
  - `M → 1`
  - `B → 0`

- Feature normalization:

```
x' = (x - mean) / std
```

- Normalization statistics are computed on the **training set only** to avoid data leakage.

---

## Model Architecture

Default architecture (best model):

```
Input (30)
↓
Hidden Layer (16 neurons, sigmoid)
↓
Hidden Layer (16 neurons, sigmoid)
↓
Output Layer (2 neurons, softmax)
```

---

## Training

- Loss function: **Cross-Entropy**
- Optimization: **Stochastic Gradient Descent (SGD)**
- Manual implementation of:
  - Forward propagation
  - Backpropagation
  - Gradient updates
- Dataset shuffled at each epoch
- Model selection based on **validation loss**
- **Early stopping** implemented to prevent overfitting

---

## Usage

### 1. Split dataset

```
python3 split.py source/data.csv train.csv valid.csv
```

---

### 2. Train model

```
python3 train.py train.csv valid.csv
```

Outputs:
- `model.json`
- `loss.png`
- `accuracy.png`
- `history.json`
- `metrics.json`

---

### 3. Predict

```
python3 predict.py model.json valid.csv
```

Outputs:
- Predictions
- Accuracy
- Precision / Recall / F1-score
- Confusion matrix (textual)

---

## Model Evaluation

### Metrics used

- Accuracy
- Precision
- Recall (important for malignant detection)
- F1-score
- Validation loss

In a medical context, **recall is critical** to minimize false negatives (missing a malignant tumor).

---

## Cross-Validation (Bonus)

To ensure robustness, a **5-fold cross-validation** was implemented.

Instead of relying on a single train/validation split, the model is trained and evaluated on multiple splits.

This provides:
- Mean performance
- Standard deviation (stability)

---

## Model Comparison

### 📊 Cross-validation results

| Rank | Model        | Accuracy | Precision | Recall | F1     | Val Loss | Std F1 | Std Recall |
|------|-------------|----------|----------|--------|--------|----------|--------|-----------|
| 1    | 16_16_lr001 | 0.9824   | 0.9863   | 0.9635 | 0.9742 | 0.0719   | 0.0139 | 0.0341    |
| 2    | 16_8_lr001  | 0.9824   | 0.9834   | 0.9635 | 0.9731 | 0.0719   | 0.0215 | 0.0341    |
| 3    | 32_32_lr001 | 0.9807   | 0.9792   | 0.9635 | 0.9710 | 0.0698   | 0.0203 | 0.0341    |
| 4    | 8_8_lr001   | 0.9772   | 0.9720   | 0.9568 | 0.9641 | 0.0709   | 0.0339 | 0.0467    |

---

## Best Model Selection

The selected model is:

```
16_16_lr001
```

### Why?

- Highest F1-score
- Strong recall (important in medical diagnosis)
- Lowest variability across folds (most stable)
- Good balance between complexity and performance

---

## Key Insights

- Increasing model complexity does not always improve performance
- Smaller models (e.g., 8-8) can perform well but are less stable
- Larger models (32-32) may introduce variance (overfitting risk)
- The best model lies in a **balance between bias and variance**

---

## Learning Curves

The project generates:
- Training and validation loss curves
- Training and validation accuracy curves
- Comparison plots across models

These help visualize:
- Convergence
- Overfitting
- Model stability

---

## Key Concepts

### Feedforward
```
Input → Hidden Layers → Output
```

### Backpropagation
```
Error → Gradients → Weight updates
```

### Gradient Descent
```
w = w - learning_rate × gradient
```

---

## Constraints

- No machine learning libraries allowed
- Full implementation from scratch
- Manual handling of:
  - Gradients
  - Loss
  - Optimizer

---

## Possible Improvements

- Mini-batch gradient descent
- Advanced optimizers (Adam, RMSprop)
- Regularization (L2, dropout)
- Automatic hyperparameter search
- ROC curve / AUC evaluation

---

## References

- https://en.wikipedia.org/wiki/Multilayer_perceptron
- https://en.wikipedia.org/wiki/Backpropagation
- https://en.wikipedia.org/wiki/Cross_entropy
