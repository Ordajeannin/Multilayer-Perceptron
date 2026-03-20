# Multilayer Perceptron (MLP) - Breast Cancer Classification

## Overview

This project consists of implementing a **Multilayer Perceptron (MLP)** from scratch to classify breast tumors as:

- **M** (Malignant)
- **B** (Benign)

The model is trained on the **Wisconsin Breast Cancer dataset**, using only basic Python and math — no machine learning libraries.

---

## Objectives

- Understand and implement:
  - Feedforward
  - Backpropagation
  - Gradient Descent
- Build a modular neural network
- Visualize learning with loss and accuracy curves
- Evaluate model performance on unseen data

---

## How it works

### 1. Dataset

Each sample contains:
- ID
- Label (`M` or `B`)
- 30 numerical features

Example:

842302,M,17.99,10.38,...

---

### 2. Preprocessing

- Convert labels:
  - `M → 1`
  - `B → 0`
- Normalize features:

x' = (x - mean) / std

- Statistics computed on **training set only**

---

### 3. Model Architecture

Input (30)
↓
Hidden Layer (16 neurons, sigmoid)
↓
Hidden Layer (16 neurons, sigmoid)
↓
Output Layer (2 neurons, softmax)

---

### 4. Training

- Loss: **Cross-Entropy**
- Optimization: **Gradient Descent (SGD)**
- Backpropagation implemented manually
- Shuffle dataset at each epoch
- Best model selected using **validation loss**

---

## Usage

### 1. Split dataset

python3 split.py source/data.csv train.csv valid.csv

---

### 2. Train model

python3 train.py train.csv valid.csv

Outputs:
- model.json
- loss.png
- accuracy.png

---

### 3. Predict

python3 predict.py model.json valid.csv

---

## Results

- Train accuracy: ~99%
- Validation accuracy: ~97%
- Good generalization

---

## Learning Curves

- Loss curve
- Accuracy curve

---

## Key Concepts

### Feedforward
X → Hidden Layers → Output

### Backpropagation
error → gradients → update

### Gradient Descent
w = w - lr × gradient

---

## Notes

- No ML libraries
- Uses validation set
- Softmax output
- Best model saved

---

## Possible Improvements

- Mini-batch
- Adam optimizer
- Early stopping
- Regularization

---

## References

https://en.wikipedia.org/wiki/Multilayer_perceptron
https://en.wikipedia.org/wiki/Backpropagation
https://en.wikipedia.org/wiki/Cross_entropy
