import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Helpers
# ============================================================

# creer un dossier s'il n'existe pas déjà
def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# convertir une liste ou un tuple en numpy array de type float
def to_numpy(x):
    return np.array(x, dtype=float)


# ============================================================
# Activations / forward pass utilities
# ============================================================

# Simple implementations of common activations, without any external library.
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

# To prevent overflow in exp, we clip the input to a reasonable range.
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# For numerical stability, we shift the input by its max before exponentiating.
def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(np.clip(shifted, -500, 500))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Apply the specified activation function to the input array.
def apply_activation(x: np.ndarray, name: str) -> np.ndarray:
    name = name.lower()
    if name == "relu":
        return relu(x)
    if name == "sigmoid":
        return sigmoid(x)
    if name == "softmax":
        return softmax(x)
    if name == "linear":
        return x
    raise ValueError(f"Unsupported activation: {name}")

# Perform a forward pass through the MLP and return detailed pre- and post-activation values for each layer.
def forward_with_details(
    sample: Sequence[float],
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    hidden_activation: str = "relu",
    output_activation: str = "softmax",
) -> List[Dict[str, np.ndarray]]:
    """
    Returns one dict per layer with:
        z: pre-activation
        a: post-activation

    Supports both weight layouts:
      - (input_size, output_size)
      - (output_size, input_size)
    """
    a = to_numpy(sample).reshape(1, -1)
    details = []

    for i, (w, b) in enumerate(zip(weights, biases)):
        w = to_numpy(w)
        b = to_numpy(b).reshape(1, -1)

        # Case 1: weights stored as (input_size, output_size)
        if a.shape[1] == w.shape[0]:
            z = a @ w

        # Case 2: weights stored as (output_size, input_size)
        elif a.shape[1] == w.shape[1]:
            z = a @ w.T

        else:
            raise ValueError(
                f"Shape mismatch on layer {i + 1}: "
                f"activation shape={a.shape}, weight shape={w.shape}"
            )

        if b.shape[1] != z.shape[1]:
            b = b.reshape(1, -1)
            if b.shape[1] != z.shape[1]:
                raise ValueError(
                    f"Bias shape mismatch on layer {i + 1}: "
                    f"z shape={z.shape}, bias shape={b.shape}"
                )

        z = z + b

        is_last = i == len(weights) - 1
        act = output_activation if is_last else hidden_activation
        a = apply_activation(z, act)

        details.append({
            "z": z.copy(),
            "a": a.copy(),
            "activation": act
        })

    return details


# ============================================================
# 1) Train / validation curves
# ============================================================

# affiche les courbes de loss et d'accuracy pour l'entraînement et la validation, à partir d'un dictionnaire d'historique
def plot_history(
    history: Dict[str, Sequence[float]],
    output_path: str = "visualizations/training_history.png",
) -> None:
    """
    Expected keys when available:
      - loss, val_loss, accuracy, val_accuracy
    """
    ensure_dir(Path(output_path).parent)

    epochs = range(1, len(history.get("loss", [])) + 1)
    plt.figure(figsize=(10, 6))

    has_loss = "loss" in history and len(history["loss"]) > 0
    has_val_loss = "val_loss" in history and len(history["val_loss"]) > 0
    has_acc = "accuracy" in history and len(history["accuracy"]) > 0
    has_val_acc = "val_accuracy" in history and len(history["val_accuracy"]) > 0

    if has_loss:
        plt.plot(epochs, history["loss"], label="Train loss")
    if has_val_loss:
        plt.plot(epochs, history["val_loss"], label="Validation loss")
    if has_acc:
        plt.plot(epochs, history["accuracy"], label="Train accuracy")
    if has_val_acc:
        plt.plot(epochs, history["val_accuracy"], label="Validation accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training history")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


# ============================================================
# 2) Weight heatmaps
# ============================================================

# Affiche des heatmaps des poids de chaque couche. Si les noms de features sont fournis, ils sont affichés sur la première couche.
def plot_weight_heatmaps(
    weights: Sequence[np.ndarray],
    feature_names: Optional[Sequence[str]] = None,
    output_dir: str = "visualizations/weights",
) -> None:
    ensure_dir(output_dir)

    for i, w in enumerate(weights, start=1):
        w = to_numpy(w)

        plt.figure(figsize=(max(6, w.shape[1] * 0.45), max(4, w.shape[0] * 0.22)))
        im = plt.imshow(w, aspect="auto")
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Weight value")
        plt.title(f"Layer {i} weights")

        # On suppose ici que les lignes = neurones de sortie
        # et les colonnes = entrées depuis la couche précédente
        plt.xlabel("Inputs from previous layer")
        plt.ylabel(f"Neurons in layer {i}")

        if i == 1 and feature_names is not None and len(feature_names) == w.shape[1]:
            plt.xticks(range(len(feature_names)), feature_names, rotation=90, fontsize=8)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"layer_{i}_weights.png", dpi=160)
        plt.close()


# ============================================================
# 3) Bias histograms
# ============================================================

# Affiche des histogrammes de la distribution des biais de chaque couche.
def plot_bias_histograms(
    biases: Sequence[np.ndarray],
    output_dir: str = "visualizations/biases",
) -> None:
    ensure_dir(output_dir)

    for i, b in enumerate(biases, start=1):
        b = to_numpy(b).ravel()
        plt.figure(figsize=(7, 4))
        plt.hist(b, bins=min(20, max(5, len(b))))
        plt.title(f"Layer {i} bias distribution")
        plt.xlabel("Bias value")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"layer_{i}_biases.png", dpi=160)
        plt.close()


# ============================================================
# 4) Activations for one sample
# ============================================================

# Affiche les activations de chaque couche pour un échantillon donné, en utilisant les poids et biais fournis.
# Les activations sont affichées sous forme de barres, avec une barre par neurone.
# Si les noms de classes sont fournis, ils sont affichés sur la couche de sortie.
def plot_sample_activations(
    sample: Sequence[float],
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    hidden_activation: str = "relu",
    output_activation: str = "softmax",
    class_names: Optional[Sequence[str]] = None,
    output_dir: str = "visualizations/activations",
    prefix: str = "sample",
) -> List[Dict[str, np.ndarray]]:
    ensure_dir(output_dir)
    details = forward_with_details(sample, weights, biases, hidden_activation, output_activation)

    for idx, layer in enumerate(details, start=1):
        values = layer["a"].ravel()
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(values)), values)
        plt.title(f"{prefix} - activations layer {idx} ({layer['activation']})")
        plt.xlabel("Neuron index")
        plt.ylabel("Activation")
        plt.grid(True, axis="y", alpha=0.3)

        if idx == len(details) and class_names is not None and len(class_names) == len(values):
            plt.xticks(range(len(values)), class_names)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"{prefix}_layer_{idx}_activations.png", dpi=160)
        plt.close()

    return details


# ============================================================
# 5) Confusion matrix
# ============================================================

# Calcule la matrice de confusion à partir des labels vrais et prédits,
# puis affiche cette matrice sous forme de heatmap avec les valeurs numériques.
def compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: Optional[int] = None,
) -> np.ndarray:
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max()) + 1)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    return cm

# Affiche la matrice de confusion sous forme de heatmap,
# avec les valeurs numériques affichées dans chaque cellule.
def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    output_path: str = "visualizations/confusion_matrix.png",
) -> np.ndarray:
    ensure_dir(Path(output_path).parent)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=len(class_names))

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return cm


# ============================================================
# 6) Softmax confidence
# ============================================================

# Affiche des histogrammes de la confiance (max probabilité softmax) des prédictions,
# avec une distinction entre les prédictions correctes et incorrectes si les labels vrais sont fournis.
def plot_softmax_confidence(
    probabilities: Sequence[Sequence[float]],
    y_true: Optional[Sequence[int]] = None,
    class_names: Optional[Sequence[str]] = None,
    output_dir: str = "visualizations/confidence",
) -> None:
    ensure_dir(output_dir)
    probas = to_numpy(probabilities)
    pred_idx = np.argmax(probas, axis=1)
    pred_conf = np.max(probas, axis=1)

    plt.figure(figsize=(7, 4))
    plt.hist(pred_conf, bins=20)
    plt.title("Prediction confidence histogram")
    plt.xlabel("Max softmax probability")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "confidence_histogram.png", dpi=160)
    plt.close()

    if y_true is not None:
        y_true = np.array(y_true, dtype=int)
        correct = pred_idx == y_true

        plt.figure(figsize=(7, 4))
        plt.hist(pred_conf[correct], bins=20, alpha=0.7, label="Correct")
        plt.hist(pred_conf[~correct], bins=20, alpha=0.7, label="Incorrect")
        plt.title("Confidence: correct vs incorrect")
        plt.xlabel("Max softmax probability")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "confidence_correct_vs_incorrect.png", dpi=160)
        plt.close()

    if class_names is not None and len(class_names) == probas.shape[1]:
        mean_proba = np.mean(probas, axis=0)
        plt.figure(figsize=(7, 4))
        plt.bar(range(len(mean_proba)), mean_proba)
        plt.xticks(range(len(mean_proba)), class_names)
        plt.title("Mean predicted probability per class")
        plt.ylabel("Probability")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "mean_probability_per_class.png", dpi=160)
        plt.close()


# ============================================================
# 7) PCA projection (without sklearn)
# ============================================================

# Effectue une projection PCA 2D sur les données d'entrée, sans utiliser de bibliothèque externe.
def pca_2d(x: Sequence[Sequence[float]]) -> np.ndarray:
    x = to_numpy(x)
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    cov = np.cov(x_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    principal_components = eigvecs[:, order[:2]]
    return x_centered @ principal_components

# Affiche la projection PCA 2D des données d'entrée, colorée par les labels vrais et prédits si disponibles.
def plot_pca_projection(
    x: Sequence[Sequence[float]],
    y_true: Sequence[int],
    y_pred: Optional[Sequence[int]] = None,
    class_names: Optional[Sequence[str]] = None,
    output_dir: str = "visualizations/pca",
) -> None:
    ensure_dir(output_dir)
    x_2d = pca_2d(x)
    y_true = np.array(y_true, dtype=int)

    plt.figure(figsize=(7, 6))
    for class_idx in np.unique(y_true):
        mask = y_true == class_idx
        label = class_names[class_idx] if class_names is not None else f"Class {class_idx}"
        plt.scatter(x_2d[mask, 0], x_2d[mask, 1], label=label, alpha=0.75)
    plt.title("PCA projection - true labels")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "pca_true_labels.png", dpi=160)
    plt.close()

    if y_pred is not None:
        y_pred = np.array(y_pred, dtype=int)
        plt.figure(figsize=(7, 6))
        for class_idx in np.unique(y_pred):
            mask = y_pred == class_idx
            label = class_names[class_idx] if class_names is not None else f"Class {class_idx}"
            plt.scatter(x_2d[mask, 0], x_2d[mask, 1], label=label, alpha=0.75)
        plt.title("PCA projection - predicted labels")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "pca_predicted_labels.png", dpi=160)
        plt.close()


# Affiche la projection PCA 2D des données d'entrée, colorée par la probabilité prédite pour une classe donnée.
def plot_pca_probability(
    x: Sequence[Sequence[float]],
    probabilities: Sequence[Sequence[float]],
    class_index: int = 1,
    output_dir: str = "visualizations/pca_proba",
) -> None:
    """
    PCA projection colored by predicted probability for one class.

    class_index=1 -> probability of class M
    class_index=0 -> probability of class B
    """
    ensure_dir(output_dir)

    x_np = to_numpy(x)
    probs_np = to_numpy(probabilities)

    if x_np.ndim != 2:
        raise ValueError("x must be a 2D array-like structure")
    if probs_np.ndim != 2:
        raise ValueError("probabilities must be a 2D array-like structure")
    if probs_np.shape[0] != x_np.shape[0]:
        raise ValueError("x and probabilities must have the same number of samples")
    if class_index < 0 or class_index >= probs_np.shape[1]:
        raise ValueError("class_index out of range")

    x_centered = x_np - np.mean(x_np, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
    components = vt[:2]
    projected = x_centered @ components.T

    confidence = probs_np[:, class_index]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        projected[:, 0],
        projected[:, 1],
        c=confidence,
        cmap="viridis",
        alpha=0.85
    )
    plt.colorbar(scatter, label=f"Predicted probability for class {class_index}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA projection - predicted probability for class {class_index}")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"pca_probability_class_{class_index}.png", dpi=160)
    plt.close()


# ============================================================
# 8) One-shot helper to generate everything
# ============================================================

# Appelle toutes les fonctions de visualisation avec les données fournies, en sauvegardant les résultats dans le dossier spécifié.
def generate_all_visualizations(
    history: Optional[Dict[str, Sequence[float]]] = None,
    weights: Optional[Sequence[np.ndarray]] = None,
    biases: Optional[Sequence[np.ndarray]] = None,
    x: Optional[Sequence[Sequence[float]]] = None,
    y_true: Optional[Sequence[int]] = None,
    y_pred: Optional[Sequence[int]] = None,
    probabilities: Optional[Sequence[Sequence[float]]] = None,
    sample: Optional[Sequence[float]] = None,
    feature_names: Optional[Sequence[str]] = None,
    class_names: Optional[Sequence[str]] = None,
    hidden_activation: str = "relu",
    output_activation: str = "softmax",
    output_dir: str = "visualizations",
) -> None:
    ensure_dir(output_dir)

    if history is not None:
        plot_history(history, output_path=str(Path(output_dir) / "training_history.png"))

    if weights is not None:
        plot_weight_heatmaps(weights, feature_names=feature_names, output_dir=str(Path(output_dir) / "weights"))

    if biases is not None:
        plot_bias_histograms(biases, output_dir=str(Path(output_dir) / "biases"))

    if y_true is not None and y_pred is not None and class_names is not None:
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            output_path=str(Path(output_dir) / "confusion_matrix.png"),
        )

    if probabilities is not None:
        plot_softmax_confidence(
            probabilities,
            y_true=y_true,
            class_names=class_names,
            output_dir=str(Path(output_dir) / "confidence"),
        )

    if sample is not None and weights is not None and biases is not None:
        plot_sample_activations(
            sample,
            weights,
            biases,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            class_names=class_names,
            output_dir=str(Path(output_dir) / "activations"),
            prefix="sample",
        )

    if x is not None and y_true is not None:
        plot_pca_projection(
            x,
            y_true,
            y_pred=y_pred,
            class_names=class_names,
            output_dir=str(Path(output_dir) / "pca"),
        )
        plot_pca_probability(
        x=x,
        probabilities=probabilities,
        class_index=1,
        output_dir=str(Path(output_dir) / "pca_proba")
        )
        plot_pca_probability(
        x=x,
        probabilities=probabilities,
        class_index=0,
        output_dir=str(Path(output_dir) / "pca_proba")
        )
