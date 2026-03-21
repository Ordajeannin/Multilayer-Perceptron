import random
import sys

from config.config import (
    LEARNING_RATE,
    EPOCHS,
    HIDDEN_SIZES,
    MODEL_PATH,
    LOSS_PLOT_PATH,
    ACCURACY_PLOT_PATH,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    HISTORY_PATH,
    METRICS_PATH,
)
from utils.data_utils import (
    load_dataset,
    compute_normalization_stats,
    normalize_dataset,
    one_hot_encode,
)
from utils.model_utils import (
    initialize_network,
    train_model,
    save_model,
    save_history,
    forward_sample,
    predict_class,
    compute_classification_metrics,
    save_metrics,
)
from utils.plot_utils import (
    plot_loss,
    plot_accuracy,
)
from script.mlp_visualization import generate_all_visualizations

# permet d'extraire les poids et les biais du réseau de neurones entraîné,
# en parcourant chaque couche du réseau et en récupérant les poids et les biais correspondants
def extract_weights_and_biases(network):
    weights = []
    biases = []

    for layer in network:
        if "weights" in layer and "biases" in layer:
            weights.append(layer["weights"])
            biases.append(layer["biases"])
        elif "W" in layer and "b" in layer:
            weights.append(layer["W"])
            biases.append(layer["b"])
        else:
            raise ValueError("Impossible de trouver les poids/biais dans une couche du réseau")

    return weights, biases

# permet d'évaluer les prédictions du réseau sur un ensemble de données, 
def evaluate_for_visualization(network, X):
    probabilities = []
    y_pred = []

    for sample in X:
        probs, _ = forward_sample(network, sample)
        pred = predict_class(probs)
        probabilities.append(probs)
        y_pred.append(pred)

    return probabilities, y_pred

# point d'entrée du script d'entraînement, qui charge les datasets d'entraînement et de validation,
# normalise les données, initialise le réseau de neurones, entraîne le modèle en utilisant l'ensemble d'entraînement
# et en évaluant sur l'ensemble de validation à chaque époque, sauvegarde le modèle entraîné
# et les courbes de perte et d'exactitude, et affiche les métriques d'évaluation finales
def main():
    if len(sys.argv) != 3:
        print("Usage: python3 train.py <train.csv> <valid.csv>")
        sys.exit(1)

    train_path = sys.argv[1]
    valid_path = sys.argv[2]

    try:
        random.seed(42)

        X_train, y_train = load_dataset(train_path)
        X_valid, y_valid = load_dataset(valid_path)

        means, stds = compute_normalization_stats(X_train)
        X_train = normalize_dataset(X_train, means, stds)
        X_valid = normalize_dataset(X_valid, means, stds)

        print(f"x_train shape: ({len(X_train)}, {len(X_train[0])})")
        print(f"y_train shape: ({len(y_train)},)")
        print(f"x_valid shape: ({len(X_valid)}, {len(X_valid[0])})")
        print(f"y_valid shape: ({len(y_valid)},)")

        network = initialize_network(
            input_size=len(X_train[0]),
            hidden_sizes=HIDDEN_SIZES,
            output_size=2
        )

        history, best_network = train_model(
            network,
            X_train,
            y_train,
            X_valid,
            y_valid,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            one_hot_encode=one_hot_encode,
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
        )

        save_model(MODEL_PATH, best_network, means, stds)
        save_history(HISTORY_PATH, history)
        plot_loss(history, LOSS_PLOT_PATH)
        plot_accuracy(history, ACCURACY_PLOT_PATH)


        weights, biases = extract_weights_and_biases(best_network)
        probabilities, y_pred = evaluate_for_visualization(best_network, X_valid)
        metrics = compute_classification_metrics(y_valid, y_pred, positive_class=1)

        final_metrics = {
            "train_loss": history["loss"][-1],
            "train_accuracy": history["accuracy"][-1],
            "val_loss": history["val_loss"][-1],
            "val_accuracy": history["val_accuracy"][-1],
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
        }
        save_metrics(METRICS_PATH, final_metrics)

        print("\nValidation classification metrics:")
        print(f"accuracy:  {metrics['accuracy']:.4f}")
        print(f"precision: {metrics['precision']:.4f}")
        print(f"recall:    {metrics['recall']:.4f}")
        print(f"f1-score:  {metrics['f1']:.4f}")
        print(f"tp: {metrics['tp']} - tn: {metrics['tn']} - fp: {metrics['fp']} - fn: {metrics['fn']}")

        feature_names = [f"f{i}" for i in range(len(X_valid[0]))]
        class_names = ["B", "M"]

        generate_all_visualizations(
            history=history,
            weights=weights,
            biases=biases,
            x=X_valid,
            y_true=y_valid,
            y_pred=y_pred,
            probabilities=probabilities,
            sample=X_valid[0],
            feature_names=feature_names,
            class_names=class_names,
            hidden_activation="sigmoid",
            output_activation="softmax",
            output_dir="visualizations"
        )


        print(f"\nModel saved to {MODEL_PATH}")
        print(f"Loss curve saved to {LOSS_PLOT_PATH}")
        print(f"Accuracy curve saved to {ACCURACY_PLOT_PATH}")
        print(f"History saved to {HISTORY_PATH}")

        print("\nFinal metrics:")
        print(f"loss: {history['loss'][-1]:.4f}")
        print(f"accuracy: {history['accuracy'][-1]:.4f}")
        print(f"val_loss: {history['val_loss'][-1]:.4f}")
        print(f"val_accuracy: {history['val_accuracy'][-1]:.4f}")

        print(f"best_epoch: {history['best_epoch']}")
        print(f"best_val_loss: {history['best_val_loss']:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()