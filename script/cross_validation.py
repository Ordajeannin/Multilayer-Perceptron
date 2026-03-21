import random

from config.config import (
    RUN_NAME,
    HIDDEN_SIZES,
    LEARNING_RATE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    CV_FOLDS,
    CV_SEED,
    CV_RESULTS_PATH,
    DATASET_PATH
)
from utils.data_utils import (
    load_dataset,
    compute_normalization_stats,
    normalize_dataset,
    one_hot_encode,
    make_k_folds,
    merge_folds,
)
from utils.model_utils import (
    initialize_network,
    train_model,
    forward_sample,
    predict_class,
    compute_classification_metrics,
    compute_mean,
    compute_std,
    save_json,
)

# permet d'évaluer les prédictions du réseau sur un ensemble de données,
# en calculant les probabilités prédites pour chaque échantillon
# et en déterminant la classe prédite à partir de ces probabilités
def evaluate_predictions(network, X):
    y_pred = []

    for sample in X:
        probabilities, _ = forward_sample(network, sample)
        predicted_label = predict_class(probabilities)
        y_pred.append(predicted_label)

    return y_pred

# permet de résumer les résultats des différents folds en calculant la moyenne et l'écart-type pour chaque métrique d'évaluation 
# accuracy, precision, recall, f1, val_loss, best_val_loss, best_epoch) à partir des résultats individuels de chaque fold
def summarize_folds(fold_results):
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "val_loss",
        "best_val_loss",
        "best_epoch",
    ]

    summary_mean = {}
    summary_std = {}

    for metric_name in metric_names:
        values = [fold_result[metric_name] for fold_result in fold_results]
        mean_value = compute_mean(values)
        std_value = compute_std(values, mean_value)

        summary_mean[metric_name] = mean_value
        summary_std[metric_name] = std_value

    return summary_mean, summary_std

# point d'entrée du script de validation croisée, qui charge le dataset, crée les folds, entraîne et évalue le modèle pour chaque fold, et résume les résultats
def main():
    import sys

    if len(sys.argv) != 1:
        print("Usage: python3 cross_validation.py")
        sys.exit(1)

    dataset_path = DATASET_PATH

    try:
        random.seed(CV_SEED)

        X, y = load_dataset(dataset_path)
        folds = make_k_folds(X, y, k=CV_FOLDS, seed=CV_SEED)

        fold_results = []

        print(f"Run name: {RUN_NAME}")
        print(f"Hidden sizes: {HIDDEN_SIZES}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Epochs: {EPOCHS}")
        print(f"Cross-validation folds: {CV_FOLDS}")

        for fold_index in range(CV_FOLDS):
            print(f"\n--- Fold {fold_index + 1}/{CV_FOLDS} ---")

            X_train, y_train, X_valid, y_valid = merge_folds(folds, fold_index)

            means, stds = compute_normalization_stats(X_train)
            X_train = normalize_dataset(X_train, means, stds)
            X_valid = normalize_dataset(X_valid, means, stds)

            network = initialize_network(
                input_size=len(X_train[0]),
                hidden_sizes=HIDDEN_SIZES,
                output_size=2,
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

            y_pred = evaluate_predictions(best_network, X_valid)
            metrics = compute_classification_metrics(y_valid, y_pred, positive_class=1)

            fold_result = {
                "fold": fold_index + 1,
                "train_size": len(X_train),
                "valid_size": len(X_valid),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "val_loss": history["val_loss"][-1],
                "val_accuracy": history["val_accuracy"][-1],
                "best_val_loss": history["best_val_loss"],
                "best_epoch": history["best_epoch"],
            }

            fold_results.append(fold_result)

            print(
                f"fold {fold_index + 1} - "
                f"accuracy: {fold_result['accuracy']:.4f} - "
                f"precision: {fold_result['precision']:.4f} - "
                f"recall: {fold_result['recall']:.4f} - "
                f"f1: {fold_result['f1']:.4f} - "
                f"best_val_loss: {fold_result['best_val_loss']:.4f}"
            )

        mean_results, std_results = summarize_folds(fold_results)

        final_results = {
            "run_name": RUN_NAME,
            "hidden_sizes": HIDDEN_SIZES,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "cv_folds": CV_FOLDS,
            "cv_seed": CV_SEED,
            "fold_results": fold_results,
            "mean": mean_results,
            "std": std_results,
        }

        save_json(CV_RESULTS_PATH, final_results)

        print("\n=== Cross-validation summary ===")
        print(f"accuracy:      {mean_results['accuracy']:.4f} ± {std_results['accuracy']:.4f}")
        print(f"precision:     {mean_results['precision']:.4f} ± {std_results['precision']:.4f}")
        print(f"recall:        {mean_results['recall']:.4f} ± {std_results['recall']:.4f}")
        print(f"f1:            {mean_results['f1']:.4f} ± {std_results['f1']:.4f}")
        print(f"best_val_loss: {mean_results['best_val_loss']:.4f} ± {std_results['best_val_loss']:.4f}")
        print(f"best_epoch:    {mean_results['best_epoch']:.2f} ± {std_results['best_epoch']:.2f}")
        print(f"\nResults saved to {CV_RESULTS_PATH}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()