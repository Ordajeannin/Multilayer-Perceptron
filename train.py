import random
import sys

from config import (
    LEARNING_RATE,
    EPOCHS,
    HIDDEN_SIZES,
    MODEL_PATH,
    LOSS_PLOT_PATH,
    ACCURACY_PLOT_PATH,
)
from data_utils import (
    load_dataset,
    compute_normalization_stats,
    normalize_dataset,
    one_hot_encode,
)
from model_utils import (
    initialize_network,
    train_model,
    save_model,
)
from plot_utils import (
    plot_loss,
    plot_accuracy,
)


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
            one_hot_encode=one_hot_encode
        )

        save_model(MODEL_PATH, best_network, means, stds)
        plot_loss(history, LOSS_PLOT_PATH)
        plot_accuracy(history, ACCURACY_PLOT_PATH)

        print(f"\nModel saved to {MODEL_PATH}")
        print(f"Loss curve saved to {LOSS_PLOT_PATH}")
        print(f"Accuracy curve saved to {ACCURACY_PLOT_PATH}")

        print("\nFinal metrics:")
        print(f"loss: {history['loss'][-1]:.4f}")
        print(f"accuracy: {history['accuracy'][-1]:.4f}")
        print(f"val_loss: {history['val_loss'][-1]:.4f}")
        print(f"val_accuracy: {history['val_accuracy'][-1]:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()