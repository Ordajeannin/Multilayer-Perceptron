import sys

from data_utils import (
    load_dataset,
    normalize_dataset,
    label_to_text,
)
from model_utils import (
    load_model,
    forward_sample,
    predict_class,
)


def evaluate_predictions(network, X, y):
    correct = 0
    predictions = []

    for i in range(len(X)):
        probabilities, _ = forward_sample(network, X[i])
        predicted_label = predict_class(probabilities)
        true_label = y[i]

        if predicted_label == true_label:
            correct += 1

        predictions.append({
            "index": i,
            "true": label_to_text(true_label),
            "pred": label_to_text(predicted_label),
            "prob_B": probabilities[0],
            "prob_M": probabilities[1]
        })

    accuracy = correct / len(X)
    return accuracy, predictions


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 predict.py <model.json> <dataset.csv>")
        sys.exit(1)

    model_path = sys.argv[1]
    dataset_path = sys.argv[2]

    try:
        network, means, stds = load_model(model_path)

        X, y = load_dataset(dataset_path)
        X = normalize_dataset(X, means, stds)

        accuracy, predictions = evaluate_predictions(network, X, y)

        print(f"Dataset shape: ({len(X)}, {len(X[0])})")
        print(f"Accuracy: {accuracy:.4f}")

        print("\nFirst 10 predictions:")
        for prediction in predictions[:10]:
            print(
                f"sample {prediction['index']:03d} - "
                f"true: {prediction['true']} - "
                f"pred: {prediction['pred']} - "
                f"prob_B: {prediction['prob_B']:.4f} - "
                f"prob_M: {prediction['prob_M']:.4f}"
            )

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()