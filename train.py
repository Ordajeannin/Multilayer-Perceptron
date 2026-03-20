import csv
import math
import random
import sys


def load_dataset(path):
    X = []
    y = []

    with open(path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            if len(row) < 3:
                continue

            label = row[1]
            features = row[2:]

            X.append([float(value) for value in features])

            if label == "M":
                y.append(1)
            elif label == "B":
                y.append(0)
            else:
                raise ValueError(f"Unknown label: {label}")

    return X, y


def compute_mean(values):
    return sum(values) / len(values)


def compute_std(values, mean):
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def compute_normalization_stats(X):
    if not X:
        raise ValueError("Empty dataset")

    n_features = len(X[0])
    means = []
    stds = []

    for j in range(n_features):
        column = [row[j] for row in X]
        mean = compute_mean(column)
        std = compute_std(column, mean)
        means.append(mean)
        stds.append(std)

    return means, stds


def normalize_dataset(X, means, stds):
    X_normalized = []

    for row in X:
        normalized_row = []

        for j in range(len(row)):
            if stds[j] == 0:
                normalized_value = 0.0
            else:
                normalized_value = (row[j] - means[j]) / stds[j]

            normalized_row.append(normalized_value)

        X_normalized.append(normalized_row)

    return X_normalized


def print_dataset_info(X_train, y_train, X_valid, y_valid):
    print(f"x_train shape: ({len(X_train)}, {len(X_train[0])})")
    print(f"y_train shape: ({len(y_train)},)")
    print(f"x_valid shape: ({len(X_valid)}, {len(X_valid[0])})")
    print(f"y_valid shape: ({len(y_valid)},)")


def initialize_layer(input_size, output_size):
    limit = math.sqrt(6 / (input_size + output_size))

    weights = []
    for _ in range(output_size):
        neuron_weights = []
        for _ in range(input_size):
            neuron_weights.append(random.uniform(-limit, limit))
        weights.append(neuron_weights)

    biases = [0.0 for _ in range(output_size)]

    return weights, biases


def initialize_network(input_size, hidden_sizes, output_size):
    layer_sizes = [input_size] + hidden_sizes + [output_size]

    network = []

    for i in range(len(layer_sizes) - 1):
        weights, biases = initialize_layer(layer_sizes[i], layer_sizes[i + 1])
        network.append({
            "weights": weights,
            "biases": biases
        })

    return network


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def softmax(values):
    max_value = max(values)
    exp_values = [math.exp(v - max_value) for v in values]
    total = sum(exp_values)
    return [v / total for v in exp_values]


def compute_layer_output(inputs, weights, biases, activation):
    outputs = []

    for neuron_index in range(len(weights)):
        weighted_sum = biases[neuron_index]

        for input_index in range(len(inputs)):
            weighted_sum += inputs[input_index] * weights[neuron_index][input_index]

        if activation == "sigmoid":
            outputs.append(sigmoid(weighted_sum))
        elif activation == "softmax":
            outputs.append(weighted_sum)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    if activation == "softmax":
        return softmax(outputs)

    return outputs


def forward_sample(network, x):
    a0 = x

    a1 = compute_layer_output(
        a0,
        network[0]["weights"],
        network[0]["biases"],
        "sigmoid"
    )

    a2 = compute_layer_output(
        a1,
        network[1]["weights"],
        network[1]["biases"],
        "sigmoid"
    )

    a3 = compute_layer_output(
        a2,
        network[2]["weights"],
        network[2]["biases"],
        "softmax"
    )

    cache = {
        "a0": a0,
        "a1": a1,
        "a2": a2,
        "a3": a3
    }

    return a3, cache


def forward_dataset(network, X):
    predictions = []

    for x in X:
        y_hat, _ = forward_sample(network, x)
        predictions.append(y_hat)

    return predictions


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

        print_dataset_info(X_train, y_train, X_valid, y_valid)

        input_size = len(X_train[0])
        hidden_sizes = [16, 16]
        output_size = 2

        network = initialize_network(input_size, hidden_sizes, output_size)

        print("\nNetwork initialized:")
        print(f"Input size : {input_size}")
        print(f"Hidden     : {hidden_sizes}")
        print(f"Output size: {output_size}")

        first_prediction, cache = forward_sample(network, X_train[0])

        print("\nFirst train sample forward pass:")
        print(f"a1 size: {len(cache['a1'])}")
        print(f"a2 size: {len(cache['a2'])}")
        print(f"output: {first_prediction}")
        print(f"sum(output): {sum(first_prediction)}")

        all_predictions = forward_dataset(network, X_train[:5])

        print("\nFirst 5 predictions:")
        for i, pred in enumerate(all_predictions):
            print(f"sample {i}: {pred}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()