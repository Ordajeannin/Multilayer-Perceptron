import csv
import json
import math
import random
import sys


LEARNING_RATE = 0.05
EPOCHS = 200


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


def one_hot_encode(label):
    if label == 0:
        return [1.0, 0.0]
    return [0.0, 1.0]


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


def sigmoid_derivative_from_activation(a):
    return a * (1.0 - a)


def softmax(values):
    max_value = max(values)
    exp_values = [math.exp(v - max_value) for v in values]
    total = sum(exp_values)
    return [v / total for v in exp_values]


def compute_layer_output(inputs, weights, biases, activation):
    z_values = []
    outputs = []

    for neuron_index in range(len(weights)):
        weighted_sum = biases[neuron_index]

        for input_index in range(len(inputs)):
            weighted_sum += inputs[input_index] * weights[neuron_index][input_index]

        z_values.append(weighted_sum)

        if activation == "sigmoid":
            outputs.append(sigmoid(weighted_sum))
        elif activation == "softmax":
            pass
        else:
            raise ValueError(f"Unknown activation: {activation}")

    if activation == "softmax":
        outputs = softmax(z_values)

    return z_values, outputs


def forward_sample(network, x):
    z1, a1 = compute_layer_output(
        x,
        network[0]["weights"],
        network[0]["biases"],
        "sigmoid"
    )

    z2, a2 = compute_layer_output(
        a1,
        network[1]["weights"],
        network[1]["biases"],
        "sigmoid"
    )

    z3, a3 = compute_layer_output(
        a2,
        network[2]["weights"],
        network[2]["biases"],
        "softmax"
    )

    cache = {
        "a0": x,
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "a2": a2,
        "z3": z3,
        "a3": a3
    }

    return a3, cache


def compute_loss(y_true, y_pred):
    epsilon = 1e-15
    total = 0.0

    for i in range(len(y_true)):
        total += y_true[i] * math.log(y_pred[i] + epsilon)

    return -total


def predict_class(probabilities):
    if probabilities[1] > probabilities[0]:
        return 1
    return 0


def evaluate_dataset(network, X, y):
    total_loss = 0.0
    correct = 0

    for i in range(len(X)):
        y_pred, _ = forward_sample(network, X[i])
        y_true = one_hot_encode(y[i])

        total_loss += compute_loss(y_true, y_pred)

        predicted_label = predict_class(y_pred)
        if predicted_label == y[i]:
            correct += 1

    avg_loss = total_loss / len(X)
    accuracy = correct / len(X)

    return avg_loss, accuracy


def backward_sample(network, cache, y_true, learning_rate):
    a0 = cache["a0"]
    a1 = cache["a1"]
    a2 = cache["a2"]
    a3 = cache["a3"]

    # Softmax + cross-entropy
    delta3 = []
    for i in range(len(a3)):
        delta3.append(a3[i] - y_true[i])

    # Layer 3 gradients
    for neuron_index in range(len(network[2]["weights"])):
        for input_index in range(len(network[2]["weights"][neuron_index])):
            gradient = delta3[neuron_index] * a2[input_index]
            network[2]["weights"][neuron_index][input_index] -= learning_rate * gradient

        network[2]["biases"][neuron_index] -= learning_rate * delta3[neuron_index]

    # Hidden layer 2 delta
    delta2 = []
    for j in range(len(a2)):
        weighted_error = 0.0
        for k in range(len(delta3)):
            weighted_error += network[2]["weights"][k][j] * delta3[k]

        delta2.append(weighted_error * sigmoid_derivative_from_activation(a2[j]))

    # Layer 2 gradients
    for neuron_index in range(len(network[1]["weights"])):
        for input_index in range(len(network[1]["weights"][neuron_index])):
            gradient = delta2[neuron_index] * a1[input_index]
            network[1]["weights"][neuron_index][input_index] -= learning_rate * gradient

        network[1]["biases"][neuron_index] -= learning_rate * delta2[neuron_index]

    # Hidden layer 1 delta
    delta1 = []
    for j in range(len(a1)):
        weighted_error = 0.0
        for k in range(len(delta2)):
            weighted_error += network[1]["weights"][k][j] * delta2[k]

        delta1.append(weighted_error * sigmoid_derivative_from_activation(a1[j]))

    # Layer 1 gradients
    for neuron_index in range(len(network[0]["weights"])):
        for input_index in range(len(network[0]["weights"][neuron_index])):
            gradient = delta1[neuron_index] * a0[input_index]
            network[0]["weights"][neuron_index][input_index] -= learning_rate * gradient

        network[0]["biases"][neuron_index] -= learning_rate * delta1[neuron_index]


def train(network, X_train, y_train, X_valid, y_valid, epochs, learning_rate):
    history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": []
    }

    for epoch in range(epochs):
        for i in range(len(X_train)):
            y_pred, cache = forward_sample(network, X_train[i])
            y_true = one_hot_encode(y_train[i])
            backward_sample(network, cache, y_true, learning_rate)

        train_loss, train_acc = evaluate_dataset(network, X_train, y_train)
        valid_loss, valid_acc = evaluate_dataset(network, X_valid, y_valid)

        history["loss"].append(train_loss)
        history["val_loss"].append(valid_loss)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(valid_acc)

        print(
            f"epoch {epoch + 1:03d}/{epochs} - "
            f"loss: {train_loss:.4f} - "
            f"accuracy: {train_acc:.4f} - "
            f"val_loss: {valid_loss:.4f} - "
            f"val_accuracy: {valid_acc:.4f}"
        )

    return history


def save_model(path, network, means, stds):
    model_data = {
        "network": network,
        "means": means,
        "stds": stds
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(model_data, file)


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
            hidden_sizes=[16, 16],
            output_size=2
        )

        history = train(
            network,
            X_train,
            y_train,
            X_valid,
            y_valid,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE
        )

        save_model("model.json", network, means, stds)
        print("\nModel saved to model.json")

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