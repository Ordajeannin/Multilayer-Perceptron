import json
import math
import random
import os
import json


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


def copy_network(network):
    copied = []

    for layer in network:
        copied.append({
            "weights": [row[:] for row in layer["weights"]],
            "biases": layer["biases"][:]
        })

    return copied


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



def compute_classification_metrics(y_true, y_pred, positive_class=1):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for true, pred in zip(y_true, y_pred):
        if true == positive_class and pred == positive_class:
            tp += 1
        elif true != positive_class and pred != positive_class:
            tn += 1
        elif true != positive_class and pred == positive_class:
            fp += 1
        elif true == positive_class and pred != positive_class:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def forward_sample(network, x):
    z1, a1 = compute_layer_output(
        x, network[0]["weights"], network[0]["biases"], "sigmoid"
    )
    z2, a2 = compute_layer_output(
        a1, network[1]["weights"], network[1]["biases"], "sigmoid"
    )
    z3, a3 = compute_layer_output(
        a2, network[2]["weights"], network[2]["biases"], "softmax"
    )

    cache = {
        "a0": x,
        "a1": a1,
        "a2": a2,
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


def evaluate_dataset(network, X, y, one_hot_encode):
    total_loss = 0.0
    correct = 0

    for i in range(len(X)):
        y_pred, _ = forward_sample(network, X[i])
        y_true = one_hot_encode(y[i])

        total_loss += compute_loss(y_true, y_pred)

        if predict_class(y_pred) == y[i]:
            correct += 1

    avg_loss = total_loss / len(X)
    accuracy = correct / len(X)

    return avg_loss, accuracy


def compute_gradients(network, cache, y_true):
    a0 = cache["a0"]
    a1 = cache["a1"]
    a2 = cache["a2"]
    a3 = cache["a3"]

    delta3 = [a3[i] - y_true[i] for i in range(len(a3))]

    delta2 = []
    for j in range(len(a2)):
        weighted_error = 0.0
        for k in range(len(delta3)):
            weighted_error += network[2]["weights"][k][j] * delta3[k]
        delta2.append(weighted_error * sigmoid_derivative_from_activation(a2[j]))

    delta1 = []
    for j in range(len(a1)):
        weighted_error = 0.0
        for k in range(len(delta2)):
            weighted_error += network[1]["weights"][k][j] * delta2[k]
        delta1.append(weighted_error * sigmoid_derivative_from_activation(a1[j]))

    gradients = {
        "dW3": [],
        "db3": delta3[:],
        "dW2": [],
        "db2": delta2[:],
        "dW1": [],
        "db1": delta1[:]
    }

    for neuron_index in range(len(delta3)):
        gradients["dW3"].append([delta3[neuron_index] * a2[i] for i in range(len(a2))])

    for neuron_index in range(len(delta2)):
        gradients["dW2"].append([delta2[neuron_index] * a1[i] for i in range(len(a1))])

    for neuron_index in range(len(delta1)):
        gradients["dW1"].append([delta1[neuron_index] * a0[i] for i in range(len(a0))])

    return gradients


def apply_gradients(network, gradients, learning_rate):
    for neuron_index in range(len(network[2]["weights"])):
        for input_index in range(len(network[2]["weights"][neuron_index])):
            network[2]["weights"][neuron_index][input_index] -= (
                learning_rate * gradients["dW3"][neuron_index][input_index]
            )
        network[2]["biases"][neuron_index] -= learning_rate * gradients["db3"][neuron_index]

    for neuron_index in range(len(network[1]["weights"])):
        for input_index in range(len(network[1]["weights"][neuron_index])):
            network[1]["weights"][neuron_index][input_index] -= (
                learning_rate * gradients["dW2"][neuron_index][input_index]
            )
        network[1]["biases"][neuron_index] -= learning_rate * gradients["db2"][neuron_index]

    for neuron_index in range(len(network[0]["weights"])):
        for input_index in range(len(network[0]["weights"][neuron_index])):
            network[0]["weights"][neuron_index][input_index] -= (
                learning_rate * gradients["dW1"][neuron_index][input_index]
            )
        network[0]["biases"][neuron_index] -= learning_rate * gradients["db1"][neuron_index]


def train_model(
    network,
    X_train,
    y_train,
    X_valid,
    y_valid,
    epochs,
    learning_rate,
    one_hot_encode,
    patience=10,
    min_delta=1e-4
):
    history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": []
    }

    epochs_without_improvement = 0
    best_val_loss = float("inf")
    best_network = copy_network(network)
    best_epoch = 0

    for epoch in range(epochs):
        indices = list(range(len(X_train)))
        random.shuffle(indices)

        for i in indices:
            y_pred, cache = forward_sample(network, X_train[i])
            y_true = one_hot_encode(y_train[i])
            gradients = compute_gradients(network, cache, y_true)
            apply_gradients(network, gradients, learning_rate)

        train_loss, train_acc = evaluate_dataset(network, X_train, y_train, one_hot_encode)
        valid_loss, valid_acc = evaluate_dataset(network, X_valid, y_valid, one_hot_encode)

        history["loss"].append(train_loss)
        history["val_loss"].append(valid_loss)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(valid_acc)

        if valid_loss < best_val_loss - min_delta:
            best_val_loss = valid_loss
            best_network = copy_network(network)
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

        print(
            f"epoch {epoch + 1:03d}/{epochs} - "
            f"loss: {train_loss:.4f} - "
            f"accuracy: {train_acc:.4f} - "
            f"val_loss: {valid_loss:.4f} - "
            f"val_accuracy: {valid_acc:.4f}"
        )

    print(f"\nBest validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss

    return history, best_network


def save_model(path, network, means, stds):
    model_data = {
        "network": network,
        "means": means,
        "stds": stds
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(model_data, file)


def save_history(path, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


def save_metrics(path, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

def save_json(path, data):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

def load_model(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "r", encoding="utf-8") as file:
        model_data = json.load(file)

    return model_data["network"], model_data["means"], model_data["stds"]

def compute_mean(values):
    return sum(values) / len(values)


def compute_std(values, mean):
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)