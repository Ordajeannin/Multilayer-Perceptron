import csv
import math


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

    means = []
    stds = []

    for j in range(len(X[0])):
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


def label_to_text(label):
    if label == 1:
        return "M"
    return "B"