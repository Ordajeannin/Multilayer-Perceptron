import csv
import math
import random

# charger le dataset à partir d'un fichier CSV
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


# calculer la moyenne 
def compute_mean(values):
    return sum(values) / len(values)

# calculer l'écart type
def compute_std(values, mean):
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

# calculer les statistiques de normalisation (moyennes et écarts types) pour chaque colonne du dataset
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

# normaliser le dataset en utilisant les moyennes et écarts types calculés
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

# encoder les labels en one-hot encoding
def one_hot_encode(label):
    if label == 0:
        return [1.0, 0.0]
    return [0.0, 1.0]

# décoder les labels à partir du one-hot encoding
# (one-hot encoding = [1.0, 0.0] pour "B" et [0.0, 1.0] pour "M")
def label_to_text(label):
    if label == 1:
        return "M"
    return "B"

# permet de diviser le dataset en k folds pour la validation croisée, en mélangeant les indices des échantillons de manière aléatoire avec une seed fixe pour la reproductibilité
def make_k_folds(X, y, k=5, seed=42):
    indices = list(range(len(X)))

    rng = random.Random(seed)
    rng.shuffle(indices)

    fold_sizes = [len(X) // k] * k
    for i in range(len(X) % k):
        fold_sizes[i] += 1

    folds = []
    start = 0
    for fold_size in fold_sizes:
        fold_indices = indices[start:start + fold_size]
        start += fold_size

        X_fold = [X[i] for i in fold_indices]
        y_fold = [y[i] for i in fold_indices]
        folds.append((X_fold, y_fold))

    return folds

# permet de fusionner les k folds en un ensemble d'entraînement et un ensemble de validation,
# en utilisant le fold spécifié par valid_index comme ensemble de validation et les autres folds comme ensemble d'entraînement
def merge_folds(folds, valid_index):
    X_train = []
    y_train = []
    X_valid, y_valid = folds[valid_index]

    for i, (X_fold, y_fold) in enumerate(folds):
        if i == valid_index:
            continue
        X_train.extend(X_fold)
        y_train.extend(y_fold)

    return X_train, y_train, X_valid, y_valid