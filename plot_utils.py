import json
import os
import matplotlib.pyplot as plt


def plot_loss(history, path):
    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve - Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_accuracy(history, path):
    epochs = range(1, len(history["accuracy"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["accuracy"], label="train accuracy")
    plt.plot(epochs, history["val_accuracy"], label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve - Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def load_history(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def plot_multiple_histories(histories, metric_key, path, title=None):
    """
    histories = [
        {"label": "16-16 lr=0.01", "history": {...}},
        {"label": "32-16 lr=0.01", "history": {...}},
    ]
    """

    plt.figure(figsize=(9, 5))

    for item in histories:
        label = item["label"]
        history = item["history"]

        if metric_key not in history:
            continue

        epochs = range(1, len(history[metric_key]) + 1)
        plt.plot(epochs, history[metric_key], label=label)

    plt.xlabel("Epoch")
    plt.ylabel(metric_key.replace("_", " ").title())
    plt.title(title if title else f"Comparison - {metric_key}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()