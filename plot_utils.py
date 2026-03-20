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