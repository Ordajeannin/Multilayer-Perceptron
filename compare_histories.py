import os
import sys

from plot_utils import load_history, plot_multiple_histories


def parse_history_argument(argument):
    """
    Format attendu :
    label=path/to/history.json

    Exemple :
    16-16=files/history_16_16.json
    """
    if "=" not in argument:
        raise ValueError(
            f"Invalid argument '{argument}'. Expected format: label=history.json"
        )

    label, path = argument.split("=", 1)
    return {"label": label, "path": path}


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print(
            "python3 compare_histories.py "
            "run1=files/history_run1.json run2=files/history_run2.json"
        )
        sys.exit(1)

    histories = []

    for arg in sys.argv[1:]:
        item = parse_history_argument(arg)

        if not os.path.exists(item["path"]):
            print(f"Error: file not found: {item['path']}")
            sys.exit(1)

        history = load_history(item["path"])
        histories.append({
            "label": item["label"],
            "history": history
        })

    os.makedirs("files", exist_ok=True)

    plot_multiple_histories(
        histories,
        metric_key="val_loss",
        path="files/compare_val_loss.png",
        title="Validation Loss Comparison"
    )

    plot_multiple_histories(
        histories,
        metric_key="val_accuracy",
        path="files/compare_val_accuracy.png",
        title="Validation Accuracy Comparison"
    )

    print("Comparison plots saved to:")
    print("- files/compare_val_loss.png")
    print("- files/compare_val_accuracy.png")


if __name__ == "__main__":
    main()