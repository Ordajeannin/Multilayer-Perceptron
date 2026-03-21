import json
import sys
import os


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_arg(arg):
    # format: label=path
    if "=" not in arg:
        raise ValueError(f"Invalid argument: {arg}")
    label, path = arg.split("=", 1)
    return label, path


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python3 compare_metrics.py model=path/to/metrics.json ...")
        sys.exit(1)

    rows = []

    for arg in sys.argv[1:]:
        label, path = parse_arg(arg)

        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(1)

        m = load_metrics(path)

        rows.append({
            "model": label,
            "accuracy": m.get("val_accuracy", m.get("accuracy")),
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "val_loss": m.get("val_loss"),
        })

    # tri par f1 décroissant
    rows.sort(key=lambda x: x["f1"], reverse=True)

    # affichage
    print("\n📊 Model comparison:\n")
    print(f"{'model':<12} {'acc':<8} {'prec':<8} {'recall':<8} {'f1':<8} {'val_loss':<10}")
    print("-" * 60)

    for r in rows:
        print(
            f"{r['model']:<12} "
            f"{r['accuracy']:.4f}   "
            f"{r['precision']:.4f}   "
            f"{r['recall']:.4f}   "
            f"{r['f1']:.4f}   "
            f"{r['val_loss']:.4f}"
        )


if __name__ == "__main__":
    main()