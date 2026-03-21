import json
import sys


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_best_model_from_paths(paths):
    cv_results_list = [load_json(path) for path in paths]
    return find_best_model(cv_results_list)


def find_best_model(cv_results_list):
    """
    cv_results_list: liste de dicts chargés depuis les cv_results_*.json

    Critères de tri :
    1. F1 moyen le plus élevé
    2. Recall moyen le plus élevé
    3. Best val_loss moyen le plus faible
    4. Stabilité F1 (std la plus faible)
    5. Stabilité recall (std la plus faible)
    """

    if not cv_results_list:
        raise ValueError("cv_results_list cannot be empty")

    def ranking_key(result):
        mean = result["mean"]
        std = result["std"]

        return (
            -mean["f1"],            # plus grand = mieux
            -mean["recall"],        # plus grand = mieux
            mean["best_val_loss"],  # plus petit = mieux
            std["f1"],              # plus petit = mieux
            std["recall"],          # plus petit = mieux
        )

    sorted_results = sorted(cv_results_list, key=ranking_key)
    best_result = sorted_results[0]

    return best_result, sorted_results


def main():
    try:
        best_model, ranked_models = find_best_model_from_paths([
            "files/8_8_lr001/cv_results_8_8_lr001.json",
            "files/16_8_lr001/cv_results_16_8_lr001.json",
            "files/16_16_lr001/cv_results_16_16_lr001.json",
            "files/32_32_lr001/cv_results_32_32_lr001.json",
        ])

        print(f"Best model: {best_model['run_name']}\n")

        print("\n📊 Model ranking:\n")

        header = f"{'Rank':<5} {'Model':<15} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'ValLoss':<10} {'StdF1':<8} {'StdRec':<8}"
        print(header)
        print("-" * len(header))

        for i, model in enumerate(ranked_models, start=1):
            print(
                f"{i:<5} "
                f"{model['run_name']:<15} "
                f"{model['mean']['accuracy']:<8.4f} "
                f"{model['mean']['precision']:<8.4f} "
                f"{model['mean']['recall']:<8.4f} "
                f"{model['mean']['f1']:<8.4f} "
                f"{model['mean']['best_val_loss']:<10.4f} "
                f"{model['std']['f1']:<8.4f} "
                f"{model['std']['recall']:<8.4f}"
            )

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()