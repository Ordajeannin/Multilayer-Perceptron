import csv
import random
import sys


def load_dataset(path):
    data = []

    with open(path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            if len(row) < 3:
                continue

            sample_id = row[0]
            label = row[1]
            features = row[2:]

            data.append({
                "id": sample_id,
                "label": label,
                "features": features
            })

    return data


def split_dataset(rows, train_ratio=0.8, seed=42):
    random.seed(seed)
    shuffled = rows[:]
    random.shuffle(shuffled)

    split_index = int(len(shuffled) * train_ratio)

    train_rows = shuffled[:split_index]
    valid_rows = shuffled[split_index:]

    return train_rows, valid_rows


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to save in {path}")

    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        for row in rows:
            writer.writerow(
                [row["id"], row["label"]] + row["features"]
            )


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 split.py <input.csv> <train.csv> <valid.csv>")
        sys.exit(1)

    input_path = sys.argv[1]
    train_path = sys.argv[2]
    valid_path = sys.argv[3]

    try:
        rows = load_dataset(input_path)

        train_rows, valid_rows = split_dataset(rows)

        save_csv(train_path, train_rows)
        save_csv(valid_path, valid_rows)

        print(f"Total samples : {len(rows)}")
        print(f"Training set  : {len(train_rows)}")
        print(f"Validation set: {len(valid_rows)}")
        print("Split done successfully.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()