import json
import os
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import joblib


def load_config(config_path: str) -> dict:
    """
    Load hyperparameters from a JSON config file.

    Expected JSON keys:
      - "C" (float): inverse regularization strength
      - "solver" (str): optimization algorithm
      - "max_iter" (int): maximum number of iterations
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)
    return cfg


def train_model(X, y, cfg: dict) -> LogisticRegression:
    """
    Train a LogisticRegression model on (X, y) using parameters in cfg.
    """
    model = LogisticRegression(
        C=cfg["C"],
        solver=cfg["solver"],
        max_iter=cfg["max_iter"],
    )
    model.fit(X, y)
    return model


def main():
    # 1. Load hyperparameters
    # Assuming you're running this from the repo root:
    #   python src/train.py
    config_path = os.path.join("config", "config.json")
    cfg = load_config(config_path)

    # 2. Load the digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    # 3. Train the model
    model = train_model(X, y, cfg)

    # 4. Save the trained model
    output_path = "model_train.pkl"
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()

