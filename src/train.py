import json
import pickle
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

def load_config(config_path='config/config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)

def train_model(X, y, config):
    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X, y)
    return model

if __name__ == "__main__":
    # Load data and config
    digits = load_digits()
    X, y = digits.data, digits.target
    config = load_config()

    # Train model
    model = train_model(X, y, config)

    # Save model artifact
    with open("model_train.pkl", "wb") as f:
        pickle.dump(model, f)

    # Evaluate and print performance metrics
    y_pred = model.predict(X)
    print("Accuracy:", accuracy_score(y, y_pred))
    print("F1 Score:", f1_score(y, y_pred, average="macro"))
    print("Log Loss:", log_loss(y, model.predict_proba(X)))

