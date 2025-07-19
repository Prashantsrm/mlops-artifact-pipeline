import json
import pickle
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

def load_config(path="config/config.json"):
    with open(path, "r") as f:
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
    model = train_model(X, y, config)

    # SAVE YOUR MODEL HERE:
    with open("model_train.pkl", "wb") as f:
        pickle.dump(model, f)

    # (Optional: print metrics for assignment)
    from sklearn.metrics import accuracy_score, f1_score, log_loss
    y_pred = model.predict(X)
    print("Accuracy:", accuracy_score(y, y_pred))
    print("F1 Score:", f1_score(y, y_pred, average="macro"))
    print("Log Loss:", log_loss(y, model.predict_proba(X)))

