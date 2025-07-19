import json
import pickle
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

if __name__ == "__main__":
    digits = load_digits()
    X, y = digits.data, digits.target

    with open('config/config.json', 'r') as f:
        config = json.load(f)

    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X, y)
    with open("model_train.pkl", "wb") as f:
        pickle.dump(model, f)

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    loss = log_loss(y, model.predict_proba(X))

    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Log Loss:", loss)

