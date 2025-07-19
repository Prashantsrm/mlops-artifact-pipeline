import pickle
from sklearn.datasets import load_digits

if __name__ == "__main__":
    # Load model
    with open("model_train.pkl", "rb") as f:
        model = pickle.load(f)
    # Load data
    digits = load_digits()
    X, y = digits.data, digits.target
    # Predict
    preds = model.predict(X)
    print("Sample predictions:", preds[:10])


