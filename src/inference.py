import joblib
from sklearn.datasets import load_digits

def run_inference():
    # Load model
    model = joblib.load("model_train.pkl")
    
    # Load digits data
    digits = load_digits()
    X = digits.data
    y_true = digits.target

    # Predict
    y_pred = model.predict(X)

    # Show sample predictions
    print("Sample Predictions:")
    for i in range(10):
        print(f"Actual: {y_true[i]}  |  Predicted: {y_pred[i]}")

if __name__ == "__main__":
    run_inference()

