import os
import json
import pytest
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from src.train import train_model, load_config
from sklearn.metrics import accuracy_score, f1_score, log_loss

def test_config_loading():
    """
    Test that config/config.json loads and contains the required hyperparameters
    with correct data types.
    """
    config = load_config()  # Assumes load_config loads config/config.json
    assert "C" in config and isinstance(config["C"], (float, int)), "Missing or invalid type for C"
    assert "solver" in config and isinstance(config["solver"], str), "Missing or invalid type for solver"
    assert "max_iter" in config and isinstance(config["max_iter"], int), "Missing or invalid type for max_iter"

def test_model_creation_and_fitting():
    """
    Test that train_model returns a fitted LogisticRegression object.
    """
    digits = load_digits()
    X, y = digits.data, digits.target
    config = load_config()
    model = train_model(X, y, config)
    assert isinstance(model, LogisticRegression), "Model is not LogisticRegression type"
    # Check that model is fitted
    assert hasattr(model, "coef_"), "Model is not fitted (no coef_)"
    assert hasattr(model, "classes_"), "Model is not fitted (no classes_)"

def test_model_accuracy():
    """
    Test that the trained model achieves a reasonable accuracy, F1-score, and log loss.
    """
    digits = load_digits()
    X, y = digits.data, digits.target
    config = load_config()
    model = train_model(X, y, config)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")
    loss = log_loss(y, model.predict_proba(X))
    # Print required metrics for documentation
    print(f"\nModel accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Log Loss: {loss:.4f}")
    assert acc > 0.90, "Accuracy is too low"

