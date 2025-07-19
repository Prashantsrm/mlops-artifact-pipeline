import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import json
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from src.train import load_config, train_model

def test_config_loads():
    cfg = load_config("config/config.json")
    for key, t in [("C", float), ("solver", str), ("max_iter", int)]:
        assert key in cfg
        assert isinstance(cfg[key], t)

def test_train_returns_model():
    digits = load_digits()
    X, y = digits.data, digits.target
    cfg = {"C":1.0, "solver":"lbfgs", "max_iter":100}
    model = train_model(X, y, cfg)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")

def test_accuracy_above_threshold():
    digits = load_digits()
    X, y = digits.data, digits.target
    cfg = {"C":1.0, "solver":"lbfgs", "max_iter":100}
    model = train_model(X, y, cfg)
    acc = model.score(X, y)
    assert acc >= 0.9  # You can tweak threshold if needed

