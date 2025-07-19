import json
import pickle
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Load hyperparameters
with open('config/config.json', 'r') as f:
    config = json.load(f)

# Create and train model
model = LogisticRegression(
    C=config['C'],
    solver=config['solver'],
    max_iter=config['max_iter']
)
model.fit(X, y)

# Save trained model
with open('model_train.pkl', 'wb') as f:
    pickle.dump(model, f)

