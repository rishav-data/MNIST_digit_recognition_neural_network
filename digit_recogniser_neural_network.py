import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            predictions = get_predictions(A2)
            acc = get_accuracy(predictions, Y)
            print(f"Iteration {i} - Accuracy: {acc:.4f}")
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)

def test_prediction(X, label, W1, b1, W2, b2):
    prediction = make_predictions(X, W1, b1, W2, b2)
    print("Prediction:", prediction[0])
    print("Label:     ", label)
    current_image = X.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def save_model(W1, b1, W2, b2, filename="model_weights.npz"):
    print("Saving model to:", filename)
    np.savez(filename, W1=W1, b1=b1, W2=W2, b2=b2)

def load_model(filename="model_weights.npz"):
    data = np.load(filename)
    return data["W1"], data["b1"], data["W2"], data["b2"]

# ✅ Optional retraining block (only works locally if train.csv exists)
def retrain_if_needed():
    csv_path = os.path.join(script_dir, "train.csv")
    if not os.path.exists(csv_path):
        print("⚠️ train.csv not found. Skipping training.")
        return

    print("📊 Loading and training on train.csv...")
    data = pd.read_csv(csv_path).values
    np.random.shuffle(data)
    data = data.T
    Y_train = data[0]
    X_train = data[1:] / 255.

    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.1, iterations=500)
    save_model(W1, b1, W2, b2)

# Run if needed
if __name__ == "__main__":
    model_path = os.path.join(script_dir, "model_weights.npz")
    if os.path.exists(model_path):
        print("✅ Loading saved model...")
        W1, b1, W2, b2 = load_model(model_path)

        # Optional test input
        input_csv = os.path.join(script_dir, "input.csv")
        if os.path.exists(input_csv):
            x = pd.read_csv(input_csv, header=None).values.T
            prediction = make_predictions(x, W1, b1, W2, b2)[0]
            print("🧠 Predicted Digit:", prediction)
        else:
            print("ℹ️ No input.csv found to test.")
    else:
        retrain_if_needed()
