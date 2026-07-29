"""
Program 4: Artificial Neural Network with Backpropagation
Implements a simple feed-forward neural network (1 hidden layer) trained
with the backpropagation algorithm, from scratch using numpy.

Dataset: hours of sleep / hours of study -> marks (classic ANN lab example)
"""

import numpy as np

# Input: [Sleep, Study] hours (normalized), Output: marks (normalized 0-1)
X = np.array([[2, 9], [1, 5], [3, 6]], dtype=float)
y = np.array([[92], [86], [89]], dtype=float)

# Normalize
X = X / np.amax(X, axis=0)
y = y / 100.0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        np.random.seed(1)
        self.lr = lr
        self.W1 = np.random.uniform(size=(input_size, hidden_size))
        self.b1 = np.random.uniform(size=(1, hidden_size))
        self.W2 = np.random.uniform(size=(hidden_size, output_size))
        self.b2 = np.random.uniform(size=(1, output_size))

    def forward(self, X):
        self.hidden_in = np.dot(X, self.W1) + self.b1
        self.hidden_out = sigmoid(self.hidden_in)
        self.final_in = np.dot(self.hidden_out, self.W2) + self.b2
        self.final_out = sigmoid(self.final_in)
        return self.final_out

    def backward(self, X, y, output):
        error = y - output
        d_output = error * sigmoid_derivative(output)

        error_hidden = d_output.dot(self.W2.T)
        d_hidden = error_hidden * sigmoid_derivative(self.hidden_out)

        self.W2 += self.hidden_out.T.dot(d_output) * self.lr
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.lr
        self.W1 += X.T.dot(d_hidden) * self.lr
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.lr

    def train(self, X, y, epochs=5000, verbose_every=1000):
        for epoch in range(1, epochs + 1):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % verbose_every == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.6f}")


if __name__ == "__main__":
    nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1, lr=0.5)
    nn.train(X, y, epochs=5000, verbose_every=1000)

    predicted = nn.forward(X)
    print("\nInput (normalized):\n", X)
    print("Actual Output (normalized):\n", y)
    print("Predicted Output (normalized):\n", predicted)
    print("\nPredicted marks (rescaled to 0-100):\n", predicted * 100)
