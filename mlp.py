"""
mlp.py
------
A Multilayer Perceptron implemented entirely from first principles using
only NumPy: forward propagation, back-propagation (manual gradient
derivation), mini-batch gradient descent, L2 regularisation and early
stopping.

Architecture: Input(8) -> Hidden(ReLU) -> Hidden(ReLU, optional) -> Output(Sigmoid)

Activation-function justification:
  - ReLU is used in hidden layers because it avoids the vanishing-gradient
    problem that saturates sigmoid/tanh units for the input ranges seen
    after standardisation, and is cheap to differentiate (derivative is a
    step function), which speeds up back-propagation.
  - Sigmoid is used at the output layer because this is a binary
    classification task (disease risk vs. no risk) and sigmoid maps the
    single output neuron to a valid probability in (0, 1), which is
    required for the binary cross-entropy loss and for probabilistic
    fusion with the Naive Bayes posterior later on.
"""
import numpy as np


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_pred, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class MLP:
    """Feed-forward network with one or two hidden layers, trained with
    manual back-propagation (chain rule applied layer by layer)."""

    def __init__(self, layer_sizes, learning_rate=0.05, l2_lambda=1e-4,
                 random_state=42, init_weights=None):
        """
        layer_sizes: list, e.g. [8, 12, 1] (single hidden layer of 12
                     units) or [8, 16, 8, 1] (two hidden layers).
        init_weights: optional pre-set weight/bias list (used by the GA to
                     inject an optimised initial configuration).
        """
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.l2 = l2_lambda
        self.n_layers = len(layer_sizes) - 1
        rng = np.random.RandomState(random_state)

        if init_weights is not None:
            self.W, self.b = init_weights
        else:
            self.W, self.b = [], []
            for i in range(self.n_layers):
                fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
                # He initialisation, appropriate for ReLU hidden units.
                limit = np.sqrt(2.0 / fan_in)
                self.W.append(rng.randn(fan_in, fan_out) * limit)
                self.b.append(np.zeros((1, fan_out)))

    def get_weights_flat(self):
        """Flatten all weights/biases into a single vector (used by the GA
        as a chromosome representation)."""
        parts = [w.flatten() for w in self.W] + [b.flatten() for b in self.b]
        return np.concatenate(parts)

    def set_weights_flat(self, flat_vector):
        idx = 0
        new_W = []
        for i in range(self.n_layers):
            size = self.layer_sizes[i] * self.layer_sizes[i + 1]
            new_W.append(flat_vector[idx: idx + size].reshape(
                self.layer_sizes[i], self.layer_sizes[i + 1]))
            idx += size
        new_b = []
        for i in range(self.n_layers):
            size = self.layer_sizes[i + 1]
            new_b.append(flat_vector[idx: idx + size].reshape(1, size))
            idx += size
        self.W, self.b = new_W, new_b

    def n_params(self):
        total = 0
        for i in range(self.n_layers):
            total += self.layer_sizes[i] * self.layer_sizes[i + 1] + self.layer_sizes[i + 1]
        return total

    # ---------------- forward propagation ----------------
    def forward(self, X):
        activations = [X]
        zs = []
        A = X
        for i in range(self.n_layers):
            Z = A @ self.W[i] + self.b[i]
            zs.append(Z)
            if i == self.n_layers - 1:
                A = sigmoid(Z)
            else:
                A = relu(Z)
            activations.append(A)
        return activations, zs

    def predict_proba(self, X):
        activations, _ = self.forward(X)
        return activations[-1].flatten()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    # ---------------- back propagation ----------------
    def backward(self, activations, zs, y):
        """Manual back-propagation via the chain rule.
        dL/dZ_out = (a_out - y)                      [BCE + sigmoid combined gradient]
        dL/dW_i   = a_{i-1}^T . dZ_i / m  + l2 * W_i
        dL/db_i   = mean(dZ_i)
        dZ_{i-1}  = (dZ_i . W_i^T) * relu'(Z_{i-1})
        """
        m = y.shape[0]
        y = y.reshape(-1, 1)
        grads_W = [None] * self.n_layers
        grads_b = [None] * self.n_layers

        dZ = activations[-1] - y  # combined d(BCE)/d(sigmoid input)
        for i in reversed(range(self.n_layers)):
            A_prev = activations[i]
            grads_W[i] = (A_prev.T @ dZ) / m + self.l2 * self.W[i]
            grads_b[i] = np.mean(dZ, axis=0, keepdims=True)
            if i > 0:
                dA_prev = dZ @ self.W[i].T
                dZ = dA_prev * relu_derivative(zs[i - 1])
        return grads_W, grads_b

    def update_params(self, grads_W, grads_b):
        for i in range(self.n_layers):
            self.W[i] -= self.lr * grads_W[i]
            self.b[i] -= self.lr * grads_b[i]

    def train(self, X, y, X_val=None, y_val=None, epochs=300, batch_size=32,
              patience=25, verbose=False, random_state=42):
        rng = np.random.RandomState(random_state)
        n = X.shape[0]
        history = {"train_loss": [], "val_loss": []}
        best_val = np.inf
        best_weights = None
        wait = 0

        for epoch in range(epochs):
            perm = rng.permutation(n)
            X_shuf, y_shuf = X[perm], y[perm]
            for start in range(0, n, batch_size):
                xb = X_shuf[start:start + batch_size]
                yb = y_shuf[start:start + batch_size]
                activations, zs = self.forward(xb)
                gW, gb = self.backward(activations, zs, yb)
                self.update_params(gW, gb)

            train_pred = self.predict_proba(X)
            train_loss = binary_cross_entropy(y, train_pred)
            history["train_loss"].append(train_loss)

            if X_val is not None:
                val_pred = self.predict_proba(X_val)
                val_loss = binary_cross_entropy(y_val, val_pred)
                history["val_loss"].append(val_loss)
                if val_loss < best_val - 1e-5:
                    best_val = val_loss
                    best_weights = (
                        [w.copy() for w in self.W], [b.copy() for b in self.b])
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
            if verbose and epoch % 50 == 0:
                msg = f"Epoch {epoch}: train_loss={train_loss:.4f}"
                if X_val is not None:
                    msg += f" val_loss={val_loss:.4f}"
                print(msg)

        if best_weights is not None:
            self.W, self.b = best_weights
        return history
